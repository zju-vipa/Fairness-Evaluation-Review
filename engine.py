import math
import sys
import os
from os.path import join
from glob import glob

import torch

from clstool import build_evaluator
from clstool.utils.misc import update, reduce_dict, MetricLogger, SmoothedValue
from betaVAE.model import BetaVAE_H, BetaVAE_B
from AttGAN.attganG import AttGAN
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import time
import torchvision.utils as vutils



def train_one_epoch(model, criterion, data_loader, optimizer, lr_scheduler, device, epoch: int, max_norm: float = 0,
                    scaler=None, print_freq: int = 10, need_targets: bool = False):
    model.train()
    criterion.train()
    n_steps = len(data_loader)

    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    for batch_idx, (samples, targets, protected, filenames) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = targets.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            if need_targets:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets, training=True)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        update(optimizer, losses, model, max_norm, scaler)

        if hasattr(lr_scheduler, 'step_update'):
            lr_scheduler.step_update(epoch * n_steps + batch_idx)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        if 'class_error' in loss_dict_reduced.keys():
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

    lr_scheduler.step(epoch)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

    return stats

@torch.no_grad()
def evaluate(model, data_loader, criterion, device, args, print_freq=10, need_targets=False, amp=False):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if args.eval:
        evaluator = build_evaluator(args)
        if args.evaluator.lower() in ['fairness']:
            if args.pert_net == 'vae':
                net = BetaVAE_H(args.z_dim, 3)
                net.to(device)
                net.eval()
                load_checkpoint(net, args.ckpt_dir, args.viz_name, args.ckpt_name)
            elif args.pert_net == 'gan':
                net = AttGAN(args)  # with to(device)
                net.G.to(device)
                net.eval()
                net.load(find_model(join(args.ckpt_dir_gan, args.giz_name, 'checkpoint'), args.load_epoch))
            elif args.pert_net == 'none':  # for test
                pass

    iters = 0
    for samples, targets, protected, filenames in metric_logger.log_every(data_loader, print_freq, header):
        print(f"==>> iters: {iters}")
        iters += 1

        samples = samples.to(device)
        targets = targets.to(device)
        protected = protected.to(device)

        if args.eval:
            transform_m = transforms.Compose([
                    transforms.Resize((224, 224)),
                ])
            samples = transform_m(samples)

        outputs = model(samples)
        # vutils.save_image(samples, 'interp/image_o.png', nrow=1) 

        loss_dict = criterion(outputs, targets, training=False)
        weight_dict = criterion.weight_dict
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled,
                             )
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if args.eval:
            if iters<=args.pert_iter:  
                start_time = time.time()
                if args.pert_net == 'vae':
                    pert_scores = evaluator.dynamic_pert_vae(samples, model, net, args.z_dim, args.num_eps, args.decrement)
                elif args.pert_net == 'gan':
                    pert_scores = evaluator.dynamic_pert_gan(samples, protected, model, net, args.sub_attrs, args.pert_min, args.pert_max, args.pert_thres, args.pert_num)
                elif args.pert_net == 'none':   # for test
                    pert_scores = {'Tol': torch.zeros(args.z_dim), 'Dev': torch.zeros(args.z_dim)}
                end_time = time.time()
                time_cost = end_time - start_time
                print(f"==>> time_cost: {time_cost}")
            else:
                break

            evaluator.update(outputs, targets, protected, pert_scores)

    metric_logger.synchronize_between_processes()
    print('Averaged stats:', metric_logger)

    if args.eval:
        evaluator.synchronize_between_processes()
        evaluator.summarize(args.num_eps)
    
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

def load_checkpoint(net, ckpt_dir, viz_name, filename):
    file_path = os.path.join(ckpt_dir, viz_name, filename)
    if os.path.isfile(file_path):
        checkpoint = torch.load(file_path)
        net.load_state_dict(checkpoint['model_states']['net'])
        print("=> loaded checkpoint '{} (iter {})'".format(file_path, 1500000))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(file_path))
    
def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file

# def dynamic_pert(image, model, net, mu, logvar, num_eps, decrement, num_classes=2, max_iter=50):
#         print('==> dynamic_pert')
#         transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#         ])
#         image_r = F.sigmoid(net._decode(mu))  # B, 3, 64, 64
#         image = transform(image)  # B, 3, 224, 224
#         image_r = transform(image_r)  # B, 3, 224, 224

#         f_image = model.forward(Variable(image, requires_grad=True)).data.cpu().numpy()  
#         I = (np.array(f_image)).argsort()[:, ::-1]  
#         I = I[:, 0:num_classes]  
#         label = I[:, 0]  

#         f_image_r = model.forward(Variable(image_r, requires_grad=True)).data.cpu().numpy() 
#         I_r = (np.array(f_image_r)).argsort()[:, ::-1]  # B, out 
#         I_r = I_r[:, 0:num_classes]  # B, 2  
#         label_r = I_r[:, 0]  # B, 1  

#         threshold = torch.zeros(mu.shape[0], mu.shape[1], 2).long()  # B, z_dim, 2
#         Tol = torch.zeros(mu.shape[0], mu.shape[1], 1)   # B, z_dim, 1 
#         Dev = torch.zeros(mu.shape[0], mu.shape[1], 1)   # B, z_dim, 1
        
#         std = logvar.div(2).exp()  # B, z_dim
#         eps = mu.clone().unsqueeze(-1).repeat(1, 1, 2 * num_eps)  # B, z_dim, 2 * num_eps
#         for m in range(mu.shape[0]):
#             for n in range(mu.shape[1]):
#                 eps_front = torch.arange(0 - num_eps * decrement * std[m, n].item(), 0, decrement * std[m, n].item())
#                 eps_back = torch.arange(0, 0 + num_eps * decrement * std[m, n].item(), decrement * std[m, n].item())
#                 row_eps = torch.cat((eps_front, eps_back))
#                 eps[m, n, :] = row_eps  # B, z_dim, r_dim     middle->num_eps
        

#         for zi in range(mu.shape[1]):
#             eps_i = num_eps  # [mu, max) 
#             label_p = label_r  # B, 1   
#             while len(torch.nonzero(threshold[:, zi, 1])) != mu.shape[0] and eps_i < 2*num_eps: 
#                 z = mu.clone()                  # B, z_dim
#                 z[:, zi] = mu[:, zi] + eps[:, zi, eps_i]    # B, z_dim
#                 x_recon = F.sigmoid(net._decode(z))  # B, 3, 64, 64
#                 f_image_p = model.forward(transform(x_recon)).data.cpu().numpy()    # B, 2  
#                 I_p = (np.array(f_image_p)).argsort()[:, ::-1]  # B, out  
#                 I_p = I_p[:, 0:num_classes]  # B, 2  
#                 label_p = I_p[:, 0]  # B, 1  
#                 condition = torch.tensor(label_p != label_r) & (threshold[:, zi, 1] == 0)  # B, 1  
#                 threshold[torch.nonzero(condition).squeeze(), zi, 1] = eps_i  # B, 1 
#                 eps_i += 1
#             threshold[threshold[:, zi, 1]==0, zi, 1] = eps_i - 1
                
#             eps_i = num_eps - 1  # (min, mu)
#             label_p = label_r  
#             while len(torch.nonzero(threshold[:, zi, 0])) != mu.shape[0] and eps_i > 0: 
#                 z = mu.clone()                  # B, z_dim
#                 z[:, zi] = mu[:, zi] + eps[:, zi, eps_i]     # B, z_dim
#                 x_recon = F.sigmoid(net._decode(z))  # B, 3, 64, 64
#                 f_image_p = model.forward(transform(x_recon)).data.cpu().numpy()    # B, 2 
#                 I_p = (np.array(f_image_p)).argsort()[:, ::-1]  # B, out
#                 I_p = I_p[:, 0:num_classes]  # B, 2
#                 label_p = I_p[:, 0]  # B, 1 
#                 condition = torch.tensor(label_p != label_r) & (threshold[:, zi, 0] == 0)  # B, 1 
#                 threshold[torch.nonzero(condition).squeeze(), zi, 0] = eps_i  # B, 1  
#                 eps_i -= 1
#             threshold[threshold[:, zi, 0]==0, zi, 0] = eps_i + 1

#             left_tail = mu[:, zi].cpu() + (threshold[:, zi, 0] - num_eps) * decrement * std[:, zi].cpu()
#             right_tail = mu[:, zi].cpu() + (threshold[:, zi, 1] - num_eps) * decrement * std[:, zi].cpu()
#             Tol[:, zi]  =  ((right_tail - left_tail) / std[:, zi].cpu()).unsqueeze(1)  # B, 1  
#             Dev[:, zi] = (torch.abs((right_tail + left_tail) / 2 - mu[:, zi].cpu()) / std[:, zi].cpu()).unsqueeze(1)  # B, 1 

#         pert_scores = {'Tol': torch.sum(Tol, dim=0), 'Dev': torch.sum(Dev, dim=0)}
#         return pert_scores