import argparse
import copy
import datetime
import json
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.utils.data as Data
from termcolor import cprint

from engine import evaluate, train_one_epoch
from clstool import __version__, build_criterion, build_dataset, build_model, build_optimizer, build_scheduler
from clstool.utils.io import checkpoint_saver, checkpoint_loader, variables_loader, variables_saver
from clstool.utils.misc import makedirs, init_distributed_mode, init_seeds, is_main_process
from clstool.utils.plot_utils import plot_logs
from clstool.utils.record import Recorder
from clstool.utils.excelor import Excelor


def get_args_parser():
    parser = argparse.ArgumentParser('Disentangled Fairness', add_help=False)

    parser.add_argument('--config', '-c', type=str)

    # runtime
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', '-b', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--clip_max_norm', default=1.0, type=float, help='gradient clipping max norm')
    parser.add_argument('--eval', action='store_true', help='evaluate only')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--sync_bn', type=bool, default=False)
    parser.add_argument('--find_unused_params', action='store_true')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--need_targets', action='store_true', help='need targets for training')
    parser.add_argument('--drop_lr_now', action='store_true')
    parser.add_argument('--drop_last', action='store_true')
    parser.add_argument('--amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('--no_dist', action='store_true', help='forcibly disable distributed mode')

    # dataset
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--dataset', '-d', type=str, default='celeba')
    parser.add_argument('--main_attr', type=str, default='Attractive')
    parser.add_argument('--sub_attrs', type=str, nargs='*', default=["Bald", "Eyeglasses", "Mouth_Slightly_Open", "No_Beard", "Pale_Skin", "5_o_Clock_Shadow", "High_Cheekbones", "Blurry", "Bangs", "Narrow_Eyes"])
    # "Bags_Under_Eyes", "Double_Chin", "Rosy_Cheeks", "Arched_Eyebrows", "Young"
    # "Goatee", "Mustache", "Big_Lips", "Big_Nose", "Male"
    # "Sideburns", "Bushy_Eyebrows", "Oval_Face", "Heavy_Makeup", "Smiling"
    # "Pointy_Nose", "Receding_Hairline", "Wearing_Hat", "Wearing_Lipstick", "Straight_Hair"

    # data augmentation
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--train_aug_kwargs', default=dict())
    parser.add_argument('--eval_aug_kwargs', default=dict())
    parser.add_argument('--train_batch_aug_kwargs', default=dict())
    parser.add_argument('--eval_batch_aug_kwargs', default=dict())
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='for LabelSmoothingCrossEntropy')

    # model
    parser.add_argument('--model_lib', default='default', type=str, choices=['default', 'timm'], help='model library')
    parser.add_argument('--model', '-m', default='resnet18', type=str, help='model name')
    parser.add_argument('--model_kwargs', default=dict(), help='model specific kwargs')
    parser.add_argument('--freeze', action='store_true', help='freeze the feature layer')

    # criterion
    parser.add_argument('--criterion', default='default', type=str, help='criterion name')

    # optimizer
    parser.add_argument('--optimizer', default='adamw', type=str, help='optimizer name')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_drop', default=-1, type=int)
    parser.add_argument('--momentum', default=0.9, type=float, help='for SGD')
    parser.add_argument('--weight_decay', default=5e-2, type=float)

    # lr_scheduler
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler name')
    parser.add_argument('--warmup_epochs', default=0, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--warmup_lr', default=1e-6, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float, help='for CosineLR')
    parser.add_argument('--step_size', type=int, help='for StepLR')
    parser.add_argument('--milestones', type=int, nargs='*', help='for MultiStepLR')
    parser.add_argument('--gamma', default=0.1, type=float, help='for StepLR and MultiStepLR')

    # evaluator
    parser.add_argument('--evaluator', '-e', default='fairness', type=str, help='evaluator name')

    # loading weights
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--resume', '-r', default='false', type=str, help='"/nfs/jhz/CV/Disentangled_Fairness/runs/" + args.main_attr + "/" + args.model + "_" + args.dataset + "/checkpoint0029.pth"')
    parser.add_argument('--load_pos', type=str)

    # saving weights
    parser.add_argument('--output_dir', '-o', type=str, default='./runs/__tmp__')
    parser.add_argument('--save_interval', type=int, default=1)
    parser.add_argument('--save_pos', type=str)

    # remarks
    parser.add_argument('--note', type=str)
    parser.add_argument('--xls_dir', default='./result/xls/Fairxls.xlsx', type=str)

    # vae
    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--viz_name', default='main', type=
    str, help='visdom env name')
    parser.add_argument('--ckpt_name', default='last', type=str, help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--ckpt_dir', default='betaVAE/checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--num_eps', default=100, type=int, help='dimension of the num_eps')
    parser.add_argument('--decrement', default=0.5, type=int, help='dimension of the decrement(0,1)')

    # gan
    parser.add_argument('--ckpt_dir_gan', default='AttGAN/output', type=str, help='checkpoint directory')
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
        
    parser.add_argument('--n_attrs', default=13, type=int, help='dimension of the representation z')
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=1)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--pert_min', default=-1, type=int)
    parser.add_argument('--pert_max', default=1, type=int)
    parser.add_argument('--pert_thres', default=0.5, type=int)
    parser.add_argument('--pert_num', default=20, type=int)


    # pert
    parser.add_argument('--pert_iter', default=10, type=int, help='dimension of the num_eps')
    parser.add_argument('--pert_net', default='vae', type=str, choices=['vae', 'gan', 'none'], help='pert_net library')

    return parser


def main(args):
    init_seeds(args.seed)
    init_distributed_mode(args)
    cprint(f'Disentangled Fairness v{__version__}', 'light_green', attrs=['bold'])
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu' or args.eval:
        args.amp = False
    if args.num_workers is None:
        args.num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    if args.resume != "false":
        args.no_pretrain = True
    if args.note is None:
        args.note = f'dataset: {args.dataset} | model: {args.model} | output_dir: {args.output_dir}'
    output_dir = Path(args.output_dir)

    print(args)
    __args__ = copy.deepcopy(vars(args))
    ignored_args = ['config', 'eval', 'local_rank', 'start_epoch']
    if args.distributed:
        ignored_args += ['rank', 'gpu']
    for ignored_arg in ignored_args:
        pop_info = __args__.pop(ignored_arg, KeyError)
        if pop_info is KeyError:
            cprint(f"Warning: The argument '{ignored_arg}' to be ignored is not in 'args'.", 'light_yellow')
    __args__ = {k: v for k, v in sorted(__args__.items(), key=lambda item: item[0])}
    variables_saver(__args__, os.path.join(args.output_dir, 'config.py'))

    # ** model **
    model = build_model(args)
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of params: {n_parameters}')

    # ** optimizer **
    param_dicts = [
        {'params': [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]},
    ]
    optimizer = build_optimizer(args, param_dicts)

    # ** criterion **
    criterion = build_criterion(args)

    # ** dataset **
    if not args.eval:
        dataset_train = build_dataset(args, split='train')
        dataset_val = build_dataset(args, split='val')

        if args.distributed:
            sampler_train = Data.distributed.DistributedSampler(dataset=dataset_train, shuffle=True)
            sampler_val = Data.distributed.DistributedSampler(dataset=dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        data_loader_train = Data.DataLoader(dataset=dataset_train,
                                            sampler=sampler_train,
                                            batch_size=args.batch_size,
                                            drop_last=bool(args.drop_last or len(dataset_train) % 2 or args.batch_size % 2),
                                            pin_memory=args.pin_memory,
                                            num_workers=args.num_workers,
                                            collate_fn=dataset_train.collate_fn)
        data_loader_val = Data.DataLoader(dataset=dataset_val,
                                      sampler=sampler_val,
                                      batch_size=args.batch_size,
                                      pin_memory=args.pin_memory,
                                      num_workers=args.num_workers,
                                      collate_fn=dataset_val.collate_fn)
    else:
        dataset_test = build_dataset(args, split='test')
        if args.distributed:
            sampler_test = Data.distributed.DistributedSampler(dataset=dataset_test, shuffle=True)
        else:
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
        data_loader_test = Data.DataLoader(dataset=dataset_test,
                                            sampler=sampler_test,
                                            batch_size=args.batch_size,
                                            pin_memory=args.pin_memory,
                                            num_workers=args.num_workers,
                                            collate_fn=dataset_test.collate_fn,
                                            shuffle=False)

                                      

    # ** scheduler **
    if not args.eval:
        lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    # ** scaler **
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    if args.resume != "false":
        print('Loading model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location='cpu')
        checkpoint_loader(model_without_ddp, checkpoint['model'], delete_keys=())
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            checkpoint_loader(optimizer, checkpoint['optimizer'])
            checkpoint_loader(lr_scheduler, checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if args.drop_lr_now:  # only works when using StepLR or MultiStepLR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1
        if scaler and 'scaler' in checkpoint:
            checkpoint_loader(scaler, checkpoint["scaler"])

        if args.freeze:
            print('Freezeing')
            args.start_epoch = 0
            args.epochs = math.floor(args.epochs / 3)
            for name, param in model.named_parameters():
                # print(name, param.size())
                if not name.startswith("fc.") and not name.startswith("classifier.") and not name.startswith("head."):
                    param.requires_grad_(False) 


    if args.eval:
        print("eval")
        test_stats = evaluate(
            model, data_loader_test, criterion, device, args, args.print_freq, args.need_targets, args.amp
        )
        return

    print('\n' + 'Start training:')
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, lr_scheduler, device, epoch, args.clip_max_norm, scaler,
            args.print_freq, args.need_targets
        )
        if args.output_dir and (epoch + 1) % args.save_interval == 0:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if scaler:
                    checkpoint['scaler'] = scaler.state_dict()
                checkpoint_saver(checkpoint, checkpoint_path)

        test_stats = evaluate(
            model, data_loader_val, criterion, device, args, args.print_freq, args.need_targets, args.amp
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and is_main_process():
            log_path = output_dir / 'log.txt'
            log_exists = True if log_path.exists() else False
            with log_path.open('a') as f:
                f.write(json.dumps(log_stats) + '\n')
            if not log_exists:
                log_path.chmod(mode=0o777)

        if args.note:
            print(f'Note: {args.note}\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    recoder = Recorder()
    log_file = './result/txt/exp_logs.txt'
    recoder.record_log(log_file, sys.argv)

    parser = argparse.ArgumentParser('Disentangled Fairness', parents=[get_args_parser()])
    argv = sys.argv[1:]
    
    idx = argv.index('-c') if '-c' in argv else (argv.index('--config') if '--config' in argv else -1)
    if idx not in [-1, len(argv) - 1] and not argv[idx + 1].startswith('-'):
        idx += 1

    sys.argv[1:] = argv[:idx + 1]
    args = parser.parse_args()

    if args.config:
        cfg = variables_loader(args.config)
        for k, v in cfg.items():
            setattr(args, k, v)

    sys.argv[1:] = argv[idx + 1:]
    args = parser.parse_args(namespace=args)

    if args.data_root:
        makedirs(args.data_root, exist_ok=True)
    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        else:
            if not args.eval:
                start_time = datetime.datetime.now().replace(microsecond=0) + datetime.timedelta(hours=8)
                os.makedirs(args.output_dir + '_' + str(start_time), exist_ok=True)

    args.viz_name = 'celeba_H_beta10_z' + str(args.z_dim)
    args.giz_name = '128_shortcut1_inject1_celeba_' + str(args.z_dim)
    args.n_attrs = args.z_dim

    main(args)

    recoder.check_finished(log_file)
    
