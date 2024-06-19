import warnings
import torch
from clstool.utils.excelor import save_datas_to_xls
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision.utils as vutils

warnings.filterwarnings('ignore')


class FairnessEvaluator:
    def __init__(self, static_metrics: list, dynamic_metrics: list, args):
        self.static_metrics = static_metrics
        self.dynamic_metrics = dynamic_metrics
        
        # check whether sa_num is equal
        if len(args.sub_attrs) != args.z_dim:
            raise ValueError("len(args.sub_attrs) and args.z_dim are not equal")
        self.sa_num = args.z_dim
        # init metrics
        self.matrix_ = {
            'sum': 0, 'T': 0, 's0': 0, 's1': 0,
            'T_s0': 0, 'T_s1': 0,
            'p0_s0': 0, 'p0_s1': 0, 'p1_s0': 0, 'p1_s1': 0,
            't0_s0': 0, 't0_s1': 0, 't1_s0': 0, 't1_s1': 0,
            'p0_t0_s0': 0, 'p0_t0_s1': 0, 'p0_t1_s0': 0, 'p0_t1_s1': 0,
            'p1_t0_s0': 0, 'p1_t0_s1': 0, 'p1_t1_s0': 0, 'p1_t1_s1': 0
        }
        self.matrixs = [dict(self.matrix_) for _ in range(self.sa_num)]
        self.static = {metric: torch.zeros(self.sa_num) for metric in static_metrics}
        self.dynamic = {metric: torch.zeros(self.sa_num) for metric in dynamic_metrics}
        self.pert_sum = args.batch_size * args.pert_iter

        # init excelor
        self.xls_dir = args.xls_dir
        self.model = args.model
        self.dataset = args.dataset

        if(args.z_dim != 1):
            task_ = {'Attractive': 'A', 'Eyeglasses': 'B', 'Pale_Skin': 'C',  # for celeba
                    'Letter': 'A',  # for fonts-v1
                    }
            sa_scale_ = {'5': 'small', '10': 'medium', '15': 'large',
                        '1': 'small',
                        }  # for both static and dynamic
            self.task = task_[args.main_attr]
            self.sa_scale = sa_scale_[str(self.sa_num)]
        else:
            self.main_attr = args.main_attr
            self.sub_attr = args.sub_attrs[0]

        # self.Fairxls = Excelor()
        # self.Fairxls.read_xls(args.xls_dir)
        # print('==>> Fairxls.df_info:\n',self.Fairxls.df_info)

        self.label_changed = 0
        self.label_total = 0
        self.pert_num = 0




    def update_static(self, outputs, targets, sensitivitys):
        if isinstance(outputs, dict):
            assert 'logits' in outputs.keys(), \
                f"When using 'update(self, outputs, targets)' in '{self.__class__.__name__}', " \
                f"if 'outputs' is a dict, 'logits' MUST be the key."
            outputs = outputs['logits']
        outputs = outputs.max(1)[1]

        for attr in range(self.sa_num):
            if (self.sa_num == 1):
                sensitivity = sensitivitys
            else:
                sensitivity = sensitivitys[:, attr]
            self.matrix_['sum'] = 1.*targets.size(0) 

            self.matrix_['T'] = (outputs == targets).float().sum() # TP+TN
            self.matrix_['s0'] = (sensitivity == 0).float().sum()  # N_s
            self.matrix_['s1'] = (sensitivity == 1).float().sum()  # P_s
            self.matrix_['T_s0'] = ((outputs == targets) & (sensitivity == 0)).float().sum()  # TN+TP | N_s 
            self.matrix_['T_s1'] = ((outputs == targets) & (sensitivity == 1)).float().sum()  # TN+TP | P_s

            # For all prerequisites
            self.matrix_['p0_s0'] = ((outputs == 0) & (sensitivity == 0)).float().sum()  # N_o | N_s
            self.matrix_['p0_s1'] = ((outputs == 0) & (sensitivity == 1)).float().sum()  # N_o | P_s
            self.matrix_['p1_s0'] = ((outputs == 1) & (sensitivity == 0)).float().sum()  # P_o | N_s
            self.matrix_['p1_s1'] = ((outputs == 1) & (sensitivity == 1)).float().sum()  # P_o | P_s
            self.matrix_['t0_s0'] = ((targets == 0) & (sensitivity == 0)).float().sum()  # N_r | N_s
            self.matrix_['t0_s1'] = ((targets == 0) & (sensitivity == 1)).float().sum()  # N_r | P_s
            self.matrix_['t1_s0'] = ((targets == 1) & (sensitivity == 0)).float().sum()  # P_r | N_s
            self.matrix_['t1_s1'] = ((targets == 1) & (sensitivity == 1)).float().sum()  # P_r | P_s

            # For all possible scenarios
            self.matrix_['p0_t0_s0'] = ((outputs == 0) & (targets == 0) & (sensitivity == 0)).float().sum()  # N_o | N_r | N_s
            self.matrix_['p0_t0_s1'] = ((outputs == 0) & (targets == 0) & (sensitivity == 1)).float().sum()  # N_o | N_r | P_s
            self.matrix_['p0_t1_s0'] = ((outputs == 0) & (targets == 1) & (sensitivity == 0)).float().sum()  # N_o | P_r | N_s
            self.matrix_['p0_t1_s1'] = ((outputs == 0) & (targets == 1) & (sensitivity == 1)).float().sum()  # N_o | P_r | P_s
            self.matrix_['p1_t0_s0'] = ((outputs == 1) & (targets == 0) & (sensitivity == 0)).float().sum()  # P_o | N_r | N_s
            self.matrix_['p1_t0_s1'] = ((outputs == 1) & (targets == 0) & (sensitivity == 1)).float().sum()  # P_o | N_r | P_s
            self.matrix_['p1_t1_s0'] = ((outputs == 1) & (targets == 1) & (sensitivity == 0)).float().sum()  # P_o | P_r | N_s
            self.matrix_['p1_t1_s1'] = ((outputs == 1) & (targets == 1) & (sensitivity == 1)).float().sum()  # P_o | P_r | P_s
            
            for key in set(self.matrixs[attr]) | set(self.matrix_):
                self.matrixs[attr][key] = self.matrixs[attr].get(key, 0) + self.matrix_.get(key, 0)

    def update_dynamic(self, pert_scores):
        self.dynamic['Tol'] += pert_scores['Tol']
        self.dynamic['Dev'] += pert_scores['Dev']

    def update(self, outputs, targets, sensitivitys, pert_scores):
        self.update_static(outputs, targets, sensitivitys)
        self.update_dynamic(pert_scores)

    def dynamic_pert_vae(self, image, model, net, z_dim, num_eps, decrement, num_classes=2, max_iter=50):
        self.pert_num = num_eps
        transform = transforms.Compose([
                transforms.Resize((64, 64)),
            ])
        image = transform(image)  # B, z, r, 3, 64, 64
        z = net._encode(image)  # B, z_dim*2
        mu, logvar = z[:, :z_dim], z[:, z_dim:]  # B, z_dim 

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        image_r = F.sigmoid(net._decode(mu))  # B, 3, 64, 64

        image = transform(image)  # B, 3, 224, 224
        image_r = transform(image_r)  # B, 3, 224, 224
        # vutils.save_image(image, 'interp/image.png', nrow=1) 
        # vutils.save_image(image_r, 'interp/image_r.png', nrow=1) 

        f_image = model.forward(Variable(image, requires_grad=True)).data.cpu().numpy()  
        I = (np.array(f_image)).argsort()[:, ::-1]  
        label = I[:, 0]  

        f_image_r = model.forward(Variable(image_r, requires_grad=True)).data.cpu().numpy() 
        I_r = (np.array(f_image_r)).argsort()[:, ::-1]  # B, out 
        label_r = I_r[:, 0]  # B, 1  

        self.label_changed += np.sum(label != label_r)
        self.label_total += image.shape[0]

        threshold = torch.zeros(2 *mu.shape[0], mu.shape[1]).long()  # 2*B, z_dim
        Tol = torch.zeros(mu.shape[0], mu.shape[1])   # B, z_dim 
        Dev = torch.zeros(mu.shape[0], mu.shape[1])   # B, z_dim
        
        std = logvar.div(2).exp()  # B, z_dim
        eps = mu.clone().unsqueeze(-1).repeat(1, 1, 2 * num_eps)  # B, z_dim, 2*num_eps, num_eps=100
        for m in range(mu.shape[0]):  # B
            for n in range(mu.shape[1]):  # z_dim
                eps_front = torch.arange(0 - num_eps * decrement * std[m, n].item(), 0, decrement * std[m, n].item())
                eps_back = torch.arange(decrement * std[m, n].item(), (1 + num_eps) * decrement * std[m, n].item(), decrement * std[m, n].item())
                row_eps = torch.cat((eps_front, eps_back))  # 2*num_eps
                eps[m, n, :] = row_eps  # B, z_dim, r_dim(2*num_eps)

        for zi in range(mu.shape[1]):
            eps_i = 0  # [mu, max) 
            label_rr = np.concatenate((label_r, label_r), axis=0)   
            while eps_i < num_eps: 
                z_f = mu.clone()  # B, z_dim
                z_b = mu.clone()  # B, z_dim
                z_f[:, zi] = mu[:, zi] + eps[:, zi, (num_eps - 1) - eps_i]    # B, z_dim
                z_b[:, zi] = mu[:, zi] + eps[:, zi, num_eps + eps_i]    # B, z_dim
                z = torch.cat((z_f, z_b), dim=0)  # 2*B, z_dim
                x_recon = F.sigmoid(net._decode(z))  # 2*B, 3, 64, 64

                f_image_p = model.forward(transform(x_recon)).data.cpu().numpy()    # 2*B, out 
                I_p = (np.array(f_image_p)).argsort()[:, ::-1]  # 2*B, out   
                label_p = I_p[:, 0]  # 2*B
                # print(f"==>> label_p_att{zi}_pert{eps_i}: {label_p}")

                condition = torch.tensor(label_p != label_rr) & (threshold[:, zi] == 0)  # 2*B, 1  
                threshold[torch.nonzero(condition).squeeze(), zi] = eps_i  # 2*B, 1 
                eps_i += 1
            threshold[threshold[:, zi]==0, zi] = eps_i - 1
            print(f"==>> zi: {zi}")

            left_tail = mu[:, zi].cpu() - threshold[:mu.shape[0], zi] * decrement * std[:, zi].cpu()  # B
            right_tail = mu[:, zi].cpu() + threshold[mu.shape[0]:, zi] * decrement * std[:, zi].cpu()  # B
            Tol[:, zi]  =  ((right_tail - left_tail) / std[:, zi].cpu())  # B
            Dev[:, zi] = (torch.abs((right_tail + left_tail) / 2 - mu[:, zi].cpu()) / std[:, zi].cpu())  # B

        pert_scores = {'Tol': torch.sum(Tol, dim=0), 'Dev': torch.sum(Dev, dim=0)}  # z_dim,  z_dim

        return pert_scores
        
    def dynamic_pert_gan(self, img_a, att_a, model, net, sub_attrs, pert_min, pert_max, thres_int, pert_num):
        print(f"==>> sub_attrs: {sub_attrs}")
        self.pert_num = pert_num
        test_att = sub_attrs

        transform_n = transforms.Compose([
                transforms.CenterCrop(170),
                transforms.Resize((128, 128)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        transform_m = transforms.Compose([
                transforms.Resize((224, 224)),
            ])

        f_img_a = model.forward(Variable(img_a, requires_grad=True)).data.cpu().numpy()  
        I = (np.array(f_img_a)).argsort()[:, ::-1]  
        label_none = I[:, 0]  

        img_a = transform_n(img_a)  # B, 3, 128, 128
        att_b = att_a.clone().type(torch.float)  # B, n_attr

        thres_pert = torch.zeros(img_a.shape[0], len(sub_attrs)).long()  # B, n_attr
        
        label_o = torch.zeros(img_a.shape[0])
        for att_i in range(len(test_att)):
            # samples = [img_a]
            # print(f"==>> att_a: {att_a[:, att_i]} # B")
            # print(f"==>> out_original: {label_none} # B")
            for pert_j in range(pert_num):
                test_int = (pert_max - pert_min) / (pert_num - 1) * pert_j + pert_min
                att_b_ = (att_b * 2 - 1) * thres_int  # len(sub_attrs) # [0,1] -> [0,2] -> [-1,1] -> [-thres, thres]
                att_b_[..., sub_attrs.index(test_att[att_i])] = test_int
                x_recon = net.G(img_a, att_b_)
                
                f_image_p = model.forward(transform_m(x_recon)).data.cpu().numpy()    # B, out
                I_p = (np.array(f_image_p)).argsort()[:, ::-1]  # B, out   
                label_p = I_p[:, 0]  # B
                if pert_j == 0:
                    label_o = label_p
                # print(f"==>> out_{test_att[att_i]}_p{pert_j}: {label_p}")

                condition = torch.tensor(label_p != label_o) & (thres_pert[:, att_i] == 0)  # 2*B, 1 
                thres_pert[torch.nonzero(condition).squeeze(), att_i] = pert_j  # 2*B, 1 

                # samples.append(x_recon)
            thres_pert[thres_pert[:, att_i]==0, att_i] = pert_num - 1
            # print(f"==>> thres_pert: {thres_pert}")

            # samples = torch.cat(samples, dim=3)
            # vutils.save_image( samples, f'interp/image_gan_{test_att[att_i]}.png', nrow=1, normalize=True, value_range=(-1., 1.) )
            # print('save_image done!')

        # thres_att = ((att_b * 2 - 1) * thres_int - pert_min) / ((pert_max - pert_min) / (pert_num - 1))   # B, n_attr
        # print(f"==>> thres_att: {thres_att.T}")  # n_attr, B

        # print(f"==>> att_a: {att_a.T.to('cpu')}")
        # print(f"==>> thres_pert: {thres_pert.T}")

        Tol = np.abs((pert_num - 1) - thres_pert.T - (1 - att_a.T.to('cpu')) * (pert_num - 1))
        Tol[Tol==0] = pert_num - 1  # n_attr, B 
        Dev = np.abs(thres_pert.T - (pert_num - 1) / 2)  # n_attr, B 
        # print(f"==>> Tol: {Tol}")
        # print(f"==>> Dev: {Dev}")

        pert_scores = {'Tol': torch.sum(Tol.T, dim=0), 'Dev': torch.sum(Dev.T, dim=0)}  # out -> B, n_attr
        return pert_scores


    def synchronize_between_processes(self):
        pass

    @staticmethod
    def metric_ACC(self, attr):
        ACC = self.matrixs[attr]['T'].float()/self.matrixs[attr]['sum'] * 100  # ACC = (TN+TP)/(TN+TP+FN+FP)
        return ACC.item()

    @staticmethod
    def metric_2ed_indicators(self, attr):
        d_PPV = abs(self.matrixs[attr]['p1_t1_s0'] / (self.matrixs[attr]['p1_s0'] + 1e-6) - self.matrixs[attr]['p1_t1_s1'] / (self.matrixs[attr]['p1_s1'] + 1e-6)) * 100  # PPV = TP / (TP + FP)
        d_NPV = abs(self.matrixs[attr]['p0_t0_s0'] / (self.matrixs[attr]['p0_s0'] + 1e-6) - self.matrixs[attr]['p0_t0_s1'] / (self.matrixs[attr]['p0_s1'] + 1e-6)) * 100  # NPV = TN / (TN + FN)
        d_TPR = abs(self.matrixs[attr]['p1_t1_s0'] / (self.matrixs[attr]['t1_s0'] + 1e-6) - self.matrixs[attr]['p1_t1_s1'] / (self.matrixs[attr]['t1_s1'] + 1e-6)) * 100  # TPR = TP / (TP + FN)
        d_TNR = abs(self.matrixs[attr]['p0_t0_s0'] / (self.matrixs[attr]['t0_s0'] + 1e-6) - self.matrixs[attr]['p0_t0_s1'] / (self.matrixs[attr]['t0_s1'] + 1e-6)) * 100  # TNR = TN / (TN + FP)
        d_FPR = abs(self.matrixs[attr]['p1_t0_s0'] / (self.matrixs[attr]['t0_s0'] + 1e-6) - self.matrixs[attr]['p1_t0_s1'] / (self.matrixs[attr]['t0_s1'] + 1e-6)) * 100  # FPR = FP / (FP + TN)
        d_FNR = abs(self.matrixs[attr]['p0_t1_s0'] / (self.matrixs[attr]['t1_s0'] + 1e-6) - self.matrixs[attr]['p0_t1_s1'] / (self.matrixs[attr]['t1_s1'] + 1e-6)) * 100  # FNR = FN / (FN + TP)
        d_FDR = abs(self.matrixs[attr]['p1_t0_s0'] / (self.matrixs[attr]['p1_s0'] + 1e-6) - self.matrixs[attr]['p1_t0_s1'] / (self.matrixs[attr]['p1_s1'] + 1e-6)) * 100  # FDR = FP / (TP + FP)
        d_FOR = abs(self.matrixs[attr]['p0_t1_s0'] / (self.matrixs[attr]['p0_s0'] + 1e-6) - self.matrixs[attr]['p0_t1_s1'] / (self.matrixs[attr]['p0_s1'] + 1e-6)) * 100  # FOR = FN / (TN + FN)
        return d_PPV.item(), d_NPV.item(), d_TPR.item(), d_TNR.item(), d_FPR.item(), d_FNR.item(), d_FDR.item(), d_FOR.item()

    @staticmethod
    # For DDP metric  -- Demographic Disparity in Predicted Positive: DDP = | FPRs0 - FPRs1 |
    def metric_DP(self, attr):
        d_DP = abs(self.matrixs[attr]['p1_s0'] / (self.matrixs[attr]['s0'] + 1e-6) - self.matrixs[attr]['p1_s1'] / (self.matrixs[attr]['s1'] + 1e-6)) * 100
        return d_DP.item()

    @staticmethod
    def metric_EOpp(self, attr):
        d_FNR = abs(self.matrixs[attr]['p0_t1_s0'] / (self.matrixs[attr]['t1_s0'] + 1e-6) - self.matrixs[attr]['p0_t1_s1'] / (self.matrixs[attr]['t1_s1'] + 1e-6)) * 100  # FNR = FN / (FN + TP)
        d_Eopp = d_FNR
        return d_Eopp.item()
    
    @staticmethod
    def metric_EOdd(self, attr):
        d_TPR = abs(self.matrixs[attr]['p1_t1_s0'] / (self.matrixs[attr]['t1_s0'] + 1e-6) - self.matrixs[attr]['p1_t1_s1'] / (self.matrixs[attr]['t1_s1'] + 1e-6)) * 100  # TPR = TP / (TP + FN)
        d_FPR = abs(self.matrixs[attr]['p1_t0_s0'] / (self.matrixs[attr]['t0_s0'] + 1e-6) - self.matrixs[attr]['p1_t0_s1'] / (self.matrixs[attr]['t0_s1'] + 1e-6)) * 100  # FPR = FP / (FP + TN)
        d_Eodd = max(d_TPR, d_FPR) 
        return d_Eodd.item()
    
    @staticmethod
    def metric_AOdd(self, attr):
        d_FPR = abs(self.matrixs[attr]['p1_t0_s0'] / (self.matrixs[attr]['t0_s0'] + 1e-6) - self.matrixs[attr]['p1_t0_s1'] / (self.matrixs[attr]['t0_s1'] + 1e-6)) * 100  # FPR = FP / (FP + TN)
        d_TPR = abs(self.matrixs[attr]['p1_t1_s0'] / (self.matrixs[attr]['t1_s0'] + 1e-6) - self.matrixs[attr]['p1_t1_s1'] / (self.matrixs[attr]['t1_s1'] + 1e-6)) * 100  # TPR = TP / (TP + FN)
        d_Aodd = 0.5*(d_TPR + d_FPR)
        return d_Aodd.item()

    def summarize(self, num_eps):
        print('Fairness Classification Metrics:')
        # compute intra-class fairness x self.sa_num
        saved_datas = []
        for metric in self.static_metrics:
            for attr in range(self.sa_num):
                value = getattr(self, f'metric_{metric}')(self, attr)
                self.static[metric][attr] = round(value, 2)
            score = torch.mean(self.static[metric]).item()
            # saved_datas.append([self.model, self.dataset, self.task, self.sa_scale, metric, round(score, 2)])
            saved_datas.append([self.model, self.dataset, self.main_attr, self.sub_attr, metric, round(score, 2)])

        # compute inter-class fairness x self.sa_num
        self.dynamic['Tol'] = self.dynamic['Tol'] / (self.pert_sum * (self.pert_num - 1)) * 100
        self.dynamic['Dev'] = self.dynamic['Dev'] / (self.pert_sum * (self.pert_num - 1)) * 100

        # compute intra-attribute fairness x self.sa_num
        coupling = 0
        for zi in range(self.sa_num):
            if self.dynamic['Tol'][zi].item() < (0.9 * 100):
                coupling += 1
        self.dynamic['Cou'] = torch.tensor(coupling / self.sa_num) * 100
        
        for metric in self.dynamic_metrics:
            score = torch.mean(self.dynamic[metric]).item()
            # saved_datas.append([self.model, self.dataset, self.task, self.sa_scale, metric, round(score, 2)])
        
        # save datas to xls
        list(map(print, saved_datas))
        save_datas_to_xls(saved_datas, self.xls_dir)

        print(f"==>> self.label_changed / self.label_total: {self.label_changed} / {self.label_total}")












        # # compute inter-class fairness x self.sa_num
        # self.dynamic['Tol'] = self.dynamic['Tol'] / self.pert_sum
        # self.dynamic['Dev'] = self.dynamic['Dev'] / self.pert_sum

        # # compute intra-attribute fairness x self.sa_num
        # coupling = 0
        # for zi in range(self.sa_num):
        #     if self.dynamic['Tol'][zi].item() < (0.9 * self.pert_num):
        #         coupling += 1
        # self.dynamic['Cou'] = torch.tensor(coupling / self.sa_num)