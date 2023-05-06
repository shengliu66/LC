from tqdm import tqdm
import wandb
import numpy as np
import torch
import torch.nn as nn

import os
import torch.optim as optim

from data.util import dataloader
from module.lc_loss.loss import GeneralizedCELoss, LogitCorrectionLoss
from module.lc_loss.group_mixup import group_mixUp
from module.util import get_model
from util import EMA, EMA_squre, sigmoid_rampup, get_bert_scheduler

import torch.nn.functional as F
import random


import matplotlib.pyplot as plt
import matplotlib.cm as cmap

def cycle(iterable):
    while True:
        for x in iterable:
            yield x



class Learner(nn.Module):
    def __init__(self, args):
        super(Learner, self).__init__()
        data2model = {'cmnist': "MLP",
                       'cifar10c': "ResNet18",
                       'bffhq': "ResNet18",
                       'waterbird': "resnet_50",
                       'civilcomments': "bert-base-uncased_pt"}

        data2batch_size = {'cmnist': 256,
                           'cifar10c': 256,
                           'bffhq': 64,
                           'waterbird': 32,
                           'civilcomments': 16}
        
        data2preprocess = {'cmnist': None,
                           'cifar10c': True,
                           'bffhq': True,
                           'waterbird': True,
                           'civilcomments': None}

        self.data2preprocess = data2preprocess
        self.data2batch_size = data2batch_size


        args.exp = '{:s}_ema_{:.2f}_tau_{:.2f}_lambda_{:.2f}_avgtype_{:s}'.format(args.exp, args.ema_alpha, args.tau, args.lambda_dis_align, args.avg_type)

        if args.wandb:
            import wandb
            wandb.init(project='Learning-with-Logit-Correction')
            wandb.run.name = args.exp

        run_name = args.exp
        if args.tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(f'result/summary/{run_name}')

        self.model = data2model[args.dataset]
        self.batch_size = data2batch_size[args.dataset]

        print(f'model: {self.model} || dataset: {args.dataset}')
        print(f'working with experiment: {args.exp}...')
        self.log_dir = os.makedirs(os.path.join(args.log_dir, args.dataset, args.exp), exist_ok=True)
        self.device = torch.device(args.device)
        self.args = args

        # logging directories
        self.log_dir = os.path.join(args.log_dir, args.dataset, args.exp)
        self.summary_dir =  os.path.join(args.log_dir, args.dataset, "summary", args.exp)
        self.summary_gradient_dir = os.path.join(self.log_dir, "gradient")
        self.result_dir = os.path.join(self.log_dir, "result")
        self.plot_dir = os.path.join(self.log_dir, "figure")
        os.makedirs(self.summary_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.loader = dataloader(
            args.dataset, 
            args.data_dir, 
            args.percent, 
            data2preprocess, 
            args.use_type0, 
            args.use_type1, 
            self.batch_size, 
            args.num_workers
            )

        self.train_loader, self.train_dataset = self.loader.run('train')

        self.valid_loader = self.loader.run('valid')
        self.test_loader = self.loader.run('test')


        if args.dataset == 'waterbird' or args.dataset == 'civilcomments':
            train_target_attr = self.train_dataset.y_array
            train_target_attr = torch.LongTensor(train_target_attr)
        else:
            train_target_attr = []
            for data in self.train_dataset.data:
                train_target_attr.append(int(data.split('_')[-2]))
            train_target_attr = torch.LongTensor(train_target_attr)

        attr_dims = []
        attr_dims.append(torch.max(train_target_attr).item() + 1)
        self.num_classes = attr_dims[0]
        num_example = len(train_target_attr)
        print('Num example in training is {:d}, Num classes is {:d} \n'.format(num_example, self.num_classes))


        self.sample_margin_ema_b = EMA(torch.LongTensor(train_target_attr), num_classes=self.num_classes, alpha=0)
        self.confusion = EMA_squre(num_classes=self.num_classes, alpha=args.ema_alpha, avg_type = args.avg_type)
        print(f'alpha : {self.sample_margin_ema_b.alpha}')

        self.best_valid_acc_b, self.best_test_acc_b = 0., 0.
        self.best_valid_acc_d, self.best_test_acc_d = 0., 0.

        self.best_valid_acc_avg, self.best_test_acc_avg = 0., 0.
        self.best_valid_acc_worst, self.best_test_acc_worst = 0., 0.

        print('finished model initialization....')


    # evaluation code for vanilla
    def evaluate(self, model, data_loader):
        model.eval()
        total_correct, total_num = 0, 0
        for _, data, attr, _, _ in tqdm(data_loader, leave=False):

            label = attr[:, 0]
            if attr.shape[1] > 2:
                group = attr[:, 2]
            else:
                group = None
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)



            # label = attr[:, 0]
            # data = data.to(self.device)
            # label = label.to(self.device)

            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

        accs = total_correct/float(total_num)
        model.train()

        return accs


    def summarize_acc(self, correct_by_groups, total_by_groups, bias = True, split = 'Train',stdout=True):
        all_correct = 0
        all_total = 0
        min_acc = 101.
        min_correct_total = [None, None]
        # if stdout:
        #     print(split + ' Accuracies by groups:')
        for yix, y_group in enumerate(correct_by_groups):
            for aix, a_group in enumerate(y_group):
                acc = a_group / total_by_groups[yix][aix] * 100
                if acc < min_acc:
                    min_acc = acc
                    min_correct_total[0] = a_group
                    min_correct_total[1] = total_by_groups[yix][aix]
                if stdout:
                    print(
                        f'{yix}, {aix}  acc: {int(a_group):5d} / {int(total_by_groups[yix][aix]):5d} = {a_group / total_by_groups[yix][aix] * 100:>7.3f}')
                all_correct += a_group
                all_total += total_by_groups[yix][aix]
        if stdout:
            if bias:
                average_str = f'Bised Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
                robust_str = f'Bised Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
            else:
                average_str = f'Average acc: {int(all_correct):5d} / {int(all_total):5d} = {100 * all_correct / all_total:>7.3f}'
                robust_str = f'Robust  acc: {int(min_correct_total[0]):5d} / {int(min_correct_total[1]):5d} = {min_acc:>7.3f}'
            print('-' * len(average_str))
            print(average_str)
            print(robust_str)
            print('-' * len(average_str))
        # return all_correct / all_total * 100, min_acc
        return min_acc

    # model_b, model_l, data_loader, n_group = 4, model='label', mode='dummy'
    def evaluate_civilcomments(self, net_b, net, dataloader, bias = False, n_group = 4, model='label', mode='dummy', split='Train', step = 0):
        

        if bias:
            net = net_b

        dataset = dataloader.dataset.dataset
        metadata = dataset.metadata_array
        correct_by_groups = np.zeros([2, len(dataset._identity_vars)])
        total_by_groups = np.zeros(correct_by_groups.shape)
        
        identity_to_ix = {}
        for idx, identity in enumerate(dataset._identity_vars):
            identity_to_ix[identity] = idx
        
        for identity_var, eval_grouper in zip(dataset._identity_vars, 
                                              dataset._eval_groupers):
            group_idx = eval_grouper.metadata_to_group(metadata).numpy()
            
            g_list, g_counts = np.unique(group_idx, return_counts=True)
            # print(identity_var, identity_to_ix[identity_var])
            # print(g_counts)
            
            for g_ix, g in enumerate(g_list):
                g_count = g_counts[g_ix]
                # Only pick from positive identities
                # e.g. only 1 and 3 from here:
                #   0 y:0_male:0
                #   1 y:0_male:1
                #   2 y:1_male:0
                #   3 y:1_male:1
                n_total = g_counts[g_ix]  #  + g_counts[3]
                if g in [1, 3]:
                    class_ix = 0 if g == 1 else 1  # 1 y:0_male:1
                    # print(g_ix, g, n_total)
        
        # net.to(args.device)
        net.eval()
        total_correct = 0
        with torch.no_grad():
            all_predictions = []
            all_correct = []

            for data in tqdm(dataloader, leave=False):
                i, inputs, attr, _, _ = data 
                # inputs, labels, data_ix = data
            #for i, data in enumerate(tqdm(dataloader)):
                labels = attr[:, 0]
                # label = attr
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Add this here to generalize NLP, CV models
                #outputs = get_output(net, inputs, labels, args)
                input_ids   = inputs[:, :, 0]
                input_masks = inputs[:, :, 1]
                segment_ids = inputs[:, :, 2]
                outputs = net(input_ids=input_ids,
                              attention_mask=input_masks,
                              token_type_ids=segment_ids,
                              labels=labels)[1]

                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == labels).detach().cpu()
                total_correct += correct.sum().item()
                all_correct.append(correct)
                all_predictions.append(predicted.detach().cpu())
                
                inputs = inputs.to(torch.device('cpu'))
                labels = labels.to(torch.device('cpu'))
                outputs = outputs.to(torch.device('cpu'))
                del inputs; del labels; del outputs
            
            all_correct = torch.cat(all_correct).numpy()
            all_predictions = torch.cat(all_predictions)
        
        # Evaluate predictions
        # dataset = dataloader.dataset
        y_pred = all_predictions  # torch.tensors
        y_true = dataset.y_array
        metadata = dataset.metadata_array
        
        correct_by_groups = np.zeros([2, len(dataset._identity_vars)])
        total_by_groups = np.zeros(correct_by_groups.shape)
        
        n_group = 2 * len(dataset._identity_vars)
        for identity_var, eval_grouper in zip(dataset._identity_vars, 
                                              dataset._eval_groupers):
            group_idx = eval_grouper.metadata_to_group(metadata).numpy()
            
            g_list, g_counts = np.unique(group_idx, return_counts=True)
            # print(g_counts)
            
            idx = identity_to_ix[identity_var]
            
            for g_ix, g in enumerate(g_list):
                g_count = g_counts[g_ix]
                # Only pick from positive identities
                # e.g. only 1 and 3 from here:
                #   0 y:0_male:0
                #   1 y:0_male:1
                #   2 y:1_male:0
                #   3 y:1_male:1
                n_total = g_count  # s[1] + g_counts[3]
                if g in [1, 3]:
                    n_correct = all_correct[group_idx == g].sum()
                    class_ix = 0 if g == 1 else 1  # 1 y:0_male:1
                    correct_by_groups[class_ix][idx] += n_correct
                    total_by_groups[class_ix][idx] += n_total
                    
        group_acc = correct_by_groups / total_by_groups
        acc_groups = {}
        for group in range(n_group):
            acc_groups[group] = group_acc[group // len(dataset._identity_vars), group % len(dataset._identity_vars)] 
        accs = total_correct/len(dataset)

        robust_acc = self.summarize_acc(correct_by_groups, total_by_groups, bias = bias, split=split, stdout=True)
        if not bias:
            if split == 'test':
                self.save_bert(step, robust_acc, net)

        net.train()


        return accs, acc_groups   
        # return 0, total_correct, len(dataset), correct_by_groups, total_by_groups, None, None, None

    def evaluate_ours(self, model_b, model_l, data_loader, n_group = 4, model='label', mode='dummy', split='Train', step = 0):
        if self.args.dataset =='civilcomments':
            return self.evaluate_civilcomments(model_b, model_l, data_loader, bias = (model == 'bias'), n_group = 4, model='label', mode='dummy', split=split, step = step)


        model_b.eval()
        model_l.eval()

        total_correct, total_num = 0, 0
        total_correct_groups = {}
        total_num_groups = {}
        acc_groups = {}
        for group in range(n_group):
            total_correct_groups[group] = 0
            total_num_groups[group] = 0


        iter = 0

        # index, data, attr, image_path, indicator

        for _, data, attr, _, _ in tqdm(data_loader, leave=False):
            label = attr[:, 0]
            if attr.shape[1] > 2:
                group = attr[:, 2]
            else:
                group = None
            # label = attr
            data = data.to(self.device)
            label = label.to(self.device)

            with torch.no_grad():
                if self.args.dataset == 'cmnist':
                    z_l = model_l.extract(data)
                    z_b = model_b.extract(data)
                elif self.args.dataset != 'civilcomments':
                    z_l, z_b = [], []
                    if mode == 'dummy':
                        hook_fn = self.model_l.avgpool.register_forward_hook(self.concat_dummy(z_l))
                    else:
                        hook_fn = self.model_l.avgpool.register_forward_hook(self.no_dummy(z_l))
                    _ = self.model_l(data)
                    hook_fn.remove()
                    z_l = z_l[0]
                    if mode == 'dummy':
                        hook_fn = self.model_b.avgpool.register_forward_hook(self.concat_dummy(z_b))
                    else:
                        hook_fn = self.model_b.avgpool.register_forward_hook(self.no_dummy(z_b))
                    _ = self.model_b(data)
                    hook_fn.remove()
                    z_b = z_b[0]
                
                    if mode == 'dummy':
                        if iter == 0:
                            # print('Current mode using is {:s} \n'.format(mode))
                            iter += 1
                        z_origin = torch.cat((z_l, z_b), dim=1)
                        if model == 'bias':
                            pred_label = model_b.fc(z_origin)
                        else:
                            pred_label = model_l.fc(z_origin)
                    else:
                        if iter == 0:
                            # print('Current mode using is {:s} \n'.format(mode))
                            iter += 1
                        z_origin = z_l
                        if model == 'bias':
                            pred_label = model_b.fc(z_origin)
                        else:
                            pred_label = model_l.fc(z_origin)
                else:
                    input_ids   = data[:, :, 0]
                    input_masks = data[:, :, 1]
                    segment_ids = data[:, :, 2]
                    if model == 'bias':
                        pred_label = model_b(input_ids=input_ids,
                                        attention_mask=input_masks,
                                        token_type_ids=segment_ids,
                                        labels=label)[1]
                    else:
                        pred_label = model_l(input_ids=input_ids,
                                        attention_mask=input_masks,
                                        token_type_ids=segment_ids,
                                        labels=label)[1]


                pred = pred_label.data.max(1, keepdim=True)[1].squeeze(1)

                correct = (pred == label).long()
                total_correct += correct.sum()
                total_num += correct.shape[0]

                if group is not None:
                    for group_id in range(n_group):
                        group_select = (group == group_id)
                        correct_select = (pred[group_select] == label[group_select]).long()
                        total_correct_groups[group_id] += correct_select.sum()
                        total_num_groups[group_id] += correct_select.shape[0]


        accs = total_correct/float(total_num)
        if group is not None:
            for group_id in range(n_group):
                acc_groups[group_id] = (total_correct_groups[group_id]/float(total_num_groups[group_id])).item()
        else:
            acc_groups = None
        model_b.train()
        model_l.train()

        return accs, acc_groups

    def save_vanilla(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model.th")
        else:
            model_path = os.path.join(self.result_dir, "model_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)
        print(f'{step} model saved ...')


    def save_ours(self, step, best=None):
        if best:
            model_path = os.path.join(self.result_dir, "best_model_l.th")
        else:
            model_path = os.path.join(self.result_dir, "model_l_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_l.state_dict(),
            'optimizer': self.optimizer_l.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        if best:
            model_path = os.path.join(self.result_dir, "best_model_b.th")
        else:
            model_path = os.path.join(self.result_dir, "model_b_{}.th".format(step))
        state_dict = {
            'steps': step,
            'state_dict': self.model_b.state_dict(),
            'optimizer': self.optimizer_b.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'{step} model saved ...')

    def save_bert(self, step, robust_acc, net):
        model_path = os.path.join(self.result_dir, f"model_{step}_{robust_acc}.th")
        state_dict = {
            'state_dict': net.state_dict(),
        }
        with open(model_path, "wb") as f:
            torch.save(state_dict, f)

        print(f'model saved ...')


    def board_vanilla_loss(self, step, loss_b):
        if self.args.wandb:
            wandb.log({
                "loss_b_train": loss_b,
            }, step=step,)

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_b_train", loss_b, step)


    def board_ours_loss(self, step, loss_dis_conflict, loss_dis_align, confusion, global_count):

        flatten_confusion = confusion.flatten()
        print('Correction: ', flatten_confusion)
        if self.args.wandb:
            wandb.log({
                "loss_dis_conflict":    loss_dis_conflict,
                "loss_dis_align":       loss_dis_align,
                "loss":                 loss_dis_conflict + loss_dis_align,
            }, step=step,)
            
            
            flatten_global_count = global_count.flatten()
            for i in range(len(flatten_confusion)):
                wandb.log({"logit_correction_"+str(i): flatten_confusion[i]}, step=step,)
                wandb.log({"global_count_"+str(i): flatten_global_count[i]}, step=step,)

        if self.args.tensorboard:
            self.writer.add_scalar(f"loss/loss_dis_conflict",  loss_dis_conflict, step)
            self.writer.add_scalar(f"loss/loss_dis_align",     loss_dis_align, step)
            self.writer.add_scalar(f"loss/loss",               loss_dis_conflict + loss_dis_align)

    def board_vanilla_acc(self, step, epoch, inference=None):
        valid_accs_b = self.evaluate(self.model_b, self.valid_loader)
        test_accs_b = self.evaluate(self.model_b, self.test_loader)

        # print(f'epoch: {epoch}')

        if valid_accs_b >= self.best_valid_acc_b:
            self.best_valid_acc_b = valid_accs_b
        if test_accs_b >= self.best_test_acc_b:
            self.best_test_acc_b = test_accs_b
            self.save_vanilla(step, best=True)

        if self.args.wandb:
            wandb.log({
                "acc_b_valid": valid_accs_b,
                "acc_b_test": test_accs_b,
            },
                step=step,)
            wandb.log({
                "best_acc_b_valid": self.best_valid_acc_b,
                "best_acc_b_test": self.best_test_acc_b,
            },
                step=step, )

        print(f'valid_b: {valid_accs_b} || test_b: {test_accs_b}')

        if self.args.tensorboard:
            self.writer.add_scalar(f"acc/acc_b_valid", valid_accs_b, step)
            self.writer.add_scalar(f"acc/acc_b_test", test_accs_b, step)

            self.writer.add_scalar(f"acc/best_acc_b_valid", self.best_valid_acc_b, step)
            self.writer.add_scalar(f"acc/best_acc_b_test", self.best_test_acc_b, step)


    def board_ours_acc(self, step, inference=None, model ='debias', mode = 'dummy', n_group = 4, eval = False, save = True):
        # check label network

        valid_accs_d, valid_acc_groups = self.evaluate_ours(self.model_b, self.model_l, self.valid_loader, n_group = n_group, model=model, mode=mode, split = 'valid', step = step)
        
        test_accs_d, test_acc_groups = self.evaluate_ours(self.model_b, self.model_l, self.test_loader, n_group = n_group, model=model, mode=mode, split = 'test', step = step)

        if eval:
            return 


        if valid_acc_groups is not None:
            valid_group_acc_list = list(valid_acc_groups.values())
            valid_accs_avg = np.nanmean(valid_group_acc_list)
            valid_accs_worst = np.nanmin(valid_group_acc_list)

            if valid_accs_avg >= self.best_valid_acc_avg:
                self.best_valid_acc_avg = valid_accs_avg
            if valid_accs_worst >= self.best_valid_acc_worst:
                self.best_valid_acc_worst = valid_accs_worst

        if test_acc_groups is not None:
            
            test_group_acc_list = list(test_acc_groups.values())
            test_accs_avg = np.nanmean(test_group_acc_list)
            test_accs_worst = np.nanmin(test_group_acc_list)

            if test_accs_avg >=self.best_test_acc_avg:
                self.best_test_acc_avg = test_accs_avg
            if test_accs_worst >=self.best_test_acc_worst:
                self.best_test_acc_worst = test_accs_worst

        # else:
        #     valid_accs_avg = 0
        #     valid_accs_worst = 0
        #     test_accs_avg = 0
        #     test_accs_worst = 0


            if inference:
                print(f'test acc: {test_accs_d.item()}')
                import sys
                sys.exit(0)

            if valid_accs_d >= self.best_valid_acc_d:
                self.best_valid_acc_d = valid_accs_d
            if test_accs_d >= self.best_test_acc_d:
                self.best_test_acc_d = test_accs_d
                if save:
                    self.save_ours(step, best=True)
            

            if self.args.wandb:
                wandb.log({
                    "acc_d_valid": valid_accs_d,
                    "acc_d_test": test_accs_d,
                    "acc_avg_valid": valid_accs_avg,
                    "acc_worst_valid": valid_accs_worst,
                    "acc_avg_test": test_accs_avg,
                    "acc_worst_test": test_accs_worst
                },
                    step=step, )
                wandb.log({
                    "best_acc_d_valid": self.best_valid_acc_d,
                    "best_acc_d_test": self.best_test_acc_d,
                    "best_acc_avg_valid": self.best_valid_acc_avg,
                    "best_acc_avg_test": self.best_test_acc_avg,
                    "best_acc_worst_valid": self.best_valid_acc_worst,
                    "best_acc_worst_test": self.best_test_acc_worst,
                },
                    step=step, )

                if (test_acc_groups is not None) and len(test_group_acc_list) < 16:

                    for g_id in range(len(test_group_acc_list)):
                        wandb.log({
                            "test_acc_group_" + str(g_id): test_group_acc_list[g_id],
                        },
                        step=step, )


            if self.args.tensorboard:
                self.writer.add_scalar(f"acc/acc_d_valid", valid_accs_d, step)
                self.writer.add_scalar(f"acc/acc_d_test", test_accs_d, step)
                self.writer.add_scalar(f"acc/best_acc_d_valid", self.best_valid_acc_d, step)
                self.writer.add_scalar(f"acc/best_acc_d_test", self.best_test_acc_d, step)
                self.writer.add_scalar(f"acc/best_acc_avg_valid", self.best_valid_acc_avg, step)
                self.writer.add_scalar(f"acc/best_acc_worst_valid", self.best_valid_acc_worst, step)
                self.writer.add_scalar(f"acc/best_acc_worst_test", self.best_test_acc_worst, step)

            if (test_acc_groups is not None) and len(test_group_acc_list) < 16:
                for g_id in range(len(test_group_acc_list)):
                    print(f"test_acc_group_{g_id}: {valid_group_acc_list[g_id]}")
                    print(f"valid_acc_group_{g_id}: {test_group_acc_list[g_id]}")
                print(f"Best Worst Test:{self.best_test_acc_worst}")
        print(f'valid_d: {valid_accs_d} || test_d: {test_accs_d} || best_test_d: {self.best_test_acc_d}')

    def concat_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return torch.cat((output, torch.zeros_like(output)), dim=1)
        return hook

    def no_dummy(self, z):
        def hook(model, input, output):
            z.append(output.squeeze())
            return output
        return hook

    def no_dummy_input(self, z):
        def hook(model, input):
            z.append(input[0])
            return input
        return hook

    def train_vanilla(self, args):
        self.criterion = nn.CrossEntropyLoss()
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'waterbird':
            self.model_l = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'civilcomments':
            self.model_l = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
        else:
            if self.args.use_resnet20: # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE_pretrained', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE_pretrained', self.num_classes).to(self.device)

        # self.model_b.load_state_dict(torch.load(os.path.join('./log/waterbird/waterbird_ours_GEC_0.9_ema_0.50_tau_0.10_lambda_2.00_avgtype_mv_batch/result/', 'best_model_b.th'))['state_dict'])

        if args.dataset == 'waterbird':
            print('!' * 10 + ' Using SGD ' + '!' * 10)

            self.optimizer_b = torch.optim.SGD(
                self.model_b.parameters(),
                lr=args.lr,
                weight_decay=0,
            )
        elif args.dataset == 'civilcomments':
            print('------------------- AdamW -------------------------------')
            self.optimizer_b = torch.optim.SGD(
                self.model_b.parameters(), #1e-3
                lr=2e-5,
                weight_decay=0
            )
            
            n_group = 16

        else:

            self.optimizer_b = torch.optim.Adam(
                self.model_b.fc.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

        step = 0

        for epoch in tqdm(range(args.num_epochs)):

            for index, data, attr, image_path, indicator in tqdm(self.train_loader, leave=False):

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx]
                spurious_label = attr[:, 1]

                logit_b = self.model_b(data)

                loss_b_update = self.criterion(logit_b, label)

                loss = loss_b_update.mean()


                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()

                ##################################################
                #################### LOGGING #####################
                ##################################################

                if (step % args.valid_freq == 0) and (args.dataset != 'waterbird'): 
                    print("------------------------ Validation Starts--------------------------------")
                    self.board_vanilla_acc(step, epoch, inference=None)
                    print("------------------------ Validation Done --------------------------------")
                step += 1

            if args.dataset == 'waterbird':
                self.board_ours_acc(epoch, model = 'bias', mode = 'no_dummy' , n_group = 4, save = False)

        if len(random_indices_all_groups) > 0:
            mixed_feature = lam * feature[indices_all_groups] +  (1 - lam) * feature[random_indices_all_groups]
            mixed_correction = lam * correction[indices_all_groups] +  (1 - lam) * correction[random_indices_all_groups]
            label_a = label[indices_all_groups]
            label_b = label[random_indices_all_groups]
        else:
            mixed_feature = None
            label_a, label_b, lam = None, None, None

        return mixed_feature, mixed_correction, label_a, label_b, lam


    def train_ours(self, args):
        epoch, cnt = 0, 0
        print('************** main training starts... ************** ')
        train_num = len(self.train_dataset)
        print('Length of training set: {:d}'.format(train_num))

        self.bias_criterion = GeneralizedCELoss(q=args.q)
        self.criterion = LogitCorrectionLoss(eta = 1.0)

        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'waterbird':
            self.model_l = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'civilcomments':
            self.model_l = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
        else:
            if self.args.use_resnet20: # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        if args.dataset == 'waterbird':
            print('!' * 10 + ' Using SGD ' + '!' * 10)

            self.optimizer_l = torch.optim.SGD(
                self.model_l.parameters(), #1e-3
                lr=args.lr,
                weight_decay=args.weight_decay,#1e-3
            )

            self.optimizer_b = torch.optim.SGD(
                self.model_b.parameters(),
                lr=args.lr*0.1,
                weight_decay=0,
            )
        elif args.dataset == 'civilcomments':
            self.optimizer_b = torch.optim.SGD(
                self.model_b.parameters(),
                lr=2e-5,
                weight_decay=0
            )
            
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model_l.named_parameters() 
                            if not any(nd in n for nd in no_decay)], 
                 'weight_decay': args.weight_decay},
                {'params': [p for n, p in self.model_l.named_parameters() 
                            if any(nd in n for nd in no_decay)], 
                 'weight_decay': 0.0}]
            self.optimizer_l = optim.AdamW(optimizer_grouped_parameters,
                                    lr=args.lr, eps=1e-8)

            n_group = 16

        else:

            self.optimizer_l = torch.optim.Adam(
                self.model_l.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            self.optimizer_b = torch.optim.Adam(
                self.model_b.parameters(),
                lr=args.lr*0.1,
                weight_decay=args.weight_decay,
            )


        if args.use_lr_decay and args.dataset == 'waterbird':
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_epoch, gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_epoch, gamma=args.lr_gamma)
        elif args.use_lr_decay and args.dataset == 'civilcomments':
            total_updates = args.num_epochs

            self.scheduler_b = get_bert_scheduler(self.optimizer_b, n_epochs=1,#args.num_epochs,
                                                  warmup_steps=0,
                                                  dataloader=self.train_loader)
            self.scheduler_l = get_bert_scheduler(self.optimizer_l, n_epochs=total_updates,#args.num_epochs,
                                                  warmup_steps=0,
                                                  dataloader=self.train_loader)
        else:
            self.scheduler_b = optim.lr_scheduler.StepLR(self.optimizer_b, step_size=args.lr_decay_step, gamma=args.lr_gamma)
            self.scheduler_l = optim.lr_scheduler.StepLR(self.optimizer_l, step_size=args.lr_decay_step, gamma=args.lr_gamma)

        print(f'criterion: {self.criterion}')
        print(f'bias criterion: {self.bias_criterion}')
        train_iter = iter(self.train_loader)

        step = 0

        for epoch in tqdm(range(args.num_epochs)):

            for index, data, attr, image_path, indicator in tqdm(self.train_loader, leave=False):

                data = data.to(self.device)
                attr = attr.to(self.device)
                label = attr[:, args.target_attr_idx].to(self.device)
                bias = attr[:, 1].to(self.device)
                if args.dataset == 'waterbird':
                    alpha = sigmoid_rampup(epoch, args.curr_epoch)*0.5
                else:
                    alpha = sigmoid_rampup(step, args.curr_step)*0.5

                if args.dataset == 'cmnist':
                    # z_l = self.model_l.extract(data)
                    z_b = self.model_b.extract(data)
                    pred_align = self.model_b.fc(z_b)
                    self.sample_margin_ema_b.update(F.softmax(pred_align.detach()/args.tau), index)
                    pred_align_mv = self.sample_margin_ema_b.parameter[index].clone().detach()
                    _, pseudo_label = torch.max(pred_align_mv, dim=1)
                    self.confusion.update(pred_align_mv, label, pseudo_label, fix = None)
                    correction_matrix = self.confusion.parameter.clone().detach()
                    if args.avg_type == 'epoch':
                        correction_matrix = correction_matrix/self.confusion.global_count_.to(self.device)

                    correction_delta = correction_matrix[:,pseudo_label]
                    correction_delta = torch.t(correction_delta)
                    return_dict = group_mixUp(data, pseudo_label, correction_delta, label, self.num_classes, alpha)
                    mixed_target_data = return_dict["mixed_feature"]
                    mixed_biased_prediction = return_dict["mixed_correction"]
                    label_a = return_dict["label_majority"]
                    label_b = return_dict["label_minority"]
                    lam_target = return_dict["lam"]

                    z_l = self.model_l.extract(mixed_target_data)
                    pred_conflict = self.model_l.fc(z_l)

                elif args.dataset == 'civilcomments':

                    input_ids   = data[:, :, 0]
                    input_masks = data[:, :, 1]
                    segment_ids = data[:, :, 2]

                    pred_align = self.model_b(input_ids=input_ids,
                                    attention_mask=input_masks,
                                    token_type_ids=segment_ids,
                                    labels=label)[1]

                    
                    self.sample_margin_ema_b.update(F.softmax(pred_align.detach()/args.tau), index)
                    pred_align_mv = self.sample_margin_ema_b.parameter[index].clone().detach()
                    _, pseudo_label = torch.max(pred_align_mv, dim=1)
                    self.confusion.update(pred_align_mv, label, pseudo_label, fix = None)
                    correction_matrix = self.confusion.parameter.clone().detach()
                    if args.avg_type == 'epoch':
                        correction_matrix = correction_matrix/self.confusion.global_count_.to(self.device)

                    correction_matrix = correction_matrix/((correction_matrix).sum(dim=0,keepdims =True) + 1e-4)
                    correction_delta = correction_matrix[:,pseudo_label]
                    correction_delta = torch.t(correction_delta)

                
                    z_l = []
                    hook_fn = self.model_l.dropout.register_forward_pre_hook(self.no_dummy_input(z_l))#register_forward_hook(self.no_dummy_input(z_l))
                    _   =  self.model_l(input_ids=input_ids,
                                        attention_mask=input_masks,
                                        token_type_ids=segment_ids,
                                        labels=label)
                    hook_fn.remove()
                    z_l = z_l[0]
                    
                    return_dict = group_mixUp(z_l, pseudo_label, correction_delta, label, self.num_classes, alpha)
                    mixed_target_z_l = return_dict["mixed_feature"]
                    mixed_biased_prediction = return_dict["mixed_correction"]
                    label_a = return_dict["label_majority"]
                    label_b = return_dict["label_minority"]
                    lam_target = return_dict["lam"]


                    pred_conflict = self.model_l.classifier(self.model_l.dropout(mixed_target_z_l)) 

                else:
                    z_b = []
                    # Use this only for reproducing CIFARC10 of LfF
                    if self.args.use_resnet20:
                        hook_fn = self.model_b.layer3.register_forward_hook(self.no_dummy(z_b))
                        _ = self.model_b(data)
                        hook_fn.remove()
                        z_b = z_b[0]

                        pred_align = self.model_b.fc(z_b)
                        self.sample_margin_ema_b.update(F.softmax(pred_align.detach()/args.tau), index)
                        pred_align_mv = self.sample_margin_ema_b.parameter[index].clone().detach()
                        _, pseudo_label = torch.max(pred_align_mv, dim=1)
                        self.confusion.update(pred_align_mv, label, pseudo_label, fix = None)
                        correction_matrix = self.confusion.parameter.clone().detach()
                        if args.avg_type == 'epoch':
                            correction_matrix = correction_matrix/self.confusion.global_count_.to(self.device)
                        correction_matrix = correction_matrix/(correction_matrix).sum(dim=0,keepdims =True)
                        correction_delta = correction_matrix[:,pseudo_label]
                        correction_delta = torch.t(correction_delta)
                        return_dict = group_mixUp(data, pseudo_label, correction_delta, label, self.num_classes, alpha)
                        mixed_target_data = return_dict["mixed_feature"]
                        mixed_biased_prediction = return_dict["mixed_correction"]
                        label_a = return_dict["label_majority"]
                        label_b = return_dict["label_minority"]
                        lam_target = return_dict["lam"]
                        

                        z_l = []
                        hook_fn = self.model_l.layer3.register_forward_hook(self.no_dummy(z_l))
                        _ = self.model_l(mixed_target_data)
                        hook_fn.remove()

                        z_l = z_l[0]

                        pred_conflict = self.model_l.fc(z_l)

                    else:
                        hook_fn = self.model_b.avgpool.register_forward_hook(self.no_dummy(z_b))
                        _ = self.model_b(data)
                        hook_fn.remove()
                        z_b = z_b[0]

                        pred_align = self.model_b.fc(z_b)
                        self.sample_margin_ema_b.update(F.softmax(pred_align.detach()/args.tau), index)
                        pred_align_mv = self.sample_margin_ema_b.parameter[index].clone().detach()

                        _, pseudo_label = torch.max(pred_align_mv, dim=1)
                        self.confusion.update(pred_align_mv, label, pseudo_label, fix = None)
                        correction_matrix = self.confusion.parameter.clone().detach()
                        if args.avg_type == 'epoch':
                            correction_matrix = correction_matrix/self.confusion.global_count_.to(self.device)
                        correction_matrix = correction_matrix/(correction_matrix).sum(dim=0,keepdims =True)
                        correction_delta = correction_matrix[:,pseudo_label]
                        correction_delta = torch.t(correction_delta)

                        return_dict = group_mixUp(data, pseudo_label, correction_delta, label, self.num_classes, alpha)
                        mixed_target_data = return_dict["mixed_feature"]
                        mixed_biased_prediction = return_dict["mixed_correction"]
                        label_a = return_dict["label_majority"]
                        label_b = return_dict["label_minority"]
                        lam_target = return_dict["lam"]

                        z_l = []
                        hook_fn = self.model_l.avgpool.register_forward_hook(self.no_dummy(z_l))
                        _ = self.model_l(mixed_target_data)
                        hook_fn.remove()

                        z_l = z_l[0]

                        pred_conflict = self.model_l.fc(z_l)


             
                self.sample_margin_ema_b.update(F.softmax(pred_align.detach()), index)               

                loss_dis_conflict = lam_target * self.criterion(pred_conflict, label_a, mixed_biased_prediction) +\
                 (1 - lam_target) * self.criterion(pred_conflict, label_b, mixed_biased_prediction)
         
                loss_dis_align = self.bias_criterion(pred_align, label) 
                loss  = loss_dis_conflict.mean() + args.lambda_dis_align * loss_dis_align.mean()               # Eq.2 L_dis

                
                self.optimizer_l.zero_grad()
                self.optimizer_b.zero_grad()

                loss.backward()

                if args.dataset == 'civilcomments':
                    torch.nn.utils.clip_grad_norm_(self.model_l.parameters(), 1.0)

                if args.use_lr_decay and args.dataset != 'waterbird':
                    self.scheduler_b.step()
                    self.scheduler_l.step()
                
                
                self.optimizer_l.step()
                self.optimizer_b.step()
                
                if args.use_lr_decay and step % args.lr_decay_step == 0 and args.dataset != 'waterbird':
                    print('******* learning rate decay .... ********')
                    print(f"self.optimizer_b lr: { self.optimizer_b.param_groups[-1]['lr']}")
                    print(f"self.optimizer_l lr: { self.optimizer_l.param_groups[-1]['lr']}")
                

                if step > 0 and step % args.save_freq == 0 and args.dataset != 'waterbird' and args.dataset != 'civilcomments':
                    self.save_ours(step)


                if step > 0 and step % args.log_freq == 0 and args.dataset != 'waterbird': #and args.dataset != 'civilcomments':
                    confusion_numpy = correction_matrix.cpu().numpy()
                    self.board_ours_loss(
                        step=step,
                        loss_dis_conflict=loss_dis_conflict.mean(),
                        loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                        confusion=confusion_numpy,
                        global_count=self.confusion.global_count_.cpu().numpy()
                        )
                if step > 0 and  (step % args.valid_freq == 0) and (args.dataset != 'waterbird'): #and (args.dataset != 'civilcomments'):
                    print('################################ #epoch {:d} ############################\n'.format(epoch))
                    self.board_ours_acc(step, model = 'debias', mode = 'no_dummy', n_group = n_group)

                step += 1

            if args.use_lr_decay and args.dataset == 'waterbird':
                self.scheduler_b.step()
                self.scheduler_l.step()

            if args.use_lr_decay and epoch % args.lr_decay_epoch == 0 and args.dataset == 'waterbird':
                print('******* learning rate decay .... ********')
                print(f"self.optimizer_b lr: { self.optimizer_b.param_groups[-1]['lr']}")
                print(f"self.optimizer_l lr: { self.optimizer_l.param_groups[-1]['lr']}")

            
            confusion_numpy = correction_matrix.cpu().numpy()
            if (args.dataset == 'waterbird'): 
                self.board_ours_loss(
                    step=epoch,
                    loss_dis_conflict=loss_dis_conflict.mean(),
                    loss_dis_align=args.lambda_dis_align * loss_dis_align.mean(),
                    confusion=confusion_numpy,
                    global_count=self.confusion.global_count_.cpu().numpy()
                    )
            if (args.dataset == 'waterbird'): 
                self.board_ours_acc(epoch, model = 'debias', mode = 'no_dummy' , n_group = 4)
            self.confusion.global_count_ = torch.zeros(self.num_classes, self.num_classes)
            if args.avg_type == 'epoch':
                self.confusion.initiate_parameter()



    def test_ours(self, args):
        if args.dataset == 'cmnist':
            self.model_l = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('mlp_DISENTANGLE', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'waterbird':
            self.model_l = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('resnet_50_pretrained', self.num_classes, bias = True).to(self.device)
        elif args.dataset == 'civilcomments':
            print('----------------- Load Bert --------------------')
            self.model_l = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
            self.model_b = get_model('bert-base-uncased_pt', self.num_classes, bias = True).to(self.device)
        else:
            if self.args.use_resnet20: # Use this option only for comparing with LfF
                self.model_l = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                self.model_b = get_model('ResNet20_OURS', self.num_classes).to(self.device)
                print('our resnet20....')
            else:
                self.model_l = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)
                self.model_b = get_model('resnet_DISENTANGLE', self.num_classes).to(self.device)

        self.model_l.load_state_dict(torch.load(os.path.join(args.pretrained_path, 'model_200_37.55377996312232.th'))['state_dict'])
        self.board_ours_acc(-1, model = 'debias', mode = 'no_dummy', n_group = 16, eval = True)
