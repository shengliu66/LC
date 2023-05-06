'''Modified from https://github.com/alinlab/LfF/blob/master/util.py'''

import io
import torch
import numpy as np
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

class EMA:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0), num_classes)
        self.updated = torch.zeros(label.size(0), num_classes)
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].max()

    def min_loss(self, label):
        label_index = torch.where(self.label == label)[0]
        return self.parameter[label_index].min()


class EMA_squre:
    def __init__(self, num_classes=None, alpha=0.9, avg_type = 'mv'):
        self.alpha = alpha
        self.parameter = torch.zeros(num_classes, num_classes)
        self.global_count_ = torch.zeros(num_classes, num_classes)
        self.updated = torch.zeros(num_classes, num_classes)
        self.num_classes = num_classes
        self.max = torch.zeros(self.num_classes).cuda()
        self.avg_type = avg_type

    def update(self, data, y_list, a_list, curve=None, iter_range=None, step=None, bias=None, fix = None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        # self.global_count_ = self.global_count_.to(data.device)
        y_list = y_list.to(data.device)
        a_list = a_list.to(data.device)


        count = torch.zeros(self.num_classes, self.num_classes).to(data.device)
        # parameter_temp = torch.zeros(self.num_classes, self.num_classes, self.num_classes).to(data.device)

        if self.avg_type == 'mv':
            if curve is None:
                for i, (y, a) in enumerate(zip(y_list, a_list)):
                    # parameter_temp[y,a] += data[i]
                    count[y,a] += 1
                    self.global_count_[y,a] += 1
                    self.parameter[y,a] = self.alpha * self.parameter[y,a] + (1 - self.alpha * self.updated[y,a]) * data[i,y]#parameter_temp[y,a]/count[y,a]
                    self.updated[y,a] = 1
            else:
                alpha = curve ** -(step / iter_range)
                for i, (y, a) in enumerate(zip(y_list, a_list)):
                    # parameter_temp[y,a] += data[i]
                    count[y,a] += 1
                    self.global_count_[y,a] += 1
                    self.parameter[y,a] = alpha * self.parameter[y,a] + (1 - alpha * self.updated[y,a]) * data[i,y]#parameter_temp[y,a]/count[y,a]
                    self.updated[y,a] = 1
        elif self.avg_type == 'mv_batch':
            self.parameter_temp = torch.zeros(self.num_classes, self.num_classes).to(data.device)
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter_temp[y,a] += data[i,y]
            self.parameter = self.alpha * self.parameter + (1 - self.alpha) * self.parameter_temp / (count + 1e-4)
        elif self.avg_type == 'batch':
            self.parameter_temp = torch.zeros(self.num_classes, self.num_classes).to(data.device)
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter_temp[y,a] += data[i,y]
            self.parameter = self.parameter_temp / (count + 1e-4)
        elif self.avg_type == 'epoch':
            for i, (y, a) in enumerate(zip(y_list, a_list)):
                count[y,a] += 1
                self.global_count_[y,a] += 1
                self.parameter[y,a] += data[i,y]
        else:
            raise NotImplementedError("This averaging type is not yet implemented!")

        if fix is not None:
            self.parameter = torch.ones(self.num_classes, self.num_classes) * 0.1#* 0.005/(self.num_classes-1)
            # for i in range(self.num_classes):
            #     self.parameter[i,i] = 0.995
            self.parameter = self.parameter.to(data.device)





    # def max_loss(self, label):
    #     label_index = torch.where(self.label == label)[0]
    #     return self.parameter[label_index].max()

    # def min_loss(self, label):
    #     label_index = torch.where(self.label == label)[0]
    #     return self.parameter[label_index].min()



class EMA_area:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))
        self.num_classes = num_classes
        # self.max = torch.zeros(self.num_classes).cuda()
        self.data_old = torch.zeros(label.size(0)).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        self.parameter[index] += 0.5 * self.data_old[index] + (1 - 0.5 * self.updated[index]) * data
        self.updated[index] = 1
        self.data_old[index] = data

    def max_area(self, label, temp = 1):
        label_index = torch.where(self.label == label)[0]
        return torch.nn.functional.sigmoid(-self.parameter[label_index]/temp).max()



class EMA_feature:
    def __init__(self, label, num_classes=None, alpha=0.9):
        self.label = label.cuda()
        self.alpha = alpha
        self.parameter = torch.zeros((label.size(0),num_classes))
        self.updated = torch.zeros((label.size(0),num_classes))
        self.num_classes = num_classes
        #self.max = torch.zeros(self.num_classes).cuda()

    def update(self, data, index, curve=None, iter_range=None, step=None):
        self.parameter = self.parameter.to(data.device)
        self.updated = self.updated.to(data.device)
        index = index.to(data.device)

        if curve is None:
            self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        else:
            alpha = curve ** -(step / iter_range)
            self.parameter[index] = alpha * self.parameter[index] + (1 - alpha * self.updated[index]) * data
        self.updated[index] = 1

    # def max_loss(self, label):
    #     label_index = torch.where(self.label == label)[0]
    #     return self.parameter[label_index].max()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length
EPS = 1e-9

def grad_norm(module):
    parameters = module.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm = param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def adaptive_gradient_clipping_(generator_module: nn.Module, mi_module: nn.Module):
    """
    Clips the gradient according to the min norm of the generator and mi estimator
    Arguments:
        generator_module -- nn.Module 
        mi_module -- nn.Module
    """
    norm_generator = grad_norm(generator_module)
    #norm_estimator = grad_norm(mi_module)

    min_norm = norm_generator#np.minimum(norm_generator, norm_estimator)

    parameters = list(
        filter(lambda p: p.grad is not None, mi_module.parameters()))
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    for p in parameters:
        p.grad.data.mul_(min_norm/(norm_estimator + EPS))



def get_bert_scheduler(optimizer, n_epochs, warmup_steps, dataloader, last_epoch=-1):
    """
    Learning rate scheduler for BERT model training
    """
    num_training_steps = int(np.round(len(dataloader) * n_epochs))
    print(f'\nt_total is {num_training_steps}\n')
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                warmup_steps,
                                                num_training_steps,
                                                last_epoch)
    return scheduler

# From pytorch-transformers:
# def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
#                                      num_training_steps, last_epoch=-1):
#     """
#     Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
#     a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
#     Args:
#         optimizer (:class:`~torch.optim.Optimizer`):
#             The optimizer for which to schedule the learning rate.
#         num_warmup_steps (:obj:`int`):
#             The number of steps for the warmup phase.
#         num_training_steps (:obj:`int`):
#             The total number of training steps.
#         last_epoch (:obj:`int`, `optional`, defaults to -1):
#             The index of the last epoch when resuming training.
#     Return:
#         :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
#     """

#     def lr_lambda(current_step: int):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )

#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
