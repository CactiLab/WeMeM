"""
This code is modified from https://github.com/Machine-Learning-Security-Lab/mia_prune
"""
import random
import numpy as np
import torch
from torch.nn import init
import torch.nn.functional as F

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
            init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            init.xavier_normal_(m.weight)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()


def get_model(model_type, num_cls, input_dim):
    if model_type == "resnet18":
        from cifar10_models import resnet18
        model = resnet18(pretrained=False, num_classes=num_cls)
    elif model_type == "vgg16bn":
        from cifar10_models import vgg16_bn
        model = vgg16_bn(pretrained=False, num_classes=num_cls)
    elif model_type == "densenet121":
        from cifar10_models import densenet121
        model = densenet121(pretrained=False, num_classes=num_cls)
    elif model_type == "columnfc":
        from models import ColumnFC
        model = ColumnFC(input_dim=input_dim, output_dim=num_cls)
    elif model_type == "mia_fc":
        from models import MIAFC
        model = MIAFC(input_dim=num_cls, output_dim=2)
    elif model_type == "transformer":
        from transformer import Transformer
        model = Transformer(input_dim=num_cls, output_dim=2)
    else:
        print(model_type)
        raise ValueError
    return model


def get_optimizer(optimizer_name, parameters, lr, weight_decay=0):
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_name == "":
        optimizer = None
        # print("Do not use optimizer.")
    else:
        print(optimizer_name)
        raise ValueError
    return optimizer


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)


def one_hot_embedding(y, num_classes=10, dtype=torch.FloatTensor):
    '''
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    '''
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype).to(y.device)
    return torch.scatter(zeros, scatter_dim, y_tensor, 1)
    