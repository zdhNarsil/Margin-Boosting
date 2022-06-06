import math
import os
from typing import Tuple, List, Dict
import sys
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms


def mkdir(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

    # abs_current_path = os.path.realpath('./')
    # root_path = os.path.join('/', *abs_current_path.split(os.path.sep)[:-2])
    # lib_dir = os.path.join(root_path, 'lib')
    # add_path(lib_dir)


def make_symlink(source, link_name):
    '''
    Note: overwriting enabled!
    '''
    if os.path.exists(link_name):
        # print("Link name already exist! Removing '{}' and overwriting".format(link_name))
        os.remove(link_name)
    if os.path.exists(source):
        os.symlink(source, link_name)
        return
    else:
        print('Source path not exists')
    # print('SymLink Wrong!')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False  # decrease efficiency
    # torch.backends.cudnn.enabled = False
    print("==> Set seed to {:}".format(seed))


def torch_accuracy(output, target, topk=(1,)) -> List[torch.Tensor]:
    '''
    param output, target: should be torch Variable
    '''
    # assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    # assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    # print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim=True)
        ans.append(is_correct_i.mul_(100.0 / batch_size).item())

    return ans


class AvgMeter(object):
    '''
    Computing mean
    '''
    name = 'No name'

    def __init__(self, name='No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0

    def update(self, mean_var, count=1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


def save_checkpoint(now_epoch, net, optimizer, lr_scheduler, file_name):
    checkpoint = {'epoch': now_epoch,
                  'state_dict': net.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'lr_scheduler_state_dict': lr_scheduler.state_dict()}
    if os.path.exists(file_name):
        print('Overwriting {}'.format(file_name))
    torch.save(checkpoint, file_name)
    link_name = os.path.join('/', *file_name.split(os.path.sep)[:-1], 'last.checkpoint')
    # print(link_name)
    make_symlink(source=file_name, link_name=link_name)


def load_checkpoint(file_name, net=None, optimizer=None, lr_scheduler=None, DEVICE=None):
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        if DEVICE:
            check_point = torch.load(file_name, map_location=DEVICE)
        else:
            check_point = torch.load(file_name)
        if net is not None:
            print('Loading network state dict')
            net.load_state_dict(check_point['state_dict'])
        if optimizer is not None:
            print('Loading optimizer state dict')
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        if lr_scheduler is not None:
            print('Loading lr_scheduler state dict')
            lr_scheduler.load_state_dict(check_point['lr_scheduler_state_dict'])

        return check_point['epoch']
    else:
        print("=> no checkpoint found at '{}'".format(file_name))


def to_onehot(inp, num_dim=10):
    # inp: (bs,) int
    # ret: (bs, num_dim) float
    assert inp.dtype == torch.long

    batch_size = inp.shape[0]
    y_onehot = torch.FloatTensor(batch_size, num_dim).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.reshape(batch_size, 1), 1)

    return y_onehot