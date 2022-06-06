import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import pdb, ipdb
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict


def add_path(path):
    if path not in sys.path:
        print('Adding {}'.format(path))
        sys.path.append(path)

add_path("../lib")
from base_model.get_model import get_model
from attack.pgd import IPGD
from utils.misc import seed_torch, torch_accuracy, mkdir
from dataset import get_CIFAR10
from utils.func import train, test, clean_train


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--device", '--d', type=int, default=0, help='Which gpu to use')
parser.add_argument('--batch_size', '--b', type=int, default=128, help='batch size')
parser.add_argument('--total_epochs', "--te", type=int, default=100)
parser.add_argument("--model", type=str,
                    default="res18", choices=["res18", "cnn", "wres28", "wres34"]
                    )
parser.add_argument("--optimizer", "--opt", type=str, default="sgd", choices=["sgd", "adam"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save_path", type=str, default="checkpoint")
parser.add_argument("--no_aug", action="store_true")
parser.add_argument("--val_interval", "--vi", type=int, default=1)
parser.add_argument("--data", type=str, default="cifar10")
parser.add_argument("--other_weight", "--ow", default=0, type=float)
args = parser.parse_args()

if args.data == "mnist":
    args.model = "mnist"
    args.total_epochs = 50
elif args.data == "svhn":
    args.lr = 0.01   # follow AWP repo

print("Args:", args)

alg_str = None

device = torch.device('cuda:{}'.format(args.device))
seed_torch(args.seed)
best_advacc = 0.  # best test accuracy
best_correspond_acc = 0.   # corresponding clean accuracy
start_epoch = -1  # start from epoch 0 or last checkpoint epoch
best_epoch = 0

print('==> Preparing data..')
if args.data == "cifar10":
    trainset, testset, trainloader, testloader = get_CIFAR10(
        data_path="~/data/cifar", batch_size=args.batch_size, aug=not args.no_aug,
        train_shuffle=True, num_workers=2)
else:
    raise NotImplementedError

print('==> Building model..')
net = get_model(args.model, num_classes=100 if args.data == "cifar100" else 10)
net.to(device)

criterion = nn.CrossEntropyLoss()
if args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
        milestones=[int(0.5 * args.total_epochs), int(0.75 * args.total_epochs)], gamma=0.1)
elif args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)


TrainAttack = IPGD(eps=8 / 255.0, sigma=2 / 255.0, nb_iter=10, norm=np.inf, DEVICE=device)
TestAttack = IPGD(eps=8 / 255.0, sigma=2 / 255.0, nb_iter=20, norm=np.inf, DEVICE=device)


def otherlabel(labels, num_classes=10):
    other_labels = torch.randint(low=0, high=num_classes-1, size=labels.size(), dtype=labels.dtype, device=labels.device)
    other_labels[other_labels >= labels] += 1
    return other_labels

NUM_CLASSES = 10


def train(epoch):
    train_adv_loss = 0.
    train_other_adv_loss = 0.

    adv_correct = 0
    total = 0
    pbar = tqdm(trainloader)

    curr_lr = lr_scheduler.get_lr()[0]
    pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inp = TrainAttack.my_mr_oa_attack(net, inputs, targets, other_weight=args.other_weight, num_classes=NUM_CLASSES)

        optimizer.zero_grad()
        net.train()

        adv_outputs = net(adv_inp)
        adv_loss = F.cross_entropy(adv_outputs, targets)
        # our proposed MCE loss
        other_advloss = - F.log_softmax(-adv_outputs, dim=1) * (1 - F.one_hot(targets, num_classes=NUM_CLASSES))
        other_advloss = other_advloss.sum() / ((NUM_CLASSES - 1) * len(targets))
        total_advloss = adv_loss + args.other_weight * other_advloss

        total_advloss.backward()
        optimizer.step()

        _, adv_predicted = adv_outputs.max(1)
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()
        train_adv_loss += adv_loss.item()
        train_other_adv_loss += other_advloss.item()

        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar_dic['adv loss'] = '{:.3f}'.format(train_adv_loss / (batch_idx + 1))
        pbar_dic['otheradv loss'] = '{:.3f}'.format(train_other_adv_loss / (batch_idx + 1))
        pbar.set_postfix(pbar_dic)


if __name__ == "__main__":

    for epoch in range(start_epoch + 1, args.total_epochs):
        train(epoch)

        if args.optimizer == "sgd":
            lr_scheduler.step()

        if (epoch + 1) % args.val_interval == 0 or epoch >= start_epoch + args.total_epochs - 10:
            acc, advacc = test(net, testloader, device, TestAttack, epoch)

            if advacc > best_advacc:
                best_advacc = advacc
                best_correspond_acc = acc
                best_epoch = epoch

    print("Best advacc: {:.2f} at {:} epoch, cleanacc: {:.2f}".format(best_advacc, best_epoch, best_correspond_acc))