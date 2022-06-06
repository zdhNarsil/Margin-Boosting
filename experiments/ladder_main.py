import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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
from utils.func import test, ensemble_test


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '--r', default=-1, type=int)
parser.add_argument('--resume_iter', '--ri', default=-1, type=int)
parser.add_argument("--device", '--d', type=int, default=0, help='Which gpu to use')
parser.add_argument('--batch_size', '--b', type=int, default=128, help='batch size')
parser.add_argument('--total_epochs', "--te", type=int, default=100)

parser.add_argument("--model", type=str,)
parser.add_argument("--optimizer", "--opt", type=str, default="sgd", choices=["sgd", "adam"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--save", action="store_true")
parser.add_argument("--save_path", type=str, default="checkpoint")
parser.add_argument("--no_aug", action="store_true")
parser.add_argument("--val_interval", "--vi", type=int, default=1)
parser.add_argument("--data", type=str, default="cifar10")

parser.add_argument("--ensemble_num", "--en", default=1, type=int)
parser.add_argument("--persistent", "--p", action="store_true")
args = parser.parse_args()

if args.data == "mnist":
    args.model = "mnist"
    args.total_epochs = 50

print("Args:", args)

setting_name = "epoch{:d}_bs{:d}".format(args.total_epochs, args.batch_size)
device = torch.device('cuda:{}'.format(args.device))
seed_torch(args.seed)

print('==> Preparing data..')
if args.data == "cifar10":
    trainset, testset, trainloader, testloader = get_CIFAR10(
        data_path="~/data/cifar", batch_size=args.batch_size, aug=not args.no_aug,
        train_shuffle=True, num_workers=1)

save_path = os.path.join(args.save_path, "ladder" + ("_pers" if args.persistent else ""))
save_path += "_feature"

save_path += "_advtr"
mkdir(save_path)
save_path = os.path.join(save_path, args.model + "_" + setting_name + "_seed{:d}".format(args.seed) + "_advtrain"
                         )
if args.save:
    mkdir(save_path)
    print("Will save at:", save_path)

if args.data == "cifar10":
    TrainAttack = IPGD(eps=8 / 255.0, sigma=2 / 255.0, nb_iter=10, norm=np.inf, DEVICE=device)
    TestAttack = IPGD(eps=8 / 255.0, sigma=2 / 255.0, nb_iter=20, norm=np.inf, DEVICE=device)


def func(model_ls, inp, targets, c):
    pred = [-F.cross_entropy(net(inp), torch.ones_like(targets).to(device).detach() * c, reduction="none")
            for net in model_ls]
    return sum(pred) / len(model_ls)


def train(epoch):
    train_loss = 0.
    train_adv_loss = 0.
    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(trainloader)
    curr_lr = lr_scheduler.get_lr()[0]
    pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        adv_inp = TrainAttack.my_ensemble_attack(model_ls, inputs, targets, rand=True)

        optimizer.zero_grad()
        model_ls[-1].train()

        pred = sum([net(adv_inp) for net in model_ls]) / len(model_ls)
        adv_loss = criterion(pred, targets)
        _, adv_predicted = pred.detach().max(1)

        adv_loss.backward()
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()

        optimizer.step()

        train_adv_loss += adv_loss.item()

        pbar_dic = OrderedDict()
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar_dic['adv loss'] = '{:.3f}'.format(train_adv_loss / (batch_idx + 1))
        pbar.set_postfix(pbar_dic)

criterion = nn.CrossEntropyLoss()


if __name__ == "__main__":
    model_ls = []
    print_ls = []

    for iteration in range(args.ensemble_num):
        print('==> Building model..')
        model_ls.append(get_model(args.model).to(device))
        if args.persistent and iteration >= 1:
            model_ls[-1].load_state_dict(model_ls[-2].state_dict())

        # load previous iterations
        if iteration <= args.resume_iter:
            iter_save_path = os.path.join(save_path, "iter{:d}".format(iteration))
            ckpt_path = os.path.join(iter_save_path, 'epoch{:}.pth'.format(99))
            checkpoint = torch.load(ckpt_path)  # if not exist, raise error?

            print('==> Resuming from checkpoint {:d}..'.format(99))
            model_ls[-1].load_state_dict(checkpoint['net'])
            best_advacc = checkpoint['best_advacc']
            best_correspond_acc = checkpoint["best_correspond_acc"]

            best_epoch = checkpoint["best_epoch"]
            advacc, acc = checkpoint['advacc'], checkpoint["acc"]

        else:
            best_advacc = -1.  # best test accuracy
            best_correspond_acc = -1  # corresponding clean accuracy
            start_epoch = -1  # start from epoch 0 or last checkpoint epoch
            best_epoch = -1

            if args.optimizer == "sgd":
                optimizer = optim.SGD(model_ls[-1].parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
                if args.data == "cifar10":
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                        milestones=[int(0.5 * args.total_epochs), int(0.75 * args.total_epochs)], gamma=0.1)
            elif args.optimizer == "adam":
                optimizer = optim.Adam(model_ls[-1].parameters(), lr=args.lr, weight_decay=5e-4)

            iter_save_path = os.path.join(save_path, "iter{:d}".format(iteration))
            ckpt_path = os.path.join(iter_save_path, 'epoch{:}.pth'.format(args.resume))
            mkdir(iter_save_path)
            if os.path.exists(ckpt_path):
                print('==> Resuming from checkpoint {:d}..'.format(args.resume))
                checkpoint = torch.load(ckpt_path)
                model_ls[-1].load_state_dict(checkpoint['net'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if args.optimizer == "sgd":
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                best_advacc = checkpoint['best_advacc']
                best_correspond_acc = checkpoint["best_correspond_acc"]
                best_epoch = checkpoint["best_epoch"]
                advacc, acc = checkpoint['advacc'], checkpoint["acc"]  # useless
                start_epoch = args.resume

            print('==> Begin training for iteration {:d} ..'.format(iteration))
            for epoch in range(start_epoch + 1, args.total_epochs):
                train(epoch)
                if args.optimizer == "sgd":
                    lr_scheduler.step()

                if (epoch + 1) % args.val_interval == 0 or epoch >= start_epoch + args.total_epochs - 10:
                    acc, advacc = ensemble_test(model_ls, testloader, device, TestAttack, 2333)

                    if advacc > best_advacc:
                        best_advacc = advacc
                        best_correspond_acc = acc
                        best_epoch = epoch

                    if args.save:
                        state = {'net': model_ls[-1].state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'best_correspond_acc': best_correspond_acc,
                                 "best_advacc": best_advacc,
                                 'best_epoch': best_epoch,
                                 "advacc": advacc, "acc": acc,
                                 }
                        if args.optimizer == "sgd":
                            state["lr_scheduler"] = lr_scheduler.state_dict()
                        torch.save(state, os.path.join(iter_save_path, "epoch{:}.pth".format(epoch)))

        print_str = "Iteration {:d}: Best advacc: {:.2f} at {:} epoch, cleanacc: {:.2f}; Last epoch cleanacc: {:.2f} advacc: {:.2f}".format(
                                    iteration, best_advacc, best_epoch, best_correspond_acc, acc, advacc)
        print(print_str)
        print_ls.append(print_str)

    for s in print_ls:
        print(s)
