from tqdm import tqdm
from collections import OrderedDict
import torch
import torch.nn.functional as F
import ipdb


def clean_train(net, trainloader, curr_lr, optimizer, device, epoch):
    train_loss = 0.
    correct = 0
    total = 0
    pbar = tqdm(trainloader)

    pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        net.train()

        outputs = net(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        _, predicted = outputs.max(1)
        train_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        optimizer.step()

        pbar_dic = OrderedDict()
        pbar_dic['Acc'] = '{:2.2f}'.format(100. * correct / total)
        pbar_dic['loss'] = '{:.3f}'.format(train_loss / (batch_idx + 1))
        pbar.set_postfix(pbar_dic)


def train(net, trainloader, curr_lr, optimizer, device, trainattack, epoch, with_clean=False):
    train_loss = 0.
    train_adv_loss = 0.
    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(trainloader)

    # curr_lr = lr_scheduler.get_lr()[0]
    # if optimizer.param_groups[0]['lr'] != lr_scheduler.get_lr()[0]:
    #     print("Different lr!", optimizer.param_groups[0]['lr'], lr_scheduler.get_lr()[0])
    pbar.set_description("Train:{:3d} epoch lr {:.1e}".format(epoch, curr_lr))
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        adv_inp = trainattack.my_attack(net, inputs, targets, rand=True)
        optimizer.zero_grad()
        net.train()
        adv_outputs = net(adv_inp)
        adv_loss = F.cross_entropy(adv_outputs, targets)
        adv_loss.backward()
        _, adv_predicted = adv_outputs.max(1)
        total += targets.size(0)
        adv_correct += adv_predicted.eq(targets).sum().item()

        if with_clean:
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

        optimizer.step()

        train_adv_loss += adv_loss.item()
        if with_clean:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

        pbar_dic = OrderedDict()

        if with_clean:
            pbar_dic['Acc'] = '{:2.2f}'.format(100. * correct / total)
            pbar_dic['loss'] = '{:.3f}'.format(train_loss / (batch_idx + 1))
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar_dic['adv loss'] = '{:.3f}'.format(train_adv_loss / (batch_idx + 1))
        pbar.set_postfix(pbar_dic)


def test(net, testloader, device, testattack, epoch):
    net.eval()
    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(testloader)
    pbar.set_description('Evaluating')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = net(inputs)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if testattack is not None:
            adv_inp = testattack.my_attack(net, inputs, targets)
            with torch.no_grad():
                adv_outputs = net(adv_inp)
            _, adv_predicted = adv_outputs.max(1)
            adv_correct += adv_predicted.eq(targets).sum().item()

        pbar_dic = OrderedDict()
        pbar_dic['Acc'] = '{:2.2f}'.format(100. * correct / total)
        if testattack is not None:
            pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar.set_postfix(pbar_dic)

    acc = 100. * correct / total
    advacc = 100. * adv_correct / total
    return acc, advacc


def ensemble_test(net_ls, testloader, device, testattack, epoch, weight=None):
    for net in net_ls:
        net.eval()
    if weight is None:
        weight = [1. for _ in net_ls]
    normalizing_constant = sum(weight) / len(weight)

    correct = 0
    adv_correct = 0
    total = 0
    pbar = tqdm(testloader)
    pbar.set_description('Evaluating')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = sum([net(inputs) * weight[i] for i, net in enumerate(net_ls)]) / len(net_ls)
            outputs /= normalizing_constant

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        adv_inp = testattack.my_ensemble_attack(net_ls, inputs, targets)
        with torch.no_grad():
            adv_outputs = sum([net(adv_inp) * weight[i] for i, net in enumerate(net_ls)]) / len(net_ls)
            adv_outputs /= normalizing_constant

        _, adv_predicted = adv_outputs.max(1)
        adv_correct += adv_predicted.eq(targets).sum().item()

        pbar_dic = OrderedDict()
        pbar_dic['Acc'] = '{:2.2f}'.format(100. * correct / total)
        pbar_dic['Adv Acc'] = '{:2.2f}'.format(100. * adv_correct / total)
        pbar.set_postfix(pbar_dic)

    acc = 100. * correct / total
    advacc = 100. * adv_correct / total
    return acc, advacc