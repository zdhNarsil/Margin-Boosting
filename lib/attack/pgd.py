import torch
import torch.nn.functional as F

import numpy as np
import os
import sys
import ipdb

father_dir = os.path.join('/', *os.path.realpath(__file__).split(os.path.sep)[:-2])
if not father_dir in sys.path:
    sys.path.append(father_dir)

# from foolbox import attacks, models
#
# class foolboxPGD():
#     def __init__(self, model, epsilon, max_iters, device, _type='linf', step_size=None):
#         self.epsilon = epsilon
#         self.steps = max_iters
#         self._type = _type
#         self.model = model
#         self.name = 'PGD'
#         self.device = device
#         self.step_size = step_size
#
#     def perturb(self, original_images, labels, random_start=True):
#         self.model.eval()
#
#         fmodel = models.PyTorchModel(self.model, device=self.device, bounds=(0, 1))
#         epsilons = [self.epsilon]
#         if self._type == 'linf':
#             if self.step_size is None:
#                 attack = attacks.LinfProjectedGradientDescentAttack(steps=self.steps, random_start=random_start)
#             else:
#                 attack = attacks.LinfProjectedGradientDescentAttack(abs_stepsize=self.step_size,
#                                                                     steps=self.steps, random_start=random_start)
#         if self._type == 'l2':
#             attack = attacks.L2ProjectedGradientDescentAttack(steps=self.steps, random_start=random_start)
#         if self._type == 'l1':
#             attack = attacks.SparseL1DescentAttack(steps=self.steps, random_start=random_start)
#         advs, _, success = attack(fmodel, original_images, labels, epsilons=epsilons)
#         # criterion = criteria.Misclassification(labels)
#
#         self.model.train()
#         return advs


class IPGD():
    # _mean = torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    # _std = torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps=8 / 255.0, sigma=2 / 255.0, nb_iter=20,
                 norm=np.inf, DEVICE=torch.device('cuda'),
                 mean=torch.tensor(np.array([0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 std=torch.tensor(np.array([1.0]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis]),
                 natural=True):
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = mean.to(DEVICE)
        self._std = std.to(DEVICE)
        self.natural = natural
        self.alpha = 0.1

    def to(self, device):
        self.DEVICE = device
        self._mean = self._mean.to(device)
        self._std = self._std.to(device)
        self.criterion = self.criterion.to(device)

    # def pgd_attack(self, net, inp, label, target=None, rand=True, inv=False):
    #     """pure linf pgd attack"""
    #     net.eval()
    #     x = inp.detach()
    #     if rand:
    #         x = x + torch.zeros_like(inp).uniform_(-self.eps / 2, self.eps / 2)
    #         x = torch.clamp(x, 0., 1.)
    #
    #     for _ in range(self.nb_iter):
    #         x.requires_grad_()
    #         with torch.enable_grad():
    #             pred = net(x)
    #             if target is not None:
    #                 loss = - self.criterion(pred, target)
    #             else:
    #                 loss = self.criterion(pred, label)
    #             if inv:
    #                 loss = - loss
    #
    #         grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
    #         x = x.detach() + self.sigma * grad_sign
    #         x = torch.min(torch.max(x, inp - self.eps), inp + self.eps)
    #         if self.natural:
    #             x = torch.clamp(x, 0., 1.)
    #
    #     return x

    def my_attack(self, net, inp, label, target=None, rand=True, inv=False,
                  perturb=-1, regularize=-1, base_x=None):
        net.eval()
        if base_x is None:
            base_x = inp.detach()

        x = inp.detach()
        if rand:
            x = x + torch.zeros_like(inp).uniform_(-self.eps / 2, self.eps / 2)
            x = torch.clamp(x, 0., 1.)

        for _ in range(self.nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                pred = net(x)
                if target is not None:
                    loss = - self.criterion(pred, target)
                else:
                    loss = self.criterion(pred, label)
                if inv:
                    loss = - loss
                if perturb > 0:
                    noise = torch.rand_like(x.detach()) * perturb
                    loss += (x * noise).mean()
                if regularize > 0:
                    loss += x.norm() ** 2 * regularize

            grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
            x = x.detach() + self.sigma * grad_sign
            x = torch.min(torch.max(x, base_x - self.eps), base_x + self.eps)
            if self.natural:
                x = torch.clamp(x, 0., 1.)

        return x

    def my_mr_oa_attack(self, net, inp, label, other_weight=0., rand=True, inv=False, num_classes=10):
        net.eval()
        base_x = inp.detach()
        x = inp.detach()
        if rand:
            x = x + torch.zeros_like(inp).uniform_(-self.eps / 2, self.eps / 2)
            x = torch.clamp(x, 0., 1.)

        for _ in range(self.nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                output = net(x)
                other_advloss = - F.log_softmax(-output, dim=1) * (1 - F.one_hot(label, num_classes=num_classes))
                other_advloss = other_advloss.sum() / ((num_classes - 1) * len(label))

                loss = self.criterion(output, label) + other_weight * other_advloss
                if inv:
                    loss = - loss

            grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
            x = x.detach() + self.sigma * grad_sign
            x = torch.min(torch.max(x, base_x - self.eps), base_x + self.eps)
            if self.natural:
                x = torch.clamp(x, 0., 1.)
        return x

    def my_ensemble_attack(self, net_ls, inp, label, weight=None, rand=True, inv=False):
        for net in net_ls:
            net.eval()
        if weight is None:
            weight = [1. for _ in net_ls]
        normalizing_constant = sum(weight) / len(weight)
        x = inp.detach()
        if rand:
            x = x + torch.zeros_like(inp).uniform_(-self.eps / 2, self.eps / 2)
            if self.natural:
                x = torch.clamp(x, 0., 1.)

        for _ in range(self.nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                pred = sum([net(x) * weight[i] for i, net in enumerate(net_ls)]) / len(net_ls)
                pred /= normalizing_constant
                loss = self.criterion(pred, label)
                if inv:
                    loss = - loss

            grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
            x = x.detach() + self.sigma * grad_sign
            x = torch.min(torch.max(x, inp - self.eps), inp + self.eps)
            if self.natural:
                x = torch.clamp(x, 0., 1.)
        return x

    def my_ensemble_mr_oa_attack(self, net_ls, inp, label, other_weight=0., rand=True, inv=False, num_classes=10):
        for net in net_ls:
            net.eval()
        x = inp.detach()
        if rand:
            x = x + torch.zeros_like(inp).uniform_(-self.eps / 2, self.eps / 2)
            if self.natural:
                x = torch.clamp(x, 0., 1.)

        for _ in range(self.nb_iter):
            x.requires_grad_()
            with torch.enable_grad():
                pred = sum([net(x) for i, net in enumerate(net_ls)]) / len(net_ls)
                loss = self.criterion(pred, label)

                # other_pred = sum([-net(x) for net in net_ls]) / len(net_ls)
                other_pred = -pred
                other_advloss = - F.log_softmax(other_pred, dim=1) * (1 - F.one_hot(label, num_classes=num_classes))
                other_advloss = other_advloss.sum() / ((num_classes - 1) * len(label))
                loss += other_weight * other_advloss

                if inv:
                    loss = - loss

            grad_sign = torch.autograd.grad(loss, x, only_inputs=True, retain_graph=False)[0].detach().sign()
            x = x.detach() + self.sigma * grad_sign
            x = torch.min(torch.max(x, inp - self.eps), inp + self.eps)
            if self.natural:
                x = torch.clamp(x, 0., 1.)
        return x