import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from PIL import Image


def get_MNIST(data_path, batch_size, train_shuffle=True, num_workers=2):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    testset = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, testset, trainloader, testloader


def get_SVHN(data_path, batch_size, train_shuffle=True, num_workers=2):
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.SVHN(
        data_path, split='train', transform=train_transform, download=True)
    testset = torchvision.datasets.SVHN(
        data_path, split='test', transform=test_transform, download=True)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=train_shuffle, pin_memory=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    return trainset, testset, trainloader, testloader


def get_CIFAR100(data_path, batch_size, train_shuffle=True, num_workers=2):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # follow AWP repo
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
    testset = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, testset, trainloader, testloader


def get_CIFAR10(data_path, batch_size, aug, train_shuffle=True, with_index=False, num_workers=2):
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if with_index:
        CIFAR10 = IndexCIFAR10
    else:
        CIFAR10 = torchvision.datasets.CIFAR10

    # 只改train data
    trainset = CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, testset, trainloader, testloader


class IndexCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(IndexCIFAR10, self).__init__(root, train=train, transform=transform,
                              target_transform=target_transform, download=download)
        # unify the interface
        if not hasattr(self, 'data'):  # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image (this part if from official implementation)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    @property
    def num_classes(self):
        return 10