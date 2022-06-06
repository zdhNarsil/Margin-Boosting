import torch.nn as nn
import torch.nn.functional as F
import pdb


class ConvNet(nn.Module):
    """
    taken from https://github.com/YisenWang/dynamic_adv_training/blob/master/models.py
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=196, kernel_size=(3, 3), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(196)
        self.conv6 = nn.Conv2d(in_channels=196, out_channels=196, kernel_size=(3, 3), padding=(1, 1))
        self.bn6 = nn.BatchNorm2d(196)

        self.fc1 = nn.Linear(196*4*4, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, stride=2)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2, stride=2)

        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = self.fc2(x)

        return x


# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3, 3), padding=(1, 1))
#         self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=(1, 1))
#         self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=(1, 1))
#         self.conv4 = nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3, 3), padding=(1, 1))
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(in_features=8 * 8 * 256, out_features=512)
#         self.fc2 = nn.Linear(in_features=512, out_features=64)
#         self.Dropout = nn.Dropout(0.25)
#         self.fc3 = nn.Linear(in_features=64, out_features=10)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))  # 32*32*48
#         x = F.relu(self.conv2(x))  # 32*32*96
#         x = self.pool(x)  # 16*16*96
#         x = self.Dropout(x)
#         x = F.relu(self.conv3(x))  # 16*16*192
#         x = F.relu(self.conv4(x))  # 16*16*256
#         x = self.pool(x)  # 8*8*256
#         x = self.Dropout(x)
#         x = x.view(-1, 8 * 8 * 256)  # reshape x
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.Dropout(x)
#         x = self.fc3(x)
#         return x

