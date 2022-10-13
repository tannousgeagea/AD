import torch
import torch.nn as nn
import torch.nn.functional as F



class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.rep_dim = 128
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 3, bias=False, padding=1)
        self.bn2d1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False, padding=1)
        self.bn2d2 = nn.BatchNorm2d(64, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(64, 128, 3, bias=False, padding=1)
        self.bn2d3 = nn.BatchNorm2d(128, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(128 * 16 * 16, self.rep_dim, bias=False)
        self.fc2 = nn.Linear(self.rep_dim, int(self.rep_dim/2), bias=False)
        self.fc3 = nn.Linear(int(self.rep_dim/2), 1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn2d1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2d2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn2d3(x)))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x