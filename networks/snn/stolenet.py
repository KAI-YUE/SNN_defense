import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .stoblocks import StoConv2d, StoLinear
from . import StoNN


class StoLeNet_5(StoNN):
    def __init__(self, in_dims, in_channels, num_classes=10):
        super(StoLeNet_5, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = StoConv2d(self.in_channels, 32, kernel_size=5, padding=2)
        # self.bn1 = nn.BatchNorm2d(32, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = StoConv2d(32, 64, kernel_size=5, padding=2)
        # self.bn2 = nn.BatchNorm2d(64, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = StoLinear(self.fc_input_size, 512, bias=False)
        # self.bn3 = nn.BatchNorm1d(512, track_running_stats=dynamic_bn, affine=dynamic_bn)
        self.fc2 = StoLinear(512, num_classes, bias=False)

        self.skip_idx = -2
        # self.skip_idx = 1

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)

        # x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        
        feature = x.view(x.shape[0], -1)
        # self.feature = feature.clone()
        x = F.relu(self.fc1(feature))
        x = self.fc2(x)
        
        return x


def StoLenet5(config):
    sample_size = config.sample_size[0] * config.sample_size[1]
    in_dims = sample_size * config.channels
    return StoLeNet_5(in_dims, config.channels, config.num_classes)
