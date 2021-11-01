from typing import ForwardRef
import torch
import torch.nn as nn
import  torch.nn.functional as F
from torch.nn.modules.pooling import AdaptiveAvgPool2d

class block_1(nn.Module):
    def __init__(self,in_channels):
        super(block_1,self).__init__()
        self.channel=nn.Sequential(
            AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels,in_channels//2,3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2,in_channels,3),
            nn.Sigmoid()
        )

    def forward(self,input):
        