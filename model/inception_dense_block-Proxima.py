# -*- coding: utf-8 -*-
#先不使用ppt中的中间层试试
import torch
import torch.nn as nn
def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
def origin(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,1,padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
class Incpetion_dense_block(nn.Moudle):
    def __init__(self) -> None:
        super().__init__()

        self.conv1=nn.Conv2d(512,512,1,padding=0)

        self.conv21=nn.Conv2d(512,512,3,padding=1)
        self.conv211=nn.Conv2d(512,512,3,padding=1)

        self.conv22=nn.Conv2d(512,512,3,padding=1)

        self.conv23=nn.Conv2d(512,512,3,padding=1)
        self.conv233=nn.Conv2d(512,512,3,padding=1)
        self.conv2333=nn.Conv2d(512,512,3,padding=1)
    
    def forward(self,input):
        input=self.conv1(input)

        l1=self.conv21(input)
        l2=self.conv22(input)
        l3=self.conv23(input)

        l11=self.conv21(l1)
        l31=self.conv233(l3)
        l32=self.conv2333(l31)

        s1=torch.add(input,l1)
        s1=torch.add(s1,l11)

        s2=torch.add(input,l2)

        s3=torch.add(input,l3)
        s3=torch.add(s3,l31)
        s3=torch.add(s3,l32)
