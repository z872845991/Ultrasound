# -*- coding: utf-8 -*-
#
import torch
import torch.nn as nn
def conv(in_channels,out_channels,kernel=3,pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
#Inception_dense_block_1  
#三路，1*1加到每一路，密集连接，最后三路结果融合
class Incpetion_dense_block_1(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1=conv(512,512,1,0)
        self.conv11=conv(512,512)
        self.conv21=conv(512,512)
        self.conv31=conv(512,512)
        self.conv12=conv(512,512)
        self.conv32=conv(512,512)
        self.conv33=conv(512,512)

        self.conve=conv(1536,1024)
    
    def forward(self,input):
        input=self.conv1(input)

        l1=self.conv11(input)
        l2=self.conv21(input)
        l3=self.conv31(input)

        l12=self.conv12(l1)
        l32=self.conv32(l3)
        l33=self.conv33(l32)

        # l1=torch.add(l1,l12)
        # l1=torch.add(l1,input)

        # l2=torch.add(l2,input)

        # l3=torch.add(l3,l32)
        # l3=torch.add(l3,l33)
        # l3=torch.add(l3,input)
        l1=l1+l12+input
        l2=l2+input
        l3=l3+l32+l33+input
        l=torch.cat([l1,l2,l3],dim=1)

        l=self.conve(l)
        return l
class Incpetion_dense_block_2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1=conv(512,512,1,0)
        self.conv11=conv(512,512)
        self.conv21=conv(512,512)
        self.conv31=conv(512,512)
        self.conv12=conv(512,512)
        self.conv32=conv(512,512)
        self.conv33=conv(512,512)

        self.conve=conv(1536,1024)
    
    def forward(self,input):
        input=self.conv1(input)

        l1=self.conv11(input)
        l2=self.conv21(input)
        l3=self.conv31(input)

        l12=self.conv12(l1)
        l32=self.conv32(l3)
        l33=self.conv33(l32)

        # l1=torch.add(l1,l12)
        # l1=torch.add(l1,input)

        # l2=torch.add(l2,input)

        # l3=torch.add(l3,l32)
        # l3=torch.add(l3,l33)
        # l3=torch.add(l3,input)
        l1=l1+l12+input
        l2=l2+input
        l3=l3+l32+l33+input
        l=torch.cat([l1,l2,l3],dim=1)

        l=self.conve(l)
        return l
