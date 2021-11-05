# -*- coding: utf-8 -*-
#
import torch
import torch.nn as nn
from model.SE_layer import SELayer
def conv(in_channels,out_channels,kernel=3,pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=kernel,padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class Incpetion_SE_block(nn.Module):
    def __init__(self,in_channel):
        super().__init__()

        self.conv1=conv(in_channel,in_channel//2,1,0)
        self.conv11=conv(in_channel//2,in_channel//2)
        self.conv21=conv(in_channel//2,in_channel//2)
        self.conv31=conv(in_channel//2,in_channel//2)
        self.conv12=conv(in_channel//2,in_channel//2)
        self.conv32=conv(in_channel//2,in_channel//2)
        self.conv33=conv(in_channel//2,in_channel//2)
        l=in_channel//2*3
        r=in_channel*2
        self.conve=conv(l,r)

        self.se=SELayer(in_channel//2)
    
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
        se=self.se(input)
        l1=l12 * se.expand_as(l12)
        l2=l2 * se.expand_as(l2)
        l3=l33 * se.expand_as(l33)
        l=torch.cat([l1,l2,l3],dim=1)
        l=self.conve(l)
        return l


class Incpetion_SE_block_decoder(nn.Module):
    def __init__(self,in_channel):
        super().__init__()

        self.conv1=conv(in_channel,in_channel//2,1,0)
        self.conv11=conv(in_channel//2,in_channel//2)
        self.conv21=conv(in_channel//2,in_channel//2)
        self.conv31=conv(in_channel//2,in_channel//2)
        self.conv12=conv(in_channel//2,in_channel//2)
        self.conv32=conv(in_channel//2,in_channel//2)
        self.conv33=conv(in_channel//2,in_channel//2)
        l=(in_channel//2)*3
        r=in_channel//2
        self.conve=conv(l,r)

        self.se=SELayer(in_channel//2)
    
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
        se=self.se(input)
        l1=l12 * se.expand_as(l12)
        l2=l2 * se.expand_as(l2)
        l3=l33 * se.expand_as(l33)
        l=torch.cat([l1,l2,l3],dim=1)
        l=self.conve(l)
        return l