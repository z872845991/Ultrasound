# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.activation import ReLU
class Denseblock(nn.Module):
    def __init__(self,n_layers,input_channel,k):
        self.n_layers=n_layers
        self.conv=nn.ModuleList()
        for i in range(self.n_layers):
            in_channel=input_channel+k*i
            self.conv.append(nn.Conv2d(in_channel,k,kernel_size=3,padding=1))
    def forward(self,x):
        for i in range(self.n_layers):
            feature=self.conv[i](x)
            x=torch.cat([feature,x],dim=1)
        return x
def Transition_down(inputs,outputs):
    return nn.Sequential(
        nn.BatchNorm2d(inputs),
        nn.ReLU(inplace=True),
        nn.Conv2d(inputs,outputs,kernel_size=1,padding=0),
        #缺少一个dropout
        nn.MaxPool2d(2)
    )
class Tiramisu(nn.Module):
    def __init__(self,input_channel,k=16,n_class=1):
        n_layers=[4,5,7,10,12,15,12,10,7,5,4]
        l=len(n_layers)/2
        self.l=l
        ## downsample
        self.conv1=nn.Conv2d(input_channel,48,kernel_size=3,padding=1)
        filters=48
        downfilter=[]
        self.down=nn.ModuleList()
        self.td=nn.ModuleList()
        for i in range(l):
            self.down.append(Denseblock(n_layers[i],filters,k))
            filters+=k*n_layers[i]
            downfilter.append(filters)
            self.td.append(Transition_down(filters,filters))
        # self.down1=Denseblock(n_layers[0],48,k)
        # self.td1=Transition_down(112,112)
        # self.down2=Denseblock(n_layers[1],112,k)
        # self.td2=Transition_down(192,192)
        # self.down3=Denseblock(n_layers[2],192,k)
        # self.td3=Transition_down(304,304)
        # self.down4=Denseblock(n_layers[3],304,k)
        # self.td4=Transition_down(464,464)
        # self.down5=Denseblock(n_layers[4],464,k)
        # self.td5=Transition_down(656,656)
        # self.down6=Denseblock(n_layers[5],656,k)
        
        ##bottleneck
        self.bottle=Denseblock(n_layers[l],filters,k)
        ##upsample
        self.up=nn.ModuleList()
        self.updb=nn.ModuleList()
        for i in range(l):
            upfilter=n_layers[i+l]*k
            self.up.append(nn.ConvTranspose2d(upfilter,upfilter,3,2,output_padding=1))
            densefilter=upfilter+downfilter[l-1-i]
            self.updb.append(Denseblock(n_layers[i+l+1],densefilter,k))
        
        # self.up5=nn.ConvTranspose2d(80,3,2)
        # self.updb5=Denseblock(n_layers[6],input_channel,k)
        # self.up4=nn.ConvTranspose2d(2,2)
        # self.updb4=Denseblock(n_layers[7],input_channel,k)
        # self.up3=nn.ConvTranspose2d(in,out,2,2)
        # self.updb3=Denseblock(n_layers[8],input_channel,k)
        # self.up2=nn.ConvTranspose2d(in,out,2,2)
        # self.updb2=Denseblock(n_layers[9],input_channel,k)
        # self.up1=nn.ConvTranspose2d(in,out,2,2)
        # self.updb1=Denseblock(n_layers[10],input_channel,k)
        
        self.conv2=nn.Conv2d(input_channel,n_class,kernel_size=1)
        
    def forward(self,x):
        feature=self.conv1(x)
        downc=[]
        for i in range(self.l):
            DB=self.down[feature]
            feature=torch.cat([feature,DB],dim=1)
            downc.append(feature)
            feature=self.td[i](feature)
        
        bottles=self.bottle(feature)
        
        for i in range(self.l):
            tu=self.up[i](bottles)
            feature=torch.cat([downc[self.l-1-i],tu],dim=1)
            feature=self.updb[i](feature)
        
        output=self.conv2(feature)
        return output
