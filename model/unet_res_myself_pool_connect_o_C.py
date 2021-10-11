# -*- coding: utf-8 -*-
#增加了与编码器双卷积的连接
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
class Unet_res_myself(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.origin1=origin(3,64)
        self.origin2=origin(64,128)
        self.origin3=origin(128,256)
        self.origin4=origin(256,512)
        self.conv_down1=double_conv(64,64)
        self.conv_down2=double_conv(128,128)
        self.conv_down3=double_conv(256,256)
        self.conv_down4=double_conv(512,512)
        self.conv_down5=double_conv(512,1024)

        self.maxpool=nn.MaxPool2d(2)

        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv_up1=double_conv(1024,512)

        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv_up2=double_conv(512,256)

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv_up3=double_conv(256,128)

        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up4=double_conv(128,64)

        self.conv_out=nn.Conv2d(64,n_class,1)

    def forward(self,input):
        o1=self.origin1(input)
        conv1=self.conv_down1(o1)    ##
        input=self.maxpool(conv1)
        o2=self.maxpool(o1)

        o2=self.origin2(o2) 
        input=torch.add(o2,input)
        conv2=self.conv_down2(input)
        input=self.maxpool(conv2)
        o3=self.maxpool(o2)

        o3=self.origin3(o3)
        input=torch.add(o3,input)
        conv3=self.conv_down3(input)
        input=self.maxpool(conv3)
        o4=self.maxpool(o3)

        o4=self.origin4(o4)
        input=torch.add(o4,input)
        conv4=self.conv_down4(input)
        input=self.maxpool(conv4)
        

        conv5=self.conv_down5(input)

        up1=self.up1(conv5)
        merge1=torch.cat([o4,up1],dim=1)
        conv_up1=self.conv_up1(merge1)

        up2=self.up2(conv_up1)
        merge2=torch.cat([o3,up2],dim=1)
        conv_up2=self.conv_up2(merge2)

        up3=self.up3(conv_up2)
        merge3=torch.cat([o2,up3],dim=1)
        conv_up3=self.conv_up3(merge3)

        up4=self.up4(conv_up3)
        merge4=torch.cat([o1,up4],dim=1)
        conv_up4=self.conv_up4(merge4)
        output=self.conv_out(conv_up4)

        return output

