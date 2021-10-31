# -*- coding: utf-8 -*-
# import sys
# sys.path.append("D:/Onedrive/Github/Ultrasound")
import torch
import torch.nn as nn
from model.nolocal.unet_nonlocal_2D import unet_nonlocal_2D
from torchsummary import summary
def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class Unet_plus_nonlocal(nn.Module):
    def __init__(self,n_class):
        super().__init__()

        self.conv_down1=double_conv(3,64)
        self.conv_down2=double_conv(64,128)
        self.conv_down3=double_conv(128,256)
        self.conv_down4=double_conv(256,512)
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
        self.convpp1=nn.Conv2d(64,3,3,1,1)
        self.convpp2=unet_nonlocal_2D()
        self.conv_out=nn.Conv2d(64,n_class,1)

    def forward(self,input):
        conv1=self.conv_down1(input)
        input=self.maxpool(conv1)

        conv2=self.conv_down2(input)
        input=self.maxpool(conv2)

        conv3=self.conv_down3(input)
        input=self.maxpool(conv3)

        conv4=self.conv_down4(input)
        input=self.maxpool(conv4)

        conv5=self.conv_down5(input)

        up1=self.up1(conv5)
        merge1=torch.cat([conv4,up1],dim=1)
        conv_up1=self.conv_up1(merge1)

        up2=self.up2(conv_up1)
        merge2=torch.cat([conv3,up2],dim=1)
        conv_up2=self.conv_up2(merge2)

        up3=self.up3(conv_up2)
        merge3=torch.cat([conv2,up3],dim=1)
        conv_up3=self.conv_up3(merge3)

        up4=self.up4(conv_up3)
        merge4=torch.cat([conv1,up4],dim=1)
        conv_up4=self.conv_up4(merge4)
        convpp1=self.convpp1(conv_up4)
        convpp2=self.convpp2(convpp1)
        output=self.conv_out(conv_up4)

        return output+convpp2

if __name__=='__main__':
    model=Unet(1)
    summary(model,(3,224,224))