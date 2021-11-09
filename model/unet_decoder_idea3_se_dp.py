# -*- coding: utf-8 -*-
import sys
sys.path.append("D:\\Onedrive\\Github\\Ultrasound")
import torch
import torch.nn as nn
from torchsummary.torchsummary import summary
from model.inception_SE_block import Incpetion_SE_block_decoder
import torch.nn.functional as F
def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class Unet_decoder_idea3_se_dp(nn.Module):
    def __init__(self,n_class,decay=2):
        super().__init__()

        self.conv_down1=double_conv(3,64)
        self.conv_down2=double_conv(64,128)
        self.conv_down3=double_conv(128,256)
        self.conv_down4=double_conv(256,512)
        self.conv_down5=double_conv(512,1024)

        self.maxpool=nn.MaxPool2d(2)

        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv_up1=Incpetion_SE_block_decoder(1024,decay)

        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv_up2=Incpetion_SE_block_decoder(512,decay)

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv_up3=Incpetion_SE_block_decoder(256,decay)

        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up4=Incpetion_SE_block_decoder(128,decay)

        self.conv_out=nn.Conv2d(64,n_class,1)

        self.dp_conv4=nn.Conv2d(512,1,1)
        self.dp_conv3=nn.Conv2d(256,1,1)
        self.dp_conv2=nn.Conv2d(128,1,1)
        self.dp_out=nn.Conv2d(4,1,1)
    def forward(self,input):
        b,c,h,w=input.size()
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
        dp4=self.dp_conv4(conv_up1)
        dp4=F.upsample(dp4, size=(h,w), mode='bilinear')

        up2=self.up2(conv_up1)
        merge2=torch.cat([conv3,up2],dim=1)
        conv_up2=self.conv_up2(merge2)
        dp3=self.dp_conv3(conv_up2)
        dp3=F.upsample(dp3, size=(h,w), mode='bilinear')

        up3=self.up3(conv_up2)
        merge3=torch.cat([conv2,up3],dim=1)
        conv_up3=self.conv_up3(merge3)
        dp2=self.dp_conv2(conv_up3)
        dp2=F.upsample(dp2, size=(h,w), mode='bilinear')

        up4=self.up4(conv_up3)
        merge4=torch.cat([conv1,up4],dim=1)
        conv_up4=self.conv_up4(merge4)
        output=self.conv_out(conv_up4)

        conv_dp=torch.cat([dp4,dp3,dp2,output],dim=1)
        output=self.dp_out(conv_dp)
        
        return output

if __name__=='__main__':
    model=Unet_decoder_idea3_se_dp(1,4)
    summary(model,(3,224,224))