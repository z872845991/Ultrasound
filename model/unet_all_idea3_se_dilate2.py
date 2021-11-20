# -*- coding: utf-8 -*-

# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 813.51
# Params size (MB): 191.69
# Estimated Total Size (MB): 1005.77
# ----------------------------------------------------------------

# import sys
# sys.path.append('D:/Onedrive/Github/Ultrasound')
import torch
import torch.nn as nn
from model.inception_SE_block_dilate import Incpetion_SE_block_dilate,Incpetion_SE_block_decoder_dilate
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
class Unet_all_idea3_se_dilate2(nn.Module):
    def __init__(self,n_class,decay=2):
        super().__init__()

        self.conv_down1=double_conv(3,64)
        self.conv_down2=Incpetion_SE_block_dilate(64,decay)
        self.conv_down3=Incpetion_SE_block_dilate(128,decay)
        self.conv_down4=Incpetion_SE_block_dilate(256,decay)
        self.conv_down5=Incpetion_SE_block_dilate(512,decay)

        self.maxpool=nn.MaxPool2d(2)

        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv_up1=Incpetion_SE_block_decoder_dilate(1024,decay)

        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv_up2=Incpetion_SE_block_decoder_dilate(512,decay)

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv_up3=Incpetion_SE_block_decoder_dilate(256,decay)

        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up4=Incpetion_SE_block_decoder_dilate(128,decay)

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
        output=self.conv_out(conv_up4)

        return output

if __name__=='__main__':
    model=Unet_all_idea3_se_dilate2(1)
    summary(model,(3,224,224))