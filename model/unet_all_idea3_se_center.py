# -*- coding: utf-8 -*-

# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 813.51
# Params size (MB): 191.69
# Estimated Total Size (MB): 1005.77
# ----------------------------------------------------------------

import sys
sys.path.append('D:/Onedrive/Github/Ultrasound')
import torch
import torch.nn as nn
from model.inception_SE_block import Incpetion_SE_block,Incpetion_SE_block_decoder
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

class Center(nn.Module):
    def __init__(self):
        super(Center, self).__init__()
        self.pool1 = nn.MaxPool2d(16)
        self.pool2 = nn.MaxPool2d(8)
        self.pool3 = nn.MaxPool2d(4)
        self.conv1 =nn.Sequential(
            nn.Conv2d(512, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(960,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1,x2,x3,x4):
        pool1 = self.pool1(x1)
        pool2 = self.pool2(x2)
        pool3 = self.pool3(x3)
        inputs = torch.cat([pool1, pool2, pool3, x4], dim=1)
        g1 = self.conv1(x4)
        plus=self.conv2(inputs)
        inputs=self.conv3(torch.cat([plus,inputs],dim=1))
        output = g1 * inputs
        return output
class Unet_all_idea3_se_center(nn.Module):
    def __init__(self,n_class,decay=2):
        super().__init__()

        self.conv_down1=double_conv(3,64)
        self.conv_down2=Incpetion_SE_block(64,decay)
        self.conv_down3=Incpetion_SE_block(128,decay)
        self.conv_down4=Incpetion_SE_block(256,decay)
        self.conv_down5=Center()

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

    def forward(self,input):
        conv1=self.conv_down1(input)
        input=self.maxpool(conv1)

        conv2=self.conv_down2(input)
        input=self.maxpool(conv2)

        conv3=self.conv_down3(input)
        input=self.maxpool(conv3)

        conv4=self.conv_down4(input)
        input=self.maxpool(conv4)

        conv5=self.conv_down5(conv1,conv2,conv3,input)

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
    model=Unet_all_idea3_se_center(1)
    summary(model,(3,224,224))