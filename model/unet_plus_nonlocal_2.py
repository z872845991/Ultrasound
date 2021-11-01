# -*- coding: utf-8 -*-
# import sys
# sys.path.append("D:/Onedrive/Github/Ultrasound")
import torch
import torch.nn as nn
from torchsummary import summary
import math
from model.nolocal.utils import unetConv2, unetUp
from model.nolocal.nonlocal_layer import NONLocalBlock2D
import torch.nn.functional as F

class unet_nonlocal_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3,
                 is_batchnorm=True, nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters] #[16,32,64,128,256]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal1 = NONLocalBlock2D(in_channels=filters[0], inter_channels=filters[0] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal2 = NONLocalBlock2D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # # upsampling
        # self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        # self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        # self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # # final conv (without any concat)
        # self.final = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        nonlocal1 = self.nonlocal1(maxpool1)

        conv2 = self.conv2(nonlocal1)
        maxpool2 = self.maxpool2(conv2)
        nonlocal2 = self.nonlocal2(maxpool2)

        conv3 = self.conv3(nonlocal2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        #print(center.shape)
        # up4 = self.up_concat4(conv4, center)
        # up3 = self.up_concat3(conv3, up4)
        # up2 = self.up_concat2(conv2, up3)
        # up1 = self.up_concat1(conv1, up2)

        # final = self.final(up1)

        return center   #(1,256,32,32)

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



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