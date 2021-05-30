#无法运行，使用的内存太大了
import torch.nn as nn 
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Dilation_conv(nn.Module):
    def __init__(self,input_channel,output_channel,num_branches=4):
        super(Dilation_conv,self).__init__()
        self.num_branches=num_branches
        self.conv11=nn.Conv2d(input_channel,output_channel,kernel_size=3,dilation=2,padding=2)
        self.conv12=nn.Conv2d(input_channel,output_channel,kernel_size=3,dilation=6,padding=6)
        self.conv13=nn.Conv2d(input_channel,output_channel,kernel_size=3,dilation=10,padding=10)
        self.conv14=nn.Conv2d(input_channel,output_channel,kernel_size=3,dilation=14,padding=14)
       
        self.conv21=nn.Conv2d(output_channel,output_channel,kernel_size=3,dilation=2,padding=2)
        self.conv22=nn.Conv2d(output_channel,output_channel,kernel_size=3,dilation=6,padding=6)
        self.conv23=nn.Conv2d(output_channel,output_channel,kernel_size=3,dilation=10,padding=10)
        self.conv24=nn.Conv2d(output_channel,output_channel,kernel_size=3,dilation=14,padding=14)
  
        self.Bn=nn.BatchNorm2d(output_channel)

        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x11=self.conv11(x)
        x12=self.conv12(x)
        x13=self.conv13(x)
        x14=self.conv14(x)
        # print(x11.size())
        # print(x12.size())
        # print(x13.size())
        # print(x14.size())
        x11=self.Bn(x11)
        x12=self.Bn(x12)
        x13=self.Bn(x13)
        x14=self.Bn(x14)
        x11=(x11+x12+x13+x14)/4
        x=self.relu(x11)   


        x21 = self.conv21(x)
        x22=self.conv22(x)
        x23=self.conv23(x)
        x24=self.conv24(x)
        x21=self.Bn(x21)
        x22=self.Bn(x22)
        x23=self.Bn(x23)
        x24=self.Bn(x24)
        x21=(x21+x22+x23+x24)/4
        x21=self.relu(x21)
        x = self.relu(x21)
        return x

class Unet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.conv_down1=Dilation_conv(3,64)
        self.conv_down2=Dilation_conv(64,128)
        self.conv_down3=Dilation_conv(128,256)
        self.conv_down4=Dilation_conv(256,512)
        self.conv_down5=Dilation_conv(512,1024)

        self.maxpool=nn.MaxPool2d(2)

        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv_up1=Dilation_conv(1024,512)

        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv_up2=Dilation_conv(512,256)

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv_up3=Dilation_conv(256,128)

        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up4=Dilation_conv(128,64)

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
        # print("up1:",up1.size())
        # print("conv4",conv4.size())
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

