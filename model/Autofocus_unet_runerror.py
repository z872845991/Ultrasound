import torch.nn as nn 
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class Autofocus_conv(nn.Module):
    def __init__(self,input_channel,output_channel,num_branches=4,padding_list=[0,4,8,12],dilation_list=[2,6,10,14]):
        super(Autofocus_conv,self).__init__()
        self.num_branches=num_branches
        self.padding_list=padding_list
        self.dilation_list=dilation_list
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=3,dilation=2)
        self.convatt11=nn.Conv2d(input_channel,int(input_channel/2),kernel_size=3)
        self.convatt12=nn.Conv2d(int(input_channel),out_channels=num_branches,kernel_size=1)
       
        self.bn_list1 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list1.append(nn.BatchNorm2d(output_channel))

        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=3,dilation=2)
        self.convatt21=nn.Conv2d(output_channel,int(output_channel/2),kernel_size=3)
        self.convatt22=nn.Conv2d(int(output_channel/2),num_branches,kernel_size=1)
        
        self.bn_list2 = nn.ModuleList()
        for i in range(self.num_branches):
            self.bn_list2.append(nn.BatchNorm2d(output_channel))

        self.relu=nn.ReLU(inplace=True)

        self.donwsample=nn.Sequential(
            nn.Conv2d(input_channel,output_channel,kernel_size=1),
            nn.BatchNorm2d(output_channel)
        )
    def forward(self,x):
        residual=x[:,:,4:-4,4:-4]
        feature=x.detach()
        att=self.relu(self.convatt11(feature))
        att=self.convatt12(att)
        att=F.softmax(att,dim=1)
        att=att[:,:,1:-1,1:-1]

        x1=self.conv1(x)
        shape=x1.size()
        x1=self.bn1(x1)*att[:,0:1,:,:]
        for i in range(self.num_branches):
            x2=F.conv2d(x,self.conv1.weight,padding=self.padding_list[i],dilation=self.dilation_list[i])
            x2 = self.bn_list1[i](x2)
            x1 += x2* att[:,i:(i+1),:,:].expand(shape)
        
        x=self.relu(x1)

        # compute attention weights for the second autofocus layer
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        att2 = F.softmax(att2, dim=1)
        att2 = att2[:,:,1:-1,1:-1]
        
        # linear combination of different rates
        x21 = self.conv2(x)
        shape = x21.size()
        x21 = self.bn_list2[0](x21)* att2[:,0:1,:,:].expand(shape)
        
        for i in range(1, self.num_branches):
            x22 = F.conv2d(x, self.conv2.weight, padding =self.padding_list[i], dilation=self.dilation_list[i])
            x22 = self.bn_list2[i](x22)
            x21 += x22* att2[:,i:(i+1),:,:].expand(shape)
                
        if self.downsample is not None:
            residual = self.downsample(residual)
     
        x = x21 + residual
        x = self.relu(x)
        return x

class Unet(nn.Module):
    def __init__(self,n_class):
        super().__init__()
        self.conv_down1=Autofocus_conv(3,64)
        self.conv_down2=Autofocus_conv(64,128)
        self.conv_down3=Autofocus_conv(128,256)
        self.conv_down4=Autofocus_conv(256,512)
        self.conv_down5=Autofocus_conv(512,1024)

        self.maxpool=nn.MaxPool2d(2)

        self.up1=nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv_up1=Autofocus_conv(1024,512)

        self.up2=nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv_up2=Autofocus_conv(512,256)

        self.up3=nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv_up3=Autofocus_conv(256,128)

        self.up4=nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv_up4=Autofocus_conv(128,64)

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

