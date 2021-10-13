# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from dataset.Fetus import FetusDataset
from model.unet_res_myself_pool_connect_o_C import Unet_res_myself
#from model.unet import Unet
from tools.metrics import dice_coef,iou_score,get_accuracy,get_precision,get_specificity,get_recall,get_F1
from tools.utils import AverageMeter

 
from torch.utils.tensorboard import SummaryWriter
def double_conv(in_channels,out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
class Unet(nn.Module):
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

if __name__=="__main__":
    num_epoch=1
    batch_size={'train':25,
                'val':2
                }
    train_path='F:/data21'
    val_path='F:/data22'

    x_transforms = transforms.Compose([
        transforms.Resize((512,512)),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # mask只需要转换为tensor
    y_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        # transforms.CenterCrop(512),
        transforms.ToTensor()
    ])
    train_dataset=FetusDataset(train_path,transform=x_transforms,target_transform=y_transforms)
    val_dataset=FetusDataset(val_path,transform=x_transforms,target_transform=y_transforms)
    dataloaders={
        'train':DataLoader(train_dataset,batch_size=batch_size['train'],shuffle=True,num_workers=0),
        'val':DataLoader(val_dataset,batch_size=batch_size['val'],shuffle=True,num_workers=0)
                }
        #tensor board

    tb=SummaryWriter()
    network=Unet(n_class=1)
#取出训练用图
    images,_,_=next(iter(dataloaders['train']))
    grid=torchvision.utils.make_grid(images)
#想用tensorboard看什么，你就tb.add什么。image、graph、scalar等
    tb.add_image('images', grid)
    tb.add_graph(model=network,input_to_model=images)
    tb.close()
    exit(0)