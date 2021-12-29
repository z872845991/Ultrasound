import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary.torchsummary import summary

'''
En_block使用原本Unet的双卷积,加上reduction为8的se模块
Out_block:dropout 去掉,双卷积，最后加入sigmoid,尝试知，加入sigmoid后会极低,训练代码错误，sigmoid待验证,原因是所选用的损失函数内部做了sigmoid操作
Center: 同En_block
decay率默认2，尝试4
在spaceatt中尝试添加se
增加dp
channel 使用1*1
space 使用3*3
space 使用sigmoid而不是relu
bottleneck与解码器输出xiangjia
'''
class Basic_blocks(nn.Module):
    def __init__(self,in_channel,out_channel,decay=1) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.Sigmoid()
        )
    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x1)
        return x1*x2

class En_blocks(nn.Module):
    def __init__(self, in_channel, out_channel,decay=1):
        super(En_blocks, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//decay, 3, padding=1),
            nn.BatchNorm2d(out_channel//decay),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel//decay, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.channelatt=Channelatt(out_channel,8)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        out=self.channelatt(conv2)
        return out


class Outblock(nn.Module):
    def __init__(self, in_channel):
        super(Outblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel//2, 3, padding=1),
            nn.BatchNorm2d(in_channel//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//2,1,3,padding=1),
        )
    def forward(self, x):
        conv1 = self.conv1(x)
        return conv1

class Channelatt(nn.Module):
    def __init__(self, in_channel,decay=2):
        super(Channelatt, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer(gp)
        return x * se

class Spaceatt(nn.Module):
    def __init__(self, in_channel,decay=2):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3,padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay,in_channel//decay,3,padding=1)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 3,padding=1),
            nn.BatchNorm2d(in_channel//decay),
            nn.Conv2d(in_channel//decay,in_channel//decay,3,padding=1)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 3,padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        #self.softmax = nn.Softmax(in_channel)

    def forward(self, low,high):
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att=att@V
        return self.sig(att)

class Attnblock(nn.Module):
    def __init__(self, in_channel, out_channel,decay=2):
        super(Attnblock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = Basic_blocks(in_channel, out_channel//2)
        self.catt = Channelatt(out_channel//2,decay)
        self.satt = Spaceatt(out_channel//2,decay)
        # self.endconv=nn.Sequential(
        #     nn.Conv2d(out_channel,out_channel,3,padding=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True)
        # )
    def forward(self, high,low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        plusatt=catt*satt
        att=torch.cat([plusatt,catt],dim=1)
        return att


class Teawater_v45(nn.Module):
    def __init__(self, n_class=1,decay=2):
        super(Teawater_v45, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = En_blocks(3, 64,decay)
        self.down_conv2 = En_blocks(64, 128,decay)
        self.down_conv3 = En_blocks(128, 256,decay)
        self.down_conv4 = En_blocks(256, 512,decay)
        self.down_conv5 = En_blocks(512, 1024,decay)
        #self.center = Center()

        self.up_conv4 = Attnblock(1024,512,decay)
        self.up_conv3 = Attnblock(512,256,decay)
        self.up_conv2 = Attnblock(256,128,decay)
        self.up_conv1 = Attnblock(128,64,decay)

        self.dp5=nn.Conv2d(1024,1,1)
        self.dp4=nn.Conv2d(512,1,1)
        self.dp3=nn.Conv2d(256,1,1)
        self.dp2=nn.Conv2d(128,1,1)
        self.out = Outblock(64)

        self.center4=nn.Conv2d(1024,512,1)
        self.center3=nn.Conv2d(1024,256,1)
        self.center2=nn.Conv2d(1024,128,1)
        self.center1=nn.Conv2d(1024,64,1)
    
    def forward(self, inputs):
        b,c,h,w=inputs.size()
        down1 = self.down_conv1(inputs)
        pool1 = self.pool(down1)
        down2 = self.down_conv2(pool1)
        pool2 = self.pool(down2)
        down3 = self.down_conv3(pool2)
        pool3 = self.pool(down3)
        down4 = self.down_conv4(pool3)
        pool4 = self.pool(down4)
        #center = self.center(down1, down2, down3, pool4)
        center=self.down_conv5(pool4)
        center4=self.center4(center)
        center3=self.center3(center)
        center2=self.center2(center)
        center1=self.center1(center)
        center4=F.interpolate(center4,(h//8,w//8),mode='bilinear',align_corners=False)
        center3=F.interpolate(center3,(h//4,w//4),mode='bilinear',align_corners=False)
        center2=F.interpolate(center2,(h//2,w//2),mode='bilinear',align_corners=False)
        center1=F.interpolate(center1,(h,w),mode='bilinear',align_corners=False)
        out5=self.dp5(center)
        out5=F.interpolate(out5,(h,w),mode='bilinear',align_corners=False)
        deco4 = self.up_conv4(center,down4)
        deco4=deco4+center4
        out4=self.dp4(deco4)
        out4=F.interpolate(out4,(h,w),mode='bilinear',align_corners=False)
        deco3 = self.up_conv3(deco4, down3)
        deco3=center3+deco3
        out3=self.dp3(deco3)
        out3=F.interpolate(out3,(h,w),mode='bilinear',align_corners=False)
        deco2 = self.up_conv2(deco3, down2)
        deco2=deco2+center2
        out2=self.dp2(deco2)
        out2=F.interpolate(out2,(h,w),mode='bilinear',align_corners=False)
        deco1 = self.up_conv1(deco2, down1)
        deco1=deco1+center1
        out = self.out(deco1)
        return out,out2,out3,out4,out5
if __name__=='__main__':
    model=Teawater_v45(1,2)
    summary(model,(3,512,512))
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))