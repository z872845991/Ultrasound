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
更改QKV的实现方式
'''

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
            nn.Conv2d(in_channel, in_channel // decay, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // decay, in_channel, 3, padding=1),
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
        self.decay=decay
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            Channelatt(in_channel // decay)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            Channelatt(in_channel // decay)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // decay, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(in_channel)

    def forward(self, low,high):
        b,c,h,w=low.size()
        c=c//self.decay
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)

        part=(h//4)**2
        Q_=Q.view(b,c,4,4,part)
        K_=K.view(b,c,4,4,part)
        V_=V.view(b,c,4,4,part)
        Q_=Q.permute(0,4,1,2,3).contiguous().view(b*part,-1)
        K_=K.permute(1,2,3,4,0).contiguous().view(-1,part*b)
        K_=K.permute(1,2,3,4,0).contiguous().view(-1,part*b)
        # Q_=Q.permute(1,2,3,0).contiguous().view(c,-1)
        # K_=K.permute(0,2,3,1).contiguous().view(-1,c)
        # V_=V.permute(0,2,3,1).contiguous().view(-1,c)
        att=(K@Q)@V
        att=att.permute(1,0).contiguous().view(b,part,c,4,4)
        return self.sig(att)


class Attnblock(nn.Module):
    def __init__(self, in_channel, out_channel,decay=2):
        super(Attnblock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = En_blocks(in_channel, out_channel)
        self.catt = Channelatt(out_channel,decay)
        self.satt = Spaceatt(out_channel,decay)

    def forward(self, high,low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        print("1",point.shape)
        print("2",catt.shape)
        satt = self.satt(point, catt)
        return satt+catt


class Teawater_v16(nn.Module):
    def __init__(self, n_class=1,decay=2):
        super(Teawater_v16, self).__init__()
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

        self.out = Outblock(64)

    def forward(self, inputs):
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

        deco4 = self.up_conv4(center,down4)
        deco3 = self.up_conv3(deco4, down3)
        deco2 = self.up_conv2(deco3, down2)
        deco1 = self.up_conv1(deco2, down1)
        out = self.out(deco1)
        return out
if __name__=='__main__':
    model=Teawater_v16(1,2)
    summary(model,(3,512,512))
    #print('# generator parameters:', sum(param.numel() for param in model.parameters()))