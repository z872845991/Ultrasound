import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from torchsummary.torchsummary import summary

'''
En_block使用原本Unet的双卷积
Out_block:dropout 去掉,双卷积，最后加入sigmoid,尝试知，加入sigmoid后会极低
Center: 同En_block
decay率默认2，尝试4
在spaceatt中尝试添加se
attnblock中输出变为satt而不是catt
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

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2


# class Outblock(nn.Module):
#     def __init__(self, in_channel):
#         super(Outblock, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channel, 3, 3, padding=1),
#             nn.BatchNorm2d(3),
#             nn.ReLU(inplace=True)
#         )
#         self.out = nn.Sequential(
#             nn.Conv2d(3, 1, 3, padding=1)
#         )
#     def forward(self, x):
#         conv1 = self.conv1(x)
#         out = self.out(conv1)
#         return out
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
# class Center(nn.Module):
#     def __init__(self):
#         super(Center, self).__init__()
#         self.pool1 = nn.MaxPool2d(16)
#         self.pool2 = nn.MaxPool2d(8)
#         self.pool3 = nn.MaxPool2d(4)
#         self.conv1 =nn.Sequential(
#             nn.Conv2d(960, 1, 3, padding=1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.conv2=nn.Conv2d(960,1024,3,padding=1)
#     def forward(self, x1,x2,x3,x4):
#         pool1 = self.pool1(x1)
#         pool2 = self.pool2(x2)
#         pool3 = self.pool3(x3)
#         inputs = torch.cat([pool1, pool2, pool3, x4], dim=1)
#         g1 = self.conv1(inputs)
#         inputs = self.conv2(inputs)
#         output = g1 * inputs
#         return output

# class Center(nn.Module):
#     def __init__(self):
#         super(Center, self).__init__()
#         self.pool1 = nn.MaxPool2d(16)
#         self.pool2 = nn.MaxPool2d(8)
#         self.pool3 = nn.MaxPool2d(4)
#         self.conv1 =nn.Sequential(
#             nn.Conv2d(512, 1, 3, padding=1),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )
#         self.conv2=nn.Sequential(
#             nn.Conv2d(960,64,3,padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True)
#         )
#         self.conv3=nn.Sequential(
#             nn.BatchNorm2d(1024),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x1,x2,x3,x4):
#         pool1 = self.pool1(x1)
#         pool2 = self.pool2(x2)
#         pool3 = self.pool3(x3)
#         inputs = torch.cat([pool1, pool2, pool3, x4], dim=1)
#         g1 = self.conv1(x4)
#         plus=self.conv2(inputs)
#         inputs=self.conv3(torch.cat([plus,inputs],dim=1))
#         output = g1 * inputs
#         return output


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
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // decay, 1),
            nn.BatchNorm2d(in_channel // decay),
            nn.Conv2d(in_channel // decay, 1, 1),
            nn.Sigmoid()
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
        self.conv = En_blocks(in_channel, out_channel)
        self.catt = Channelatt(out_channel,decay)
        self.satt = Spaceatt(out_channel,decay)

    def forward(self, high,low):
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt(point, catt)
        return satt


class Teawater_v7(nn.Module):
    def __init__(self, n_class=1,decay=2):
        super(Teawater_v7, self).__init__()
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
    model=Teawater_v7(1,2)
    summary(model,(3,224,224))
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))