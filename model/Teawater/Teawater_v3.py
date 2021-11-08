import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary

'''
去掉En_block的dropout
Out_block:dropout 去掉
Center: 采用下采样
跳连后的两个注意力暂且不变
'''

class En_blocks(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(En_blocks, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        return conv2


class Outblock(nn.Module):
    def __init__(self, in_channel):
        super(Outblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, 3, 3, padding=1)
        )
        self.out = nn.Sequential(
            nn.Conv2d(3, 1, 3, padding=1)
        )
    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.out(conv1)
        return out


class Center(nn.Module):
    def __init__(self):
        super(Center, self).__init__()
        self.pool1 = nn.MaxPool2d(16)
        self.pool2 = nn.MaxPool2d(8)
        self.pool3 = nn.MaxPool2d(4)
        self.conv1 = nn.Conv2d(960, 1, 3, padding=1)
        self.conv2=nn.Conv2d(960,1024,3,padding=1)
    def forward(self, x):
        x1, x2, x3, x4 = x
        pool1 = self.pool1(x1)
        pool2 = self.pool2(x2)
        pool3 = self.pool3(x3)
        inputs = torch.cat([pool1, pool2, pool3, x4], dim=1)
        g1 = self.conv1(inputs)
        inputs = self.conv2(inputs)
        output = g1 * inputs
        return output


class Channelatt(nn.Module):
    def __init__(self, in_channel):
        super(Channelatt, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // 2, in_channel, 3, padding=1),
            nn.Sigmoid()
        )
        self.gpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        gp = self.gpool(x)
        se = self.layer(gp)
        return x * se


class Spaceatt(nn.Module):
    def __init__(self, in_channel):
        super(Spaceatt, self).__init__()
        self.Q = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.BatchNorm2d(in_channel // 2),
            nn.Conv2d(in_channel // 2, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.K = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.BatchNorm2d(in_channel // 2)
        )
        self.V = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 2, 1),
            nn.BatchNorm2d(in_channel // 2)
        )
        self.sig = nn.Sequential(
            nn.Conv2d(in_channel // 2, in_channel, 1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.softmax = nn.Softmax(in_channel)

    def forward(self, x):
        low, high = x
        b, c, h, w = low.size()
        Q = self.Q(low)
        K = self.K(low)
        V = self.V(high)
        att = Q * K
        att = att+V
        return self.sig(att)


class Attnblock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Attnblock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channel, out_channel, 2, stride=2)
        self.conv = En_blocks(in_channel, out_channel)
        self.catt = Channelatt(out_channel)
        self.satt = Spaceatt(out_channel)

    def forward(self, x):
        high, low = x
        up = self.upsample(high)
        concat = torch.cat([up, low], dim=1)
        point = self.conv(concat)
        catt = self.catt(point)
        satt = self.satt([point, catt])
        return satt


class Teawater_v1(nn.Module):
    def __init__(self, n_class=1):
        super(Teawater_v1, self).__init__()
        self.pool = nn.MaxPool2d(2)

        self.down_conv1 = En_blocks(3, 64)
        self.down_conv2 = En_blocks(64, 128)
        self.down_conv3 = En_blocks(128, 256)
        self.down_conv4 = En_blocks(256, 512)

        self.center = Center()

        self.up_conv4 = Attnblock(1024,512)
        self.up_conv3 = Attnblock(512,256)
        self.up_conv2 = Attnblock(256,128)
        self.up_conv1 = Attnblock(128,64)

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
        center = self.center([down1, down2, down3, pool4])
        deco4 = self.up_conv4([center, down4])
        deco3 = self.up_conv3([deco4, down3])
        deco2 = self.up_conv2([deco3, down2])
        deco1 = self.up_conv1([deco2, down1])
        out = self.out(deco1)
        return out
if __name__=='__main__':
    model=Teawater(1)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))