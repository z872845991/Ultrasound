import torch
from torch import nn


class ECA_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)



class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = ECA_layer(out_ch)

    def forward(self, input):
        input = self.se(input)
        return self.conv(input)

class Attention_block(nn.Module):
    def __init__(self,Fg,Fl,Fint):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(Fg,Fint,1,1),
            nn.BatchNorm2d(Fint)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(Fl,Fint,1,1),
            nn.BatchNorm2d(Fint)
        )

        self.sig = nn.Sequential(
            nn.Conv2d(Fint,1,kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        # print(g.shape)
        x1 = self.W_x(x)
        # print(x.shape)
        sig = self.relu(g1+x1)
        # print(sig.shape)
        sig = self.sig(sig)
        # print(sig.shape)
        out = sig*x
        # print(out.shape)
        return out

class Att_Unet(nn.Module):
    def __init__(self,inch ,outch):
        super(Att_Unet, self).__init__()

        self.conv1 = DoubleConv(inch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = Attention_block(512,512,256)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = Attention_block(256,256,128)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = Attention_block(128,128,64)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = Attention_block(64,64,32)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,outch, 1)

    def forward(self,x):
        # 64
        c1=self.conv1(x)
        # print(c1.shape)
        p1=self.pool1(c1)
        # 128
        c2=self.conv2(p1)
        # print(c2.shape)
        p2=self.pool2(c2)
        # 256
        c3=self.conv3(p2)
        # print(c3.shape)
        p3=self.pool3(c3)
        # 512
        c4=self.conv4(p3)
        # print(c4.shape)
        p4=self.pool4(c4)
        # 1024
        c5=self.conv5(p4)
        # print(c5.shape)
        # 512
        up_6= self.up6(c5)
        # print(up_6.shape)
        att1 = self.att1(g=up_6,x=c4)
        # print(att1.shape)
        merge6 = torch.cat([up_6, att1], dim=1)
        # print(merge6.shape)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        # print(up_7.shape)
        att2 = self.att2(g=up_7,x=c3)
        merge7 = torch.cat([up_7, att2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        # print(up_8.shape)
        att3 = self.att3(g=up_8,x=c2)
        merge8 = torch.cat([up_8, att3], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        # print(up_9.shape)
        att4 = self.att4(g=up_9, x=c1)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        #out = nn.Sigmoid()(c10)
        return c10

if __name__ == '__main__':
    model = Att_Unet(3,1)
    input = torch.rand(2,3,512,512)
    output = model(input)

    print(output.shape)








