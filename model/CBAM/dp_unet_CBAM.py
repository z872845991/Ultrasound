import sys
sys.path.append('D:/Onedrive/Github/Ultrasound')
import torch
from torch import nn
import torch.nn.functional as F
from model.CBAM.cbam import CBAM
from torchsummary import summary
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

        self.cbam = CBAM(out_ch)

    def forward(self, input):
        input = self.conv(input)
        output = self.cbam(input)
        return output


class DP_Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DP_Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.dp_conv6 = nn.Conv2d(512,out_ch,1)

        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.dp_conv7 = nn.Conv2d(256,out_ch,1)

        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.dp_conv8 = nn.Conv2d(128,out_ch,1)
        
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.dp_conv9 = nn.Conv2d(64, out_ch, 1)
        self.conv10 = nn.Conv2d(4,out_ch, 1)


    def forward(self,x):
        c1=self.conv1(x)  #b,c,h,w
        p1=self.pool1(c1)  
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        dp1 = self.dp_conv6(c6)
        dp1 = _upsample_like(dp1,c1)
        
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        dp2 = self.dp_conv7(c7)
        dp2 = _upsample_like(dp2,c1)


        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        dp3 = self.dp_conv8(c8)
        dp3 = _upsample_like(dp3,c1)

        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9 = self.conv9(merge9)
        c9 = self.dp_conv9(c9)
        out=self.conv10(torch.cat((dp1,dp2,dp3,c9),dim=1))

        return out,dp1,dp2,dp3


def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='bilinear')
    return src



if __name__=='__main__':
    a = torch.randn((1,3,512,512))
    model = DP_Unet(3,1)
    summary(model,(3,224,224))
    b = model(a)[0]
    print(b.shape)





