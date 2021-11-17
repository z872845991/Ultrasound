import torch.nn as nn
import torch
import torch.nn.functional as F
from torchsummary.torchsummary import summary


def Double_conv(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_channel, out_channel,
                  kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=False)
    )


class Autofocus_conv(nn.Module):
    def __init__(self, input_channel, output_channel, num_branches=3):
        super(Autofocus_conv, self).__init__()
        self.num_branches = num_branches
        self.conv11 = nn.Conv2d(
            input_channel, output_channel//2, kernel_size=3, dilation=2, padding=2)
        self.conv12 = nn.Conv2d(
            input_channel, output_channel//2, kernel_size=3, dilation=6, padding=6)
        self.conv13 = nn.Conv2d(
            input_channel, output_channel//2, kernel_size=3, dilation=10, padding=10)
        # self.conv14=nn.Conv2d(input_channel,output_channel//2,kernel_size=3,dilation=14,padding=14)
        self.convatt11 = nn.Conv2d(
            input_channel, input_channel//2, kernel_size=3, dilation=2, padding=2)
        self.convatt12 = nn.Conv2d(
            input_channel//2, out_channels=num_branches, kernel_size=1)

        self.conv21 = nn.Conv2d(
            output_channel//2, output_channel, kernel_size=3, dilation=2, padding=2)
        self.conv22 = nn.Conv2d(
            output_channel//2, output_channel, kernel_size=3, dilation=6, padding=6)
        self.conv23 = nn.Conv2d(
            output_channel//2, output_channel, kernel_size=3, dilation=10, padding=10)
        # self.conv24=nn.Conv2d(output_channel//2,output_channel,kernel_size=3,dilation=14,padding=14)

        self.convatt21 = nn.Conv2d(
            output_channel//2, output_channel//2, kernel_size=3, dilation=2, padding=2)
        self.convatt22 = nn.Conv2d(
            output_channel//2, num_branches, kernel_size=1)

        self.Bn = nn.BatchNorm2d(output_channel//2)
        self.Bn2 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=False)

        # self.donwsample=nn.Sequential(
        #    nn.Conv2d(input_channel,output_channel,kernel_size=1),
        #    nn.BatchNorm2d(output_channel)
        # )
    def forward(self, x):
        att = self.relu(self.convatt11(x))
        att = self.convatt12(att)
        att = F.softmax(att, dim=1)
        x11 = self.conv11(x)
        x12 = self.conv12(x)
        x13 = self.conv13(x)
        # x14=self.conv14(x)
        x11 = self.Bn(x11)
        x12 = self.Bn(x12)
        x13 = self.Bn(x13)
        # x14=self.Bn(x14)
        x11 = x11*att[:, 0:1, :, :]+x12*att[:, 1:2, :, :]+x13*att[:, 2:3, :, :]
        x = self.relu(x11)

        # compute attention weights for the second autofocus layer
        feature2 = x.detach()
        att2 = self.relu(self.convatt21(feature2))
        att2 = self.convatt22(att2)
        att2 = F.softmax(att2, dim=1)

        # linear combination of different rates
        x21 = self.conv21(x)
        x22 = self.conv22(x)
        x23 = self.conv23(x)
        # x24=self.conv24(x)
        x21 = self.Bn2(x21)
        x22 = self.Bn2(x22)
        x23 = self.Bn2(x23)
        # x24=self.Bn(x24)
        x21 = x21*att2[:, 0:1, :, :]+x22 * \
            att2[:, 1:2, :, :]+x23*att2[:, 2:3, :, :]
        x21 = self.relu(x21)
        x = self.relu(x21)
        return x


class Autofocus_unet_v2(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.conv_down1 = Double_conv(3, 64)
        self.conv_down2 = Double_conv(64, 128)
        self.conv_down3 = Double_conv(128, 256)
        self.conv_down4 = Double_conv(256, 512)
        self.conv_down5 = Autofocus_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)

        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv_up1 = Double_conv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv_up2 = Double_conv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up3 = Double_conv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up4 = Double_conv(128, 64)

        self.conv_out = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        conv1 = self.conv_down1(input)
        input = self.maxpool(conv1)

        conv2 = self.conv_down2(input)
        input = self.maxpool(conv2)

        conv3 = self.conv_down3(input)
        input = self.maxpool(conv3)

        conv4 = self.conv_down4(input)
        input = self.maxpool(conv4)

        conv5 = self.conv_down5(input)

        up1 = self.up1(conv5)
        merge1 = torch.cat([conv4, up1], dim=1)
        conv_up1 = self.conv_up1(merge1)

        up2 = self.up2(conv_up1)
        merge2 = torch.cat([conv3, up2], dim=1)
        conv_up2 = self.conv_up2(merge2)

        up3 = self.up3(conv_up2)
        merge3 = torch.cat([conv2, up3], dim=1)
        conv_up3 = self.conv_up3(merge3)

        up4 = self.up4(conv_up3)
        merge4 = torch.cat([conv1, up4], dim=1)
        conv_up4 = self.conv_up4(merge4)
        output = self.conv_out(conv_up4)

        return output


if __name__ == '__main__':
    model = Autofocus_unet_v2(1)
    summary(model, (3, 224, 224))
