import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNET, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = DoubleConv(in_ch, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        self.conv5 = DoubleConv(512, 1024)
        self.upconv6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.upconv7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.upconv8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.upconv9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = nn.MaxPool2d(2)(conv1)
        conv2 = self.conv2(pool1)
        pool2 = nn.MaxPool2d(2)(conv2)
        conv3 = self.conv3(pool2)
        pool3 = nn.MaxPool2d(2)(conv3)
        conv4 = self.conv4(pool3)
        pool4 = nn.MaxPool2d(2)(conv4)
        conv5 = self.conv5(pool4)
        upconv6 = self.upconv6(conv5)
        merge6 = torch.cat([conv4, upconv6], dim=1)
        conv6 = self.conv6(merge6)
        upconv7 = self.upconv7(conv6)
        merge7 = torch.cat([conv3, upconv7], dim=1)
        conv7 = self.conv7(merge7)
        upconv8 = self.upconv8(conv7)
        merge8 = torch.cat([conv2, upconv8], dim=1)
        conv8 = self.conv8(merge8)
        upconv9 = self.upconv9(conv8)
        merge9 = torch.cat([conv1, upconv9], dim=1)
        conv9 = self.conv9(merge9)
        conv10 = self.conv10(conv9)
        return conv10


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.kaiming_normal_(model.weight, mode="fan_in", nonlinearity="relu")
        # nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
