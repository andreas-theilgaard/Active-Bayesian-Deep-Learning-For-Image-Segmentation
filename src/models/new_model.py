import torch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class UNET(nn.Module):
    def __init__(self, in_ch, out_ch, momentum=0.1, bias=False):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.bias = bias

        self.d1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(64, momentum=momentum),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(64, momentum=momentum),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.d2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(128, momentum=momentum),
            # nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(128, momentum=momentum),
            # nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.d3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(256, momentum=momentum),
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(256, momentum=momentum),
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.d4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(512, momentum=momentum),
            # nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(512, momentum=momentum),
            # nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(1024, momentum=momentum),
            # nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(1024, momentum=momentum),
            # nn.InstanceNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.u1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.u1_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(512, momentum=momentum),
            # nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(512, momentum=momentum),
            # nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.u2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.u2_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(256, momentum=momentum),
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(256, momentum=momentum),
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.u3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.u3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(128, momentum=momentum),
            # nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(128, momentum=momentum),
            # nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.u4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.u4_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(64, momentum=momentum),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=self.bias),
            nn.BatchNorm2d(64, momentum=momentum),
            # nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.out = nn.Conv2d(64, out_ch, kernel_size=1, bias=self.bias)

    def forward(self, x):
        # [n_channels,img_width,img_height]
        down1 = self.d1(x)  # [64,64,64]
        x = self.maxpool(down1)  # [64,32,32]

        down2 = self.d2(x)  # [128,32,32]
        x = self.maxpool(down2)  # [128,16,16]

        down3 = self.d3(x)  # [256,16,16]
        x = self.maxpool(down3)  # [256,8,8]

        down4 = self.d4(x)  # [512,8,8]
        x = self.maxpool(down4)  # [512,4,4]

        bottleneck = self.bottleneck(x)  # [1024,4,4]

        # Decoder
        up1 = self.u1(bottleneck)  # [512,8,8]
        x = torch.cat([up1, down4], dim=1)  # [1024,8,8]
        x = self.u1_conv(x)  # [512,8,8]

        up2 = self.u2(x)  # [256,16,16]
        x = torch.cat([up2, down3], dim=1)  # [512,16,16]
        x = self.u2_conv(x)  # [256,16,16]

        up3 = self.u3(x)  # [128,32,32]
        x = torch.cat([up3, down2], dim=1)  # [256,32,32]
        x = self.u3_conv(x)  # [128,32,32]

        up4 = self.u4(x)  # [64,64,64]
        x = torch.cat([up4, down1], dim=1)  # [128,64,64]
        x = self.u4_conv(x)  # [64,64,64]

        # output layer
        output = self.out(x)
        return output


class init_store:
    def __init__(self, init_):
        self.init_ = init_


global store_init
store_init = init_store(init_=0)
store_init.init_


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d]:
        # if init_==0:
        #    nn.init.kaiming_normal_(model.weight, mode="fan_in", nonlinearity="relu")
        #    nn.init.zeros_(model.bias)
        # if init_==1:
        if store_init.init_ == 0:
            nn.init.xavier_uniform_(model.weight)
        elif store_init.init_ == 1:
            nn.init.kaiming_normal_(model.weight, mode="fan_in", nonlinearity="relu")
        elif store_init.init_ == 2:
            nn.init.kaiming_uniform_(model.weight)
        elif store_init.init_ == 3:
            nn.init.xavier_normal_(model.weight)
        try:
            nn.init.normal_(model.bias, std=0.001)
        except:
            pass


# if __name__ == "__main__":
#     model= UNET(in_ch=1,out_ch=1)
#     img = torch.rand(4,1,64,64)
#     out = model(img)
#     print(img.shape)
#     print(out.shape)
