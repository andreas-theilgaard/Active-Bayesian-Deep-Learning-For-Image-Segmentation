import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F


class Conv_Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, momentum):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_ch, momentum=momentum),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch, momentum):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), Conv_Block(in_ch, out_ch, momentum)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bilinear_method: bool = True, momentum=0.1):
        super().__init__()

        if bilinear_method:
            self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
            self.conv = Conv_Block(in_ch, out_ch, momentum)
        else:
            self.up_sample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = Conv_Block(in_ch, out_ch, momentum)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)

        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]

        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip_connection], dim=1)
        # x = torch.cat([skip_connection,x], dim=1)
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear_method=True, momentum=0.99):
        super(UNET, self).__init__()
        self.factor = 2 if bilinear_method else 1
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.init = Conv_Block(in_ch, 64, momentum)
        self.Encode1 = Encoder(64, 128, momentum)
        self.Encode2 = Encoder(128, 256, momentum)
        self.Encode3 = Encoder(256, 512, momentum)

        self.bottleneck = Encoder(512, 1024 // self.factor, momentum)

        self.Decode1 = Decoder(1024, 512 // self.factor, bilinear_method, momentum)
        self.Decode2 = Decoder(512, 256 // self.factor, bilinear_method, momentum)
        self.Decode3 = Decoder(256, 128 // self.factor, bilinear_method, momentum)
        self.Decode4 = Decoder(128, 64, bilinear_method, momentum)

        # Output
        self.out = nn.Conv2d(64, out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        encode1 = self.init(x)
        encode2 = self.Encode1(encode1)
        encode3 = self.Encode2(encode2)
        encode4 = self.Encode3(encode3)
        bottle = self.bottleneck(encode4)

        up1 = self.Decode1(bottle, encode4)
        up2 = self.Decode2(up1, encode3)
        up3 = self.Decode3(up2, encode2)
        up4 = self.Decode4(up3, encode1)

        # output
        output = self.out(up4)
        return output


def init_weights(model):
    if type(model) in [nn.Conv2d]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    return model


# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms.functional as TF
# import torch.nn.functional as F


# class Conv_Block(nn.Module):
#     def __init__(self, in_ch : int, out_ch : int):
#         super().__init__()

#         self.layers = nn.Sequential(
#             nn.Conv2d(in_ch,out_ch,kernel_size=3,padding=1,bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch,out_ch,kernel_size=3,padding=1,bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.layers(x)

# class Encoder(nn.Module):
#     def __init__(self,in_ch,out_ch):
#         super().__init__()

#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Conv_Block(in_ch,out_ch)
#         )

#     def forward(self,x):
#         return self.maxpool_conv(x)

# class Decoder(nn.Module):
#     def __init__(self,in_ch : int,out_ch : int,bilinear_method : bool =True):
#         super().__init__()

#         if bilinear_method:
#             self.up_sample = nn.Upsample(scale_factor=2,mode='bilinear')
#             self.conv = Conv_Block(in_ch,out_ch)
#         else:
#             self.up_sample = nn.ConvTranspose2d(in_ch,out_ch, kernel_size=2,stride=2)
#             self.conv = Conv_Block(in_ch,out_ch)

#     def forward(self,x,skip_connection):
#         x = self.up_sample(x)

#         diffY = skip_connection.size()[2] - x.size()[2]
#         diffX = skip_connection.size()[3] - x.size()[3]

#         if diffX !=0 or diffY !=0:
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2,
#                     diffY // 2, diffY - diffY // 2])

#         x = torch.cat([x, skip_connection], dim=1)
#         #x = torch.cat([skip_connection,x], dim=1)
#         return self.conv(x)


# class UNET(nn.Module):
#     def __init__(self,in_ch,out_ch,bilinear_method=True):
#         super(UNET,self).__init__()
#         self.factor = 2 if bilinear_method else 1
#         self.in_ch = in_ch
#         self.out_ch = out_ch

#         self.init = (Conv_Block(in_ch,64))
#         self.Encode1 = (Encoder(64,128))
#         self.Encode2 = (Encoder(128,256))
#         self.Encode3 = (Encoder(256,512))

#         self.bottleneck = (Encoder(512,1024//self.factor))

#         self.Decode1 = (Decoder(1024, 512 //self.factor,bilinear_method))
#         self.Decode2 = (Decoder(512, 256 //self.factor,bilinear_method))
#         self.Decode3 = (Decoder(256, 128 //self.factor,bilinear_method))
#         self.Decode4 = (Decoder(128, 64,bilinear_method))

#         # Output
#         self.out = (nn.Conv2d(64,out_ch,kernel_size=1,bias=True))

#     def forward(self,x):
#         encode1 = self.init(x)
#         encode2 = self.Encode1(encode1)
#         encode3 = self.Encode2(encode2)
#         encode4 = self.Encode3(encode3)
#         bottle = self.bottleneck(encode4)

#         up1 = self.Decode1(bottle,encode4)
#         up2 = self.Decode2(up1,encode3)
#         up3 = self.Decode3(up2,encode2)
#         up4 = self.Decode4(up3,encode1)

#         # output
#         output = self.out(up4)
#         return output
