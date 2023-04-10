import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib


class Conv_Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, momentum, enable_dropout=False, dropout_prob=0.5):
        super().__init__()

        self.layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)]
        if enable_dropout:
            self.layers.append(nn.Dropout(p=dropout_prob))
        self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True))
        if enable_dropout:
            self.layers.append(nn.Dropout(p=dropout_prob))
        self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))
        self.layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        momentum,
        enable_dropout,
        dropout_prob,
        enable_pool_dropout,
        pool_dropout_prob,
    ):
        super().__init__()

        self.maxpool_conv = [nn.MaxPool2d(kernel_size=2, stride=2)]
        if enable_pool_dropout:
            self.maxpool_conv.append(nn.Dropout(p=pool_dropout_prob))
        self.maxpool_conv.append(Conv_Block(in_ch, out_ch, momentum, enable_dropout, dropout_prob))

        self.maxpool_conv = nn.Sequential(*self.maxpool_conv)

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bilinear_method: bool = True,
        momentum=0.1,
        enable_dropout=False,
        dropout_prob=0.5,
    ):
        super().__init__()

        if bilinear_method:
            self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
            self.conv = Conv_Block(in_ch, out_ch, momentum, enable_dropout, dropout_prob)
        else:
            self.up_sample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            self.conv = Conv_Block(in_ch, out_ch, momentum, enable_dropout, dropout_prob)

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
    def __init__(
        self,
        in_ch,
        out_ch,
        bilinear_method=True,
        momentum=0.1,
        enable_dropout=False,
        dropout_prob=0.5,
        enable_pool_dropout=False,
        pool_dropout_prob=0.5,
    ):

        super(UNET, self).__init__()
        self.factor = 2 if bilinear_method else 1
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.init = Conv_Block(in_ch, 64, momentum, enable_dropout=False, dropout_prob=dropout_prob)
        self.Encode1 = Encoder(
            64,
            128,
            momentum,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
            enable_pool_dropout=enable_pool_dropout,
            pool_dropout_prob=pool_dropout_prob,
        )
        self.Encode2 = Encoder(
            128, 256, momentum, enable_dropout, dropout_prob, enable_pool_dropout, pool_dropout_prob
        )
        self.Encode3 = Encoder(
            256, 512, momentum, enable_dropout, dropout_prob, enable_pool_dropout, pool_dropout_prob
        )

        self.bottleneck = Encoder(
            512,
            1024 // self.factor,
            momentum,
            enable_dropout,
            dropout_prob,
            enable_pool_dropout,
            pool_dropout_prob,
        )

        self.Decode1 = Decoder(
            1024,
            512 // self.factor,
            bilinear_method,
            momentum,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
        )
        self.Decode2 = Decoder(
            512,
            256 // self.factor,
            bilinear_method,
            momentum,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
        )
        self.Decode3 = Decoder(
            256,
            128 // self.factor,
            bilinear_method,
            momentum,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
        )
        self.Decode4 = Decoder(
            128,
            64,
            bilinear_method,
            momentum,
            enable_dropout=enable_dropout,
            dropout_prob=dropout_prob,
        )

        # Output
        self.out = nn.Conv2d(64, out_ch, kernel_size=1, bias=True)

    def forward(self, x, features=False):
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
        if features:
            return up4
        output = self.out(up4)
        return output


def init_weights(model):
    if type(model) in [nn.Conv2d, nn.ConvTranspose2d]:
        nn.init.xavier_normal_(model.weight)
        nn.init.normal_(model.bias, std=0.001)


###

# class Conv_Block(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, momentum, enable_dropout=False, dropout_prob=0.5):
#         super().__init__()

#         self.layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)]
#         if enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_prob))
#         self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))
#         self.layers.append(nn.ReLU(inplace=True))
#         self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True))
#         if enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_prob))
#         self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))
#         self.layers.append(nn.ReLU(inplace=True))

#         self.layers = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.layers(x)

# class Conv_Block_Up(nn.Module):
#     def __init__(self, in_ch: int, out_ch: int, momentum, enable_dropout=False, dropout_prob=0.5):
#         super().__init__()

#         self.layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)]
#         if enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_prob))
#         self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))
#         self.layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True))
#         if enable_dropout:
#             self.layers.append(nn.Dropout(p=dropout_prob))
#         self.layers.append(nn.BatchNorm2d(out_ch, momentum=momentum))

#         self.layers = nn.Sequential(*self.layers)

#     def forward(self, x):
#         return self.layers(x)

# class Encoder(nn.Module):
#     def __init__(
#         self,
#         in_ch,
#         out_ch,
#         momentum,
#         enable_dropout,
#         dropout_prob,
#         enable_pool_dropout,
#         pool_dropout_prob,
#     ):
#         super().__init__()

#         self.maxpool_conv = [nn.MaxPool2d(kernel_size=2, stride=2)]
#         if enable_pool_dropout:
#             self.maxpool_conv.append(nn.Dropout(p=pool_dropout_prob))
#         self.maxpool_conv.append(Conv_Block(in_ch, out_ch, momentum, enable_dropout, dropout_prob))

#         self.maxpool_conv = nn.Sequential(*self.maxpool_conv)

#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Decoder(nn.Module):
#     def __init__(
#         self,
#         in_ch: int,
#         out_ch: int,
#         bilinear_method: bool = True,
#         momentum=0.1,
#         enable_dropout=False,
#         dropout_prob=0.5,
#     ):
#         super().__init__()

#         if bilinear_method:
#             self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
#             self.conv = Conv_Block_Up(in_ch, out_ch, momentum, enable_dropout, dropout_prob)
#         else:
#             self.up_sample = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
#             self.conv = Conv_Block_Up(in_ch, out_ch, momentum, enable_dropout, dropout_prob)

#     def forward(self, x, skip_connection):
#         x = self.up_sample(x)

#         diffY = skip_connection.size()[2] - x.size()[2]
#         diffX = skip_connection.size()[3] - x.size()[3]

#         if diffX != 0 or diffY != 0:
#             x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

#         x = torch.cat([x, skip_connection], dim=1)
#         # x = torch.cat([skip_connection,x], dim=1)
#         return self.conv(x)


# class UNET(nn.Module):
#     def __init__(
#         self,
#         in_ch,
#         out_ch,
#         bilinear_method=True,
#         momentum=0.1,
#         enable_dropout=False,
#         dropout_prob=0.5,
#         enable_pool_dropout=False,
#         pool_dropout_prob=0.5,
#     ):

#         super(UNET, self).__init__()
#         self.factor = 2 if bilinear_method else 1
#         self.in_ch = in_ch
#         self.out_ch = out_ch

#         self.init = Conv_Block(in_ch, 64, momentum, enable_dropout, dropout_prob)
#         self.Encode1 = Encoder(
#             64, 128, momentum, enable_dropout, dropout_prob, enable_pool_dropout, pool_dropout_prob
#         )
#         self.Encode2 = Encoder(
#             128, 256, momentum, enable_dropout, dropout_prob, enable_pool_dropout, pool_dropout_prob
#         )
#         self.Encode3 = Encoder(
#             256, 512, momentum, enable_dropout, dropout_prob, enable_pool_dropout, pool_dropout_prob
#         )

#         self.bottleneck = Encoder(
#             512,
#             1024 // self.factor,
#             momentum,
#             enable_dropout,
#             dropout_prob,
#             enable_pool_dropout,
#             pool_dropout_prob,
#         )

#         self.Decode1 = Decoder(
#             1024, 512 // self.factor, bilinear_method, momentum, enable_dropout, dropout_prob
#         )
#         self.Decode2 = Decoder(
#             512, 256 // self.factor, bilinear_method, momentum, enable_dropout, dropout_prob
#         )
#         self.Decode3 = Decoder(
#             256, 128 // self.factor, bilinear_method, momentum, enable_dropout, dropout_prob
#         )
#         self.Decode4 = Decoder(128, 64, bilinear_method, momentum, enable_dropout, dropout_prob)

#         # Output
#         self.out = nn.Conv2d(64, out_ch, kernel_size=1, bias=True)

#     def forward(self, x):
#         encode1 = self.init(x)
#         encode2 = self.Encode1(encode1)
#         encode3 = self.Encode2(encode2)
#         encode4 = self.Encode3(encode3)
#         bottle = self.bottleneck(encode4)

#         up1 = self.Decode1(bottle, encode4)
#         up2 = self.Decode2(up1, encode3)
#         up3 = self.Decode3(up2, encode2)
#         up4 = self.Decode4(up3, encode1)

#         # output
#         output = self.out(up4)
#         return output


# def init_weights(model):
#     if type(model) in [nn.Conv2d]:
#         nn.init.xavier_uniform_(model.weight)
#         nn.init.zeros_(model.bias)
#     return model
