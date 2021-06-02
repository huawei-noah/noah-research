# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import torch
import torch.nn as nn

class Flatten(nn.Module):

    def forward(self, x):
        """Flatten a Tensor to a Vector

        :param x: Tensor
        :returns: 1D Tensor
        :rtype: Tensor

        """
        return x.view(x.size()[0], -1)


class TED(nn.Module):

    def __init__(self):
        """Initialisation function for the Transformed Encoder Decoder (TED)

        :returns: N/A
        :rtype: N/A

        """
        super().__init__()

        def layer(nIn, nOut, k, s, p, d=1):
            return nn.Sequential(nn.Conv2d(nIn, nOut, k, s, p, d), nn.LeakyReLU(inplace=True))

        self.conv1 = nn.Conv2d(16, 64, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)

        self.mid_net2_1 = MidNet2(in_channels=16)
        self.mid_net4_1 = MidNet4(in_channels=16)
        self.local_net = LocalNet(16)

        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_3 = nn.Conv2d(32, 32, 1)
        self.up_conv1x1_4 = nn.Conv2d(16, 16, 1)

        self.dconv_up4 = LocalNet(128, 64)
        self.dconv_up3 = LocalNet(64, 32)
        self.dconv_up2 = LocalNet(32, 16)
        self.dconv_up1 = LocalNet(32, 3)

        self.conv_fuse1 = nn.Conv2d(208, 16, 1)

        self.glob_net1 = nn.Sequential(
            layer(16, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            layer(64, 64, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),

        )

    def forward(self, x):
        """Forward function for the TED network

        :param x: input image
        :returns: convolutional features
        :rtype: Tensor

        """
        x_in_tile = x.clone()

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.dconv_down5(x)

        x = self.up_conv1x1_1(self.upsample(x))

        if x.shape[3] != conv4.shape[3] and x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv4.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv4.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        del conv4

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        if x.shape[3] != conv3.shape[3] and x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv3.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv3.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        del conv3

        if x.shape[3] != conv2.shape[3] and x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv2.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv2.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        del conv2

        mid_features1 = self.mid_net2_1(conv1)
        mid_features2 = self.mid_net4_1(conv1)
        glob_features = self.glob_net1(conv1)
        glob_features = glob_features.unsqueeze(2)
        glob_features = glob_features.unsqueeze(3)
        glob_features = glob_features.repeat(
            1, 1, mid_features1.shape[2], mid_features1.shape[3])
        fuse = torch.cat(
            (conv1, mid_features1, mid_features2, glob_features), 1)
        conv1_fuse = self.conv_fuse1(fuse)

        if x.shape[3] != conv1.shape[3] and x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv1.shape[2]:
            x = torch.nn.functional.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv1.shape[3]:
            x = torch.nn.functional.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv1_fuse], dim=1)
        del conv1

        x = self.dconv_up1(x)

        out = x+x_in_tile

        return out


class LocalNet(nn.Module):

    def forward(self, x_in):
        """Defines a double convolution

        :param x_in: input convolutional features
        :returns: convolutional features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))

        return x

    def __init__(self, in_channels=16, out_channels=64):
        """Initialisation function

        :param in_channels:  number of input channels
        :param out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)


class MidNet2(nn.Module):

    def forward(self, x_in):
        """Network with dilation rate 2

        :param x_in: input convolutional features        
        :returns: processed convolutional features        
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1((x_in)))
        x = self.lrelu(self.conv2((x)))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)

        return x

    def __init__(self, in_channels=16):
        """FIXME! briefly describe function

        :param in_channels: Input channels
        :returns: N/A
        :rtype: N/A

        """
        super(MidNet2, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 2, 2)


class MidNet4(nn.Module):

    def forward(self, x_in):
        """Network with dilation rate 4

        :param x_in: input convolutional features
        :returns: processed convolutional features
        :rtype: Tensor

        """
        x = self.lrelu(self.conv1((x_in)))
        x = self.lrelu(self.conv2((x)))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)

        return x

    def __init__(self, in_channels=16):
        """FIXME! briefly describe function

        :param in_channels: Input channels
        :returns: N/A
        :rtype: N/A

        """
        super(MidNet4, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 4, 4)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 4, 4)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 4, 4)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 4, 4)

# Model definition
class TEDModel(nn.Module):

    def __init__(self):
        """Initialisation function from the TED model

        :returns: N/A
        :rtype: N/A

        """
        super(TEDModel, self).__init__()

        self.ted = TED()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, img):
        """Forward function for TED

        :param image: image tensor to process
        :returns: convolutional features
        :rtype: Tensor

        """
        output_img= self.ted(img.float())

        return self.final_conv(self.refpad(output_img))
