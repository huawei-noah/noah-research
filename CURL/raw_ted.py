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

# Shared with the RGB backbone: these four were duplicated verbatim.
from rgb_ted import Flatten, LocalNet, MidNet2, MidNet4


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

        self.dconv_down1 = LocalNet(4, 16)
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
        self.dconv_up1 = LocalNet(32, 16)

        self.conv_last = LocalNet(16, 64)

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
        x_in_tile = x.repeat(1, 4, 1, 1)

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
        x = x+x_in_tile

        out = self.conv_last(x)

        return out


class SimpleUpsampler(nn.Sequential):

    def __init__(self, scale):
        """Pixelshuffle upsampling

        :param scale: scale of upsampling
        :returns: upsampled image
        :rtype: Tensor

        """
        m = []
        m.append(nn.PixelShuffle(scale))
        super(SimpleUpsampler, self).__init__(*m)


def DownSamplingShuffle(x):
    """Pixelshuffle downsample

    :param x: RAW image 
    :returns: RAW image shuffled to 4 channels
    :rtype: Tensor

    """
    [N, C, W, H] = x.shape
    x1 = x[:, :, 0:W:2, 0:H:2]
    x2 = x[:, :, 0:W:2, 1:H:2]
    x3 = x[:, :, 1:W:2, 0:H:2]
    x4 = x[:, :, 1:W:2, 1:H:2]

    return torch.cat((x1, x2, x3, x4), 1)


# Model definition
class TEDModel(nn.Module):

    def __init__(self):
        """Initialisation function from the TED model

        :returns: N/A
        :rtype: N/A

        """
        super(TEDModel, self).__init__()

        self.ted = TED()
        self.final_conv = nn.Conv2d(16, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, image):
        """Forward function for TED

        :param image: image tensor to process
        :returns: convolutional features
        :rtype: Tensor

        """
        image_shuffled = DownSamplingShuffle(image)
        output_image = self.ted(image_shuffled.float())

        upsampler = SimpleUpsampler(2)
        upsampler = nn.Sequential(*upsampler)
        output_image = upsampler(output_image)

        return self.final_conv(self.refpad(output_image))
