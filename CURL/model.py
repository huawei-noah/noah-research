# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""The CURL network: a TED backbone followed by the neural curve layer.

The loss lives in :mod:`losses` and the convolutional blocks in :mod:`blocks`.
``CURLLoss`` is re-exported here so that ``model.CURLLoss`` keeps working.
"""
import torch
import torch.nn as nn

import rgb_ted
from blocks import Block, ConvBlock, GlobalPoolingBlock, MaxPoolBlock
from colour import hsv_to_rgb, lab_to_rgb, rgb_to_hsv, rgb_to_lab
from curves import adjust_hsv, adjust_lab, adjust_rgb
from losses import CURLLoss

__all__ = ['CURLNet', 'CURLLayer', 'CURLLoss', 'Block', 'ConvBlock',
           'MaxPoolBlock', 'GlobalPoolingBlock']


class CURLLayer(nn.Module):

    def __init__(self, num_in_channels=64, num_out_channels=64):
        """Initialisation of class

        :param num_in_channels: number of input channels
        :param num_out_channels: number of output channels
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        """ Initialise the CURL block layers

        :returns: N/A
        :rtype: N/A

        """
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock()

        self.fc_lab = torch.nn.Linear(64, 48)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock()

        self.fc_rgb = torch.nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock()

        self.fc_hsv = torch.nn.Linear(64, 64)

    def forward(self, x):
        """Forward function for the CURL layer

        :param x: forward the data x through the network 
        :returns: Tensor representing the predicted image
        :rtype: Tensor

        """

        '''
        This function is where the magic happens :)
        '''
        x.contiguous()  # remove memory holes

        feat = x[:, 3:64, :, :]
        img = x[:, 0:3, :, :]

        img_clamped = torch.clamp(img, 0, 1)
        img_lab = torch.clamp(rgb_to_lab(img_clamped), 0, 1)

        feat_lab = torch.cat((feat, img_lab), 1)

        x = self.lab_layer1(feat_lab)
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout1(x)
        L = self.fc_lab(x)

        img_lab, gradient_regulariser_lab = adjust_lab(
            img_lab, L[:, 0:48])
        img_rgb = lab_to_rgb(img_lab)
        img_rgb = torch.clamp(img_rgb, 0, 1)

        feat_rgb = torch.cat((feat, img_rgb), 1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout2(x)
        R = self.fc_rgb(x)

        img_rgb, gradient_regulariser_rgb = adjust_rgb(
            img_rgb, R[:, 0:48])
        img_rgb = torch.clamp(img_rgb, 0, 1)

        img_hsv = rgb_to_hsv(img_rgb)
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv), 1)

        x = self.hsv_layer1(feat_hsv)
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout3(x)
        H = self.fc_hsv(x)

        img_hsv, gradient_regulariser_hsv = adjust_hsv(
            img_hsv, H[:, 0:64])
        img_hsv = torch.clamp(img_hsv, 0, 1)

        img_residual = torch.clamp(hsv_to_rgb(img_hsv), 0, 1)

        img = torch.clamp(img + img_residual, 0, 1)

        gradient_regulariser = gradient_regulariser_rgb + \
            gradient_regulariser_lab+gradient_regulariser_hsv

        return img, gradient_regulariser


class CURLNet(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = rgb_ted.TEDModel()
        self.curllayer = CURLLayer()

    def forward(self, img):
        """Neural network forward function

        :param img: forward the data img through the network
        :returns: residual image
        :rtype: numpy ndarray

        """
        feat = self.tednet(img)
        img, gradient_regulariser = self.curllayer(feat)
        return img, gradient_regulariser
