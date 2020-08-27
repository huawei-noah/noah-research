# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
import matplotlib
matplotlib.use('agg')
import os
import glob
from skimage.measure import compare_ssim as ssim
import os.path
import torch.nn.functional as F
from math import exp
from skimage import io, color
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from scipy.ndimage.filters import convolve
import torch.nn.init as net_init
import datetime
from util import ImageProcessing
import math
import numpy as np
import copy
import torch.optim as optim
import shutil
import argparse
from shutil import copyfile
from PIL import Image
import logging
import data
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import torchvision.transforms as transforms
import traceback
import torch.nn as nn
import torch
import time
import random
import skimage
import ted
from data import SamsungDataLoader, Dataset
from abc import ABCMeta, abstractmethod
import imageio
import cv2
from skimage.transform import resize
print(torch.__version__)
np.set_printoptions(threshold=np.nan)


class CURLLoss(nn.Module):

    def __init__(self, ssim_window_size=5, alpha=0.5):
        """Initialisation of the CURL loss function

        :param ssim_window_size: size of averaging window for SSIM
        :param alpha: interpolation paramater for L1 and SSIM parts of the loss
        :returns: N/A
        :rtype: N/A

        """
        super(CURLLoss, self).__init__()
        self.alpha = alpha
        self.ssim_window_size = ssim_window_size

    def create_window(self, window_size, num_channel):
        """Window creation function for SSIM metric. Gaussian weights are applied to the window.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param window_size: size of the window to compute statistics
        :param num_channel: number of channels
        :returns: Tensor of shape Cx1xWindow_sizexWindow_size
        :rtype: Tensor

        """
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(
            _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(
            num_channel, 1, window_size, window_size).contiguous())
        return window

    def gaussian(self, window_size, sigma):
        """
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
        :param window_size: size of the SSIM sampling window e.g. 11
        :param sigma: Gaussian variance
        :returns: 1xWindow_size Tensor of Gaussian weights
        :rtype: Tensor

        """
        gauss = torch.Tensor(
            [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()

    def compute_ssim(self, img1, img2):
        """Computes the structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        (_, num_channel, _, _) = img1.size()
        window = self.create_window(self.ssim_window_size, num_channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())
            window = window.type_as(img1)

        mu1 = F.conv2d(
            img1, window, padding=self.ssim_window_size // 2, groups=num_channel)
        mu2 = F.conv2d(
            img2, window, padding=self.ssim_window_size // 2, groups=num_channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(
            img1 * img1, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_sq
        sigma2_sq = F.conv2d(
            img2 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu2_sq
        sigma12 = F.conv2d(
            img1 * img2, window, padding=self.ssim_window_size // 2, groups=num_channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map1 = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
        ssim_map2 = ((mu1_sq.cuda() + mu2_sq.cuda() + C1) *
                     (sigma1_sq.cuda() + sigma2_sq.cuda() + C2))
        ssim_map = ssim_map1.cuda() / ssim_map2.cuda()

        v1 = 2.0 * sigma12.cuda() + C2
        v2 = sigma1_sq.cuda() + sigma2_sq.cuda() + C2
        cs = torch.mean(v1 / v2)

        return ssim_map.mean(), cs

    def compute_msssim(self, img1, img2):
        """Computes the multi scale structural similarity index between two images. This function is differentiable.
        Code adapted from: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

        :param img1: image Tensor BxCxHxW
        :param img2: image Tensor BxCxHxW
        :returns: mean SSIM
        :rtype: float

        """
        if img1.size() != img2.size():
            raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (
                img1.size(), img2.size()))
        if len(img1.size()) != 4:
            raise RuntimeError(
                'Input images must have four dimensions, not %d' % len(img1.size()))

        if type(img1) is not Variable or type(img2) is not Variable:
            raise RuntimeError(
                'Input images must be Variables, not %s' % img1.__class__.__name__)

        weights = Variable(torch.FloatTensor(
            [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]))
        # weights = Variable(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))
        if img1.is_cuda:
            weights = weights.cuda(img1.get_device())

        levels = weights.size()[0]
        mssim = []
        mcs = []
        for _ in range(levels):
            sim, cs = self.compute_ssim(img1, img2)
            mssim.append(sim)
            mcs.append(cs)

            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        mssim = torch.cat(mssim)
        mcs = torch.cat(mcs)

        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

        prod = (torch.prod(mcs[0:levels - 1] ** weights[0:levels - 1])
                * (mssim[levels - 1] ** weights[levels - 1]))
        return prod

    def forward(self, predicted_img_batch, target_img_batch, gradient_regulariser):
        """Forward function for the CURL loss

        :param predicted_img_batch_high_res: 
        :param predicted_img_batch_high_res_rgb: 
        :param target_img_batch: Tensor of shape BxCxWxH
        :returns: value of loss function
        :rtype: float

        """
        num_images = target_img_batch.shape[0]
        target_img_batch = target_img_batch

        ssim_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        l1_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        cosine_rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        sat_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        hue_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        value_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        a_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        b_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        hsv_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        deep_isp_loss = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))
        rgb_loss_value = Variable(
            torch.cuda.FloatTensor(torch.zeros(1, 1).cuda()))

        for i in range(0, num_images):

            target_img = target_img_batch[i, :, :, :].permute(2, 0, 1).cuda()
            predicted_img = predicted_img_batch[i, :, :, :].cuda()

            predicted_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(predicted_img.squeeze(0)), 0, 1)
            target_img_lab = torch.clamp(
                ImageProcessing.rgb_to_lab(target_img.squeeze(0)), 0, 1)

            target_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                target_img.squeeze(0)), 0, 1)
            predicted_img_hsv = torch.clamp(ImageProcessing.rgb_to_hsv(
                predicted_img.squeeze(0)), 0, 1)

            predicted_img_hue = (predicted_img_hsv[0, :, :]*2*math.pi)
            predicted_img_val = predicted_img_hsv[2, :, :]
            predicted_img_sat = predicted_img_hsv[1, :, :]
            target_img_hue = (target_img_hsv[0, :, :]*2*math.pi)
            target_img_val = target_img_hsv[2, :, :]
            target_img_sat = target_img_hsv[1, :, :]

            target_img_L_ssim = target_img_lab[0, :, :].unsqueeze(0)
            predicted_img_L_ssim = predicted_img_lab[0, :, :].unsqueeze(0)
            target_img_L_ssim = target_img_L_ssim.unsqueeze(0)
            predicted_img_L_ssim = predicted_img_L_ssim.unsqueeze(0)

            ssim_value = self.compute_msssim(
                predicted_img_L_ssim, target_img_L_ssim)

            ssim_loss_value += (1.0 - ssim_value)
            predicted_img_1 = predicted_img_val * \
                predicted_img_sat*torch.cos(predicted_img_hue)
            predicted_img_2 = predicted_img_val * \
                predicted_img_sat*torch.sin(predicted_img_hue)
            target_img_1 = target_img_val * \
                target_img_sat*torch.cos(target_img_hue)
            target_img_2 = target_img_val * \
                target_img_sat*torch.sin(target_img_hue)

            p = torch.stack(
                (predicted_img_1, predicted_img_2, predicted_img_val), 2)
            d = torch.stack((target_img_1, target_img_2, target_img_val), 2)

            l1_loss_value += F.l1_loss(predicted_img_lab, target_img_lab)
            rgb_loss_value += F.l1_loss(predicted_img, target_img)
            hsv_loss_value += F.l1_loss(p, d)

            cosine_rgb_loss_value += (1-torch.mean(
                torch.nn.functional.cosine_similarity(predicted_img, target_img, dim=0)))

        l1_loss_value = l1_loss_value/num_images
        rgb_loss_value_hsv = rgb_loss_value/num_images
        ssim_loss_value = ssim_loss_value/num_images
        cosine_rgb_loss_value = cosine_rgb_loss_value/num_images
        hsv_loss_value = hsv_loss_value/num_images

        '''
        Note the hyperparameters 1e-3, 1e-6 below work well for SamsungS7. They
        may need changed for other datasets.
        '''
        curl_loss = (rgb_loss_value + cosine_rgb_loss_value + l1_loss_value +
                     hsv_loss_value + 1e-3*ssim_loss_value + 1e-6*gradient_regulariser)/6

        return curl_loss


class CURLLayer(nn.Module):

    import torch.nn.functional as F

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
        self.lab_layer8 = GlobalPoolingBlock(2)

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
        self.rgb_layer8 = GlobalPoolingBlock(2)

        self.fc_rgb = torch.nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock(2)

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

        torch.cuda.empty_cache()
        shape = x.shape

        img_clamped = torch.clamp(img, 0, 1)
        img_lab = torch.clamp(ImageProcessing.rgb_to_lab(
            img_clamped.squeeze(0)), 0, 1)

        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), 1)

        x = self.lab_layer1(feat_lab)
        del feat_lab
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

        img_lab, gradient_regulariser_lab = ImageProcessing.adjust_lab(
            img_lab.squeeze(0), L[0, 0:48])
        img_rgb = ImageProcessing.lab_to_rgb(img_lab.squeeze(0))
        img_rgb = torch.clamp(img_rgb, 0, 1)

        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), 1)

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

        img_rgb, gradient_regulariser_rgb = ImageProcessing.adjust_rgb(
            img_rgb.squeeze(0), R[0, 0:48])
        img_rgb = torch.clamp(img_rgb, 0, 1)

        img_hsv = ImageProcessing.rgb_to_hsv(img_rgb.squeeze(0))
        img_hsv = torch.clamp(img_hsv, 0, 1)
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), 1)

        x = self.hsv_layer1(feat_hsv)
        del feat_hsv
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

        img_hsv, gradient_regulariser_hsv = ImageProcessing.adjust_hsv(
            img_hsv, H[0, 0:64])
        img_hsv = torch.clamp(img_hsv, 0, 1)

        img_residual = torch.clamp(ImageProcessing.hsv_to_rgb(
           img_hsv.squeeze(0)), 0, 1)

        img = torch.clamp(img + img_residual.unsqueeze(0), 0, 1)

        gradient_regulariser = gradient_regulariser_rgb + \
            gradient_regulariser_lab+gradient_regulariser_hsv

        return img, gradient_regulariser


class Block(nn.Module):

    def __init__(self):
        """Initialisation for a lower-level DeepLPF conv block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

    def conv3x3(self, in_channels, out_channels, stride=1):
        """Represents a convolution of shape 3x3

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param stride: the convolution stride
        :returns: convolution function with the specified parameterisation
        :rtype: function

        """
        return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                         stride=stride, padding=1, bias=True)


class ConvBlock(Block, nn.Module):

    def __init__(self, num_in_channels, num_out_channels, stride=1):
        """Initialise function for the higher level convolution block

        :param in_channels:
        :param out_channels:
        :param stride:
        :param padding:
        :returns:
        :rtype:

        """
        super(Block, self).__init__()
        self.conv = self.conv3x3(num_in_channels, num_out_channels, stride=2)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        """ Forward function for the higher level convolution block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(Block, nn.Module):

    def __init__(self):
        """Initialise function for the max pooling block

        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """ Forward function for the max pooling block

        :param x: Tensor representing the input BxCxWxH, where B is the batch size, C is the number of channels, W and H are the width and image height
        :returns: Tensor representing the output of the block
        :rtype: Tensor

        """
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(Block, nn.Module):

    def __init__(self, receptive_field):
        """Implementation of the global pooling block. Takes the average over a 2D receptive field.
        :param receptive_field:
        :returns: N/A
        :rtype: N/A

        """
        super(Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """Forward function for the high-level global pooling block

        :param x: Tensor of shape BxCxAxA
        :returns: Tensor of shape BxCx1x1, where B is the batch size
        :rtype: Tensor

        """
        out = self.avg_pool(x)
        return out


class CURLNet(nn.Module):

    def __init__(self):
        """Initialisation function

        :returns: initialises parameters of the neural networ
        :rtype: N/A

        """
        super(CURLNet, self).__init__()
        self.tednet = ted.TEDModel()
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
