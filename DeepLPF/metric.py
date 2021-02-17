# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import matplotlib
matplotlib.use('agg')
from skimage.transform import resize
import cv2
import imageio
from abc import ABCMeta, abstractmethod
from data import Adobe5kDataLoader, Dataset
import unet
import skimage
import random
import time
import torch
import torch.nn as nn
import traceback
import torchvision.transforms as transforms
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import data
import logging
from PIL import Image
from shutil import copyfile
import argparse
import shutil
import torch.optim as optim
import copy
import numpy as np
import math
from util import ImageProcessing
import datetime
import torch.nn.init as net_init
from scipy.ndimage.filters import convolve
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage import io, color
from math import exp
import torch.nn.functional as F
import os.path
from skimage.metrics import structural_similarity as ssim
import glob
import os
import model
import sys
np.set_printoptions(threshold=sys.maxsize)


class Evaluator():

    def __init__(self, criterion, data_loader, split_name, log_dirpath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        super().__init__()
        self.criterion = criterion
        self.data_loader = data_loader
        self.split_name = split_name
        self.log_dirpath = log_dirpath

    def evaluate(self, net, epoch=0):
        """Evaluates a network on a specified split of a dataset e.g. test, validation

        :param net: PyTorch neural network data structure
        :param data_loader: an instance of the DataLoader class for the dataset of interest
        :param split_name: name of the split e.g. "test", "validation"
        :param log_dirpath: logging directory
        :returns: average loss, average PSNR
        :rtype: float, float

        """
        
        psnr_avg = 0.0
        ssim_avg = 0.0
        examples = 0
        running_loss = 0
        num_batches = 0
        batch_size = 1

        out_dirpath = self.log_dirpath + "/" + self.split_name.lower()
        if not os.path.isdir(out_dirpath):
            os.mkdir(out_dirpath)

        # switch model to evaluation mode
        net.eval()
        net.cuda()

        with torch.no_grad():
            for batch_num, data in enumerate(self.data_loader, 0):

                input_img_batch, output_img_batch, name = Variable(data['input_img'], requires_grad=False).cuda(), Variable(data['output_img'],
                                                                                                   requires_grad=False).cuda(), \
                    data['name']
                input_img_batch = input_img_batch.unsqueeze(0)

                for i in range(0, input_img_batch.shape[0]):

                    img = input_img_batch[i, :, :, :]
                    img = torch.clamp(img, 0, 1)

                    net_output_img_example = net(img)

                    if net_output_img_example.shape[2]!=output_img_batch.shape[2]:
                        net_output_img_example=net_output_img_example.transpose(2,3)

                    loss = self.criterion(net_output_img_example[:, 0:3, :, :],
                                          output_img_batch[:, 0:3, :, :])

                    input_img_example = (input_img_batch.cpu(
                    ).data[0, 0:3, :, :].numpy() * 255).astype('uint8')

                    output_img_batch_numpy = output_img_batch.squeeze(
                        0).data.cpu().numpy()
                    output_img_batch_numpy = ImageProcessing.swapimdims_3HW_HW3(
                        output_img_batch_numpy)
                    output_img_batch_rgb = output_img_batch_numpy
                    output_img_batch_rgb = ImageProcessing.swapimdims_HW3_3HW(
                        output_img_batch_rgb)
                    output_img_batch_rgb = np.expand_dims(
                        output_img_batch_rgb, axis=0)

                    net_output_img_example_numpy = net_output_img_example.squeeze(
                        0).data.cpu().numpy()
                    net_output_img_example_numpy = ImageProcessing.swapimdims_3HW_HW3(
                        net_output_img_example_numpy)
                    net_output_img_example_rgb = net_output_img_example_numpy
                    net_output_img_example_rgb = ImageProcessing.swapimdims_HW3_3HW(
                        net_output_img_example_rgb)
                    net_output_img_example_rgb = np.expand_dims(
                        net_output_img_example_rgb, axis=0)
                    net_output_img_example_rgb = np.clip(
                        net_output_img_example_rgb, 0, 1)

                    running_loss += loss.data[0]
                    examples += batch_size
                    num_batches += 1

                    psnr_example = ImageProcessing.compute_psnr(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32), 1.0)
                    ssim_example = ImageProcessing.compute_ssim(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32))

                    psnr_avg += psnr_example
                    ssim_avg += ssim_example
                    
                    if batch_num > 30:
                        '''
                        We save only the first 30 images down for time saving
                        purposes
                        '''
                        continue
                    else:

                        output_img_example = (
                            output_img_batch_rgb[0, 0:3, :, :] * 255).astype('uint8')
                        net_output_img_example = (
                            net_output_img_example_rgb[0, 0:3, :, :] * 255).astype('uint8')

                        plt.imsave(out_dirpath + "/" + name[0].split(".")[0] + "_" + self.split_name.upper() + "_" + str(epoch + 1) + "_" + str(
                            examples) + "_PSNR_" + str("{0:.3f}".format(psnr_example)) + "_SSIM_" + str(
                            "{0:.3f}".format(ssim_example)) + ".jpg",
                            ImageProcessing.swapimdims_3HW_HW3(net_output_img_example))

                    del net_output_img_example_numpy
                    del net_output_img_example_rgb
                    del output_img_batch_rgb
                    del output_img_batch_numpy
                    del input_img_example
                    del output_img_batch

        psnr_avg = psnr_avg / num_batches
        ssim_avg = ssim_avg / num_batches

        logging.info('loss_%s: %.5f psnr_%s: %.3f ssim_%s: %.3f' % (
            self.split_name, (running_loss / examples), self.split_name, psnr_avg, self.split_name, ssim_avg))

        loss = (running_loss / examples)

        return loss, psnr_avg, ssim_avg
