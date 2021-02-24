# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 0-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

'''
import os
from skimage.metrics import structural_similarity as ssim
import os.path
import torch.nn.functional as F
from skimage import io, color
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
from scipy.ndimage.filters import convolve
import torch.nn.init as net_init
import datetime
import torchvision.transforms.functional as TF
import util
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
from collections import defaultdict
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
import unet
from abc import ABCMeta, abstractmethod
import imageio
import cv2
from skimage.transform import resize
import matplotlib
import sys
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False, is_inference=False):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.is_valid = is_valid
        self.is_inference = is_inference

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.data_dict.keys()))

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """
        while True:

            if self.is_inference:

                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)
                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_pil_image(output_img)
                output_img = TF.to_tensor(output_img)

                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}

            elif idx in self.data_dict:

                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)
                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                output_img = TF.to_pil_image(output_img)

                if not self.is_valid:

                    if random.random()>0.5:

                        # Random horizontal flipping
                        if random.random() > 0.5:
                            input_img = TF.hflip(input_img)
                            output_img = TF.hflip(output_img)

                        # Random vertical flipping
                        if random.random() > 0.5:
                            input_img = TF.vflip(input_img)
                            output_img = TF.vflip(output_img)

                # Transform to tensor
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_tensor(output_img)


                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}


class DataLoader():

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        self.data_dirpath = data_dirpath
        self.img_ids_filepath = img_ids_filepath

    @abstractmethod
    def load_data(self):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass

    @abstractmethod
    def perform_inference(self, net, data_dirpath):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass


class Adobe5kDataLoader(DataLoader):

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        super().__init__(data_dirpath, img_ids_filepath)
        self.data_dict = defaultdict(dict)

    def load_data(self):
        """ Loads the Samsung image data into a Python dictionary

        :returns: Python two-level dictionary containing the images
        :rtype: Dictionary of dictionaries

        """

        logging.info("Loading Adobe5k dataset ...")

        with open(self.img_ids_filepath) as f:
            '''
            Load the image ids into a list data structure
            '''
            image_ids = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            image_ids_list = [x.rstrip() for x in image_ids]

        idx = 0
        idx_tmp = 0
        img_id_to_idx_dict = {}

        for root, dirs, files in os.walk(self.data_dirpath):

            for file in files:

                img_id = file.split("-")[0]

                is_id_in_list = False
                for img_id_test in image_ids_list:
                    if img_id_test == img_id:
                        is_id_in_list = True
                        break

                if is_id_in_list:  # check that the image is a member of the appropriate training/test/validation split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "input" in root:  # change this to the name of your
                                        # input data folder

                        input_img_filepath = file

                        self.data_dict[idx_tmp]['input_img'] = root + \
                            "/" + input_img_filepath

                    elif ("output" in root):  # change this to the name of your
                                             # output data folder

                        output_img_filepath = file

                        self.data_dict[idx_tmp]['output_img'] = root + \
                            "/" + output_img_filepath

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():
            assert ('input_img' in imgs)
            assert ('output_img' in imgs)

        return self.data_dict
