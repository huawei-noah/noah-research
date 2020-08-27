# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

'''
from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
import cv2
import imageio
from abc import ABCMeta, abstractmethod
import ted
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
from collections import defaultdict
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
import util
import datetime
import torch.nn.init as net_init
from scipy.ndimage.filters import convolve
from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from copy import deepcopy
from skimage import io, color
import torch.nn.functional as F
import os.path
from skimage.measure import compare_ssim as ssim
import os
np.set_printoptions(threshold=np.nan)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser # normaliser for groundtruth data
        self.is_valid = is_valid

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

            if idx in self.data_dict:

                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)
                input_img = np.load(self.data_dict[idx]['input_img'])

                input_img = input_img / (2**10-1)  # change this normalisation
                                        # factor for your data
                shape = input_img.shape
                input_img = np.clip(input_img, 0, 1)
                input_img[np.isnan(input_img)] = 0

                seed = random.uniform(0, 10000)

                if not self.is_valid:
                    random.seed(seed)  # make a seed with numpy generation
                    i = random.randint(0, input_img.shape[0]-512)  # patch size
                                        # of 512 pixels
                    j = random.randint(0, input_img.shape[1]-512)
                    i = i-(i % 2)  # ensure on Bayer pattern boundary
                    j = j-(j % 2)
                    input_img = input_img[i:(i+512), j:(j+512)]
                    output_img = output_img[i:(i+512), j:(j+512), :]

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


'''
This data loading class only works for the Samsung S7 dataset. You will need to
edit this class to handle a new dataset.
'''
class SamsungDataLoader(DataLoader):

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

        logging.info("Loading Samsung dataset ...")

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

                if "medium" in file:
                    img_id = file.split("-medium")[0]
                else:
                    img_id = file.split("-short")[0]

                is_id_in_list = False
                for img_id_test in image_ids_list:
                    if img_id_test == img_id:
                        is_id_in_list = True
                        break

                if is_id_in_list:   # check that the image is a member of the appropriate training/test/validation split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "medium_input" in root:  # change medium_input to match
                                        # name of your data input subdirectory

                        input_img_filepath = file

                        if file.endswith(".dng"):

                            if not os.path.isfile(root+"/"+input_img_filepath.split(".")[0]+".npy"):

                                raw_img = rawpy.imread(
                                    root+"/"+input_img_filepath)
                                np.save(root+"/"+input_img_filepath.split(".")
                                        [0]+".npy", raw_img.raw_image)

                        self.data_dict[idx_tmp]['input_img'] = root + \
                            "/"+input_img_filepath.split(".")[0]+".npy"

                    elif ("output" in root):  # change output to match name of
                                        # your data groundtruth subdirectory

                        if (file.endswith(".jpg")) and (not file.endswith(".proc.jpg")):
                            '''
                            The target images are rgb format.
                            '''
                            output_img_filepath = root + "/" + file

                            if not os.path.isfile(output_img_filepath+".proc.jpg"):

                                output_img = ImageProcessing.load_image(
                                     output_img_filepath, normaliser=2**8-1)
                                plt.imsave(output_img_filepath +
                                           ".proc.jpg", output_img)

                            self.data_dict[idx_tmp]['output_img'] = output_img_filepath+".proc.jpg"

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():

            assert('input_img' in imgs)
            assert('output_img' in imgs)

        return self.data_dict
