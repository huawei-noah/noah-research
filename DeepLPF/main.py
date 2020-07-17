# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory (anything with "/aiml/data")

2. data.py, lines 216, 224, change the folder names of the data input and
output directories to point to your folder names
'''
import model
import metric
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
import unet
from data import Adobe5kDataLoader, Dataset
from abc import ABCMeta, abstractmethod
import imageio
import cv2
from skimage.transform import resize
import matplotlib
matplotlib.use('agg')
np.set_printoptions(threshold=np.nan)


def main():

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "/aiml/data/log_" + timestamp
    os.mkdir(log_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=500)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance", default=None)

    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(num_epoch))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(valid_every))
    logging.info('##############################')

    training_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                             img_ids_filepath="/aiml/data/images_train.txt")
    training_data_dict = training_data_loader.load_data()
    training_dataset = Dataset(data_dict=training_data_dict, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)

    validation_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                               img_ids_filepath="/aiml/data/images_valid.txt")
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict,
                                 transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                                 is_valid=True)

    testing_data_loader = Adobe5kDataLoader(data_dirpath="/aiml/data/",
                                            img_ids_filepath="/aiml/data/images_test.txt")
    testing_data_dict = testing_data_loader.load_data()
    testing_dataset = Dataset(data_dict=testing_data_dict,
                              transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                              is_valid=True)

    training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                                       num_workers=4)
    testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                      num_workers=4)
    validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                         shuffle=False,
                                                         num_workers=4)

    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        inference_data_loader = Adobe5kDataLoader(data_dirpath=inference_img_dirpath,
                                                  img_ids_filepath=inference_img_dirpath+"/images_inference.txt")
        inference_data_dict = inference_data_loader.load_data()
        inference_dataset = Dataset(data_dict=inference_data_dict,
                                    transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                                    is_valid=True)

        inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                                            num_workers=4)

        '''
        Performs inference on all the images in inference_img_dirpath
        '''
        logging.info(
            "Performing inference with images in directory: " + inference_img_dirpath)

        net = torch.load(checkpoint_filepath,
                         map_location=lambda storage, location: storage)

        # switch model to evaluation mode
        net.eval()

        criterion = model.DeepLPFLoss()

        testing_evaluator = metric.Evaluator(
            criterion, inference_data_loader, "test", log_dirpath)

        testing_evaluator.evaluate(net, epoch=0)

    else:

        net = model.DeepLPFNet()

        logging.info('######### Network created #########')
        logging.info('Architecture:\n' + str(net))

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)

        criterion = model.DeepLPFLoss(ssim_window_size=5)

        '''
        The following objects allow for evaluation of a model on the testing and validation splits of a dataset
        '''
        validation_evaluator = metric.Evaluator(
            criterion, validation_data_loader, "valid", log_dirpath)
        testing_evaluator = metric.Evaluator(
            criterion, testing_data_loader, "test", log_dirpath)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999),
                               eps=1e-08)
        best_valid_psnr = 0.0

        alpha = 0.0
        optimizer.zero_grad()
        net.train()

        running_loss = 0.0
        examples = 0
        psnr_avg = 0.0
        ssim_avg = 0.0
        batch_size = 1
        net.cuda()

        for epoch in range(num_epoch):

            # Train loss
            examples = 0.0
            running_loss = 0.0
            
            for batch_num, data in enumerate(training_data_loader, 0):

                input_img_batch, output_img_batch, category = Variable(data['input_img'],
                                                                       requires_grad=False).cuda(), Variable(data['output_img'],
                                                                                                             requires_grad=False).cuda(), data[
                    'name']

                start_time = time.time()
                net_output_img_batch = net(
                    input_img_batch)
                net_output_img_batch = torch.clamp(
                    net_output_img_batch, 0.0, 1.0)

                elapsed_time = time.time() - start_time

                loss = criterion(net_output_img_batch, output_img_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                examples += batch_size
            
            logging.info('[%d] train loss: %.15f' %
                         (epoch + 1, running_loss / examples))

            # Valid loss
            examples = 0.0
            running_loss = 0.0

            for batch_num, data in enumerate(validation_data_loader, 0):

                net.eval()

                input_img_batch, output_img_batch, category = Variable(
                    data['input_img'],
                    requires_grad=False).cuda(), Variable(data['output_img'],
                                                         requires_grad=False).cuda(), \
                    data[
                    'name']

                net_output_img_batch = net(
                    input_img_batch)
                net_output_img_batch = torch.clamp(
                    net_output_img_batch, 0.0, 1.0)

                optimizer.zero_grad()

                loss = criterion(net_output_img_batch, output_img_batch)

                running_loss += loss.data[0]
                examples += batch_size

            logging.info('[%d] valid loss: %.15f' %
                         (epoch + 1, running_loss / examples))

            net.train()

            if (epoch + 1) % valid_every == 0:

                logging.info("Evaluating model on validation and test dataset")

                valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                    net, epoch)
                test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                    net, epoch)

                # update best validation set psnr
                if valid_psnr > best_valid_psnr:

                    logging.info(
                        "Validation PSNR has increased. Saving the more accurate model to file: " + 'deeplpf_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                                                                         valid_loss.tolist()[0], test_psnr, test_loss.tolist()[
                                                                                                                                                                                             0],
                                                                                                                                                                                         epoch))

                    best_valid_psnr = valid_psnr
                    snapshot_prefix = os.path.join(
                        log_dirpath, 'deeplpf')
                    snapshot_path = snapshot_prefix + '_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                    valid_loss.tolist()[
                                                                                                                                        0],
                                                                                                                                    test_psnr, test_loss.tolist()[
                                                                                                                                        0],
                                                                                                                                    epoch)
                    torch.save(net, snapshot_path)

                net.train()

        '''
        Run the network over the testing dataset split
        '''
        testing_evaluator.evaluate(net, epoch=0)

        snapshot_prefix = os.path.join(log_dirpath, 'deep_lpf')
        snapshot_path = snapshot_prefix + "_" + str(num_epoch)
        torch.save(net.state_dict(), snapshot_path)


if __name__ == "__main__":
    main()
