# -*- coding: utf-8 -*-
'''
This is a PyTorch implementation of CURL: Neural Curve Layers for Global Image Enhancement
https://arxiv.org/pdf/1911.13175.pdf

Please cite paper if you use this code.

Tested with Pytorch 0.3.1, Python 3.5

Authors: Sean Moran (sean.j.moran@gmail.com), 2020

Instructions:

To get this code working on your system / problem you will need to edit the
data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data
directory (anything with "/aiml/data")

2. main.py, requires images_train.txt, images_valid.txt, images_test.txt, 
that list the training, validation and test images, one per line of each
txt file

3. data.py, lines 217, 223, change the folder names of the data input and
output directories to point to your folder names. 

We used the Samsung S7 and the Adobe datasets in the paper. They can be 
found at the following URLs:

1. Samsung S7: https://elischwartz.github.io/DeepISP/
2. Adobe5k: https://data.csail.mit.edu/graphics/fivek/

To train the model:

python main.py --valid_every=250 --num_epoch=10000 

With the above arguments, the model will be tested on the validation dataset
every 250 epochs, and the total number of epochs for training will be 10,000.

'''

from skimage.transform import resize
import cv2
import imageio
from abc import ABCMeta, abstractmethod
from data import SamsungDataLoader, Dataset
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
from skimage.measure import compare_ssim as ssim
import glob
import os
import matplotlib
import metric
import model
np.set_printoptions(threshold=np.nan)

def main():

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "/aiml/data/log_" + timestamp
    os.mkdir(log_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/curl.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epoches (default 5000)", default=100000)
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epoches after which to compute validation accuracy",
        default=500)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved CURL model instance", default=None)

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

    training_data_loader = SamsungDataLoader(data_dirpath="/aiml/data/",
                                             img_ids_filepath="/aiml/data/images_train.txt")
    training_data_dict = training_data_loader.load_data()
    training_dataset = Dataset(data_dict=training_data_dict, transform=transforms.Compose(
        [transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
         transforms.ToTensor()]),
        normaliser=2 ** 8 - 1, is_valid=False)

    validation_data_loader = SamsungDataLoader(data_dirpath="/aiml/data/",
                                               img_ids_filepath="/aiml/data/images_valid.txt")
    validation_data_dict = validation_data_loader.load_data()
    validation_dataset = Dataset(data_dict=validation_data_dict,
                                 transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                                 is_valid=True)
    validation_crop_dataset = Dataset(data_dict=validation_data_dict,
                                      transform=transforms.Compose([transforms.ToTensor()]), normaliser=2 ** 8 - 1,
                                      is_valid=False)

    testing_data_loader = SamsungDataLoader(data_dirpath="/aiml/data/",
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
                                                         shuffle=False, num_workers=4)
    validation_crop_data_loader = torch.utils.data.DataLoader(validation_crop_dataset, batch_size=1,
                                                              shuffle=False,
                                                              num_workers=4)

    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        inference_data_loader = SamsungDataLoader(data_dirpath=inference_img_dirpath,
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

        criterion = model.CURLLoss()

        testing_evaluator = metric.Evaluator(
            criterion, inference_data_loader, "test", log_dirpath)

        testing_evaluator.evaluate(net, epoch=0)

    else:
        
        net = model.CURLNet()

        logging.info('######### Network created #########')
        logging.info('Architecture:\n' + str(net))

        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)

        criterion = model.CURLLoss(ssim_window_size=5)

        '''
        The following objects allow for evaluation of a model on the testing and validation splits of a dataset
        '''
        validation_evaluator = metric.Evaluator(
            criterion, validation_data_loader, "valid", log_dirpath)
        testing_evaluator = metric.Evaluator(
            criterion, testing_data_loader, "test", log_dirpath)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      net.parameters()), lr=5e-5, betas=(0.9, 0.999), eps=1e-08)

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

            # train loss
            examples = 0.0
            running_loss = 0.0
            
            for batch_num, data in enumerate(training_data_loader, 0):

                input_img_batch, output_img_batch, category = Variable(data['input_img'],
                                                                       requires_grad=False).cuda(), Variable(data['output_img'],
                                                                                                             requires_grad=False).cuda(), data[
                    'name']

                start_time = time.time()
                net_output_img_batch, gradient_regulariser = net(
                    input_img_batch.unsqueeze(0))
                net_output_img_batch = torch.clamp(
                    net_output_img_batch, 0.0, 1.0)

                elapsed_time = time.time() - start_time

                loss = criterion(net_output_img_batch,
                                 output_img_batch, gradient_regulariser)

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

            for batch_num, data in enumerate(validation_crop_data_loader, 0):

                net.eval()

                input_img_batch, output_img_batch, category = Variable(
                    data['input_img'],
                    requires_grad=True).cuda(), Variable(data['output_img'],
                                                         requires_grad=False).cuda(), \
                    data[
                    'name']

                net_output_img_batch, gradient_regulariser = net(
                    input_img_batch.unsqueeze(0))
                net_output_img_batch = torch.clamp(
                    net_output_img_batch, 0.0, 1.0)

                optimizer.zero_grad()

                loss = criterion(net_output_img_batch,
                                 output_img_batch, gradient_regulariser)

                running_loss += loss.data[0]
                examples += batch_size

            logging.info('[%d] valid loss: %.15f' %
                         (epoch + 1, running_loss / examples))


            net.train()

            if (epoch + 1) % valid_every == 0:

                logging.info("Evaluating model on validation dataset")

                valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                    net, epoch)
                test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                    net, epoch)

                # update best validation set psnr
                if valid_psnr > best_valid_psnr:

                    logging.info(
                        "Validation PSNR has increased. Saving the more accurate model to file: " + 'curl_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(valid_psnr,
                                                                                                                                                                                         valid_loss.tolist()[0], test_psnr, test_loss.tolist()[
                                                                                                                                                                                             0],
                                                                                                                                                                                         epoch))

                    best_valid_psnr = valid_psnr
                    snapshot_prefix = os.path.join(
                        log_dirpath, 'curl')
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
        testing_evaluator.evaluate(net, epoch)

        snapshot_prefix = os.path.join(log_dirpath, 'curl')
        snapshot_path = snapshot_prefix + "_" + str(num_epoch)
        torch.save(net.state_dict(), snapshot_path)


if __name__ == "__main__":
    main()
