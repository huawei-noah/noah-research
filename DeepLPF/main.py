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

Instructions:

To get this code working on your system / problem please see the README.

*** BATCH SIZE: Note this code is designed for a batch size of 1. The code needs re-engineered to support higher batch sizes. Using higher batch sizes is not supported currently and could lead to artefacts. To replicate our reported results 
please use a batch size of 1 only ***
'''
import model
import metric
import os
import os.path
import datetime
import torch.optim as optim
import argparse
import logging
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch
import time
from data import Adobe5kDataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
matplotlib.use('agg')

def main():

    print("*** Before running this code ensure you keep the default batch size of 1. The code has not been engineered to support higher batch sizes. See README for more detail. Remove the exit() statement to use code. ***")
    exit()

    writer = SummaryWriter()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
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
        default=25)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance", default=None)
    parser.add_argument(
        "--training_img_dirpath", required=True,
        help="Directory containing images to train a DeepLPF model instance", default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument(
        "--inference_img_list_path", required=False,
        help="Plain text file containing the names of the images to inference")
    parser.add_argument(
        "--train_img_list_path", required=True,
        help="Plain text file containing the names of the training images")
    parser.add_argument(
        "--valid_img_list_path", required=True,
        help="Plain text file containing the names of the validation images")
    parser.add_argument(
        "--test_img_list_path", required=False,
        help="Plain text file containing the names of the test images")

    args = parser.parse_args()
    num_epoch = args.num_epoch
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath
    training_img_dirpath = args.training_img_dirpath
    inference_img_list_path = args.inference_img_list_path
    test_img_list_path = args.test_img_list_path
    valid_img_list_path = args.valid_img_list_path
    train_img_list_path = args.train_img_list_path

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(num_epoch))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(valid_every))
    logging.info('Training image directory: ' + str(training_img_dirpath))
    logging.info('List of images to inference: ' + str(inference_img_list_path))
    logging.info('List of test images: ' + str(test_img_list_path))
    logging.info('List of validation images: ' + str(valid_img_list_path))
    logging.info('List of training images: ' + str(train_img_list_path))

    logging.info('##############################')

    

    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        '''
        inference_img_dirpath: the actual filepath should have "input" in the name an in the level above where the images 
        for inference are located, there should be a file "images_inference.txt with each image filename as one line i.e."
        
        images_inference.txt    ../
                                a1000.tif
                                a1242.tif
                                etc
        '''
        inference_data_loader = Adobe5kDataLoader(data_dirpath=inference_img_dirpath,
                                                  img_ids_filepath=inference_img_list_path)
        inference_data_dict = inference_data_loader.load_data()
        inference_dataset = Dataset(data_dict=inference_data_dict,
                                    transform=transforms.Compose([transforms.ToTensor()]), normaliser=1,
                                    is_inference=True)

        inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                                            num_workers=6)

        '''
        Performs inference on all the images in inference_img_dirpath
        '''
        logging.info(
            "Performing inference with images in directory: " + inference_img_dirpath)

        net = model.DeepLPFNet()
        net.load_state_dict(torch.load(checkpoint_filepath))
        net.eval()

        criterion = model.DeepLPFLoss()

        inference_evaluator = metric.Evaluator(
            criterion, inference_data_loader, "test", log_dirpath)

        inference_evaluator.evaluate(net, epoch=0)

    else:

        training_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                                 img_ids_filepath=train_img_list_path)
        training_data_dict = training_data_loader.load_data()

        training_dataset = Dataset(data_dict=training_data_dict, normaliser=1, is_valid=False)

        validation_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                               img_ids_filepath=valid_img_list_path)
        validation_data_dict = validation_data_loader.load_data()
        validation_dataset = Dataset(data_dict=validation_data_dict, normaliser=1, is_valid=True)

        testing_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                            img_ids_filepath=test_img_list_path)
        testing_data_dict = testing_data_loader.load_data()
        testing_dataset = Dataset(data_dict=testing_data_dict, normaliser=1,is_valid=True)

        training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=1, shuffle=True,
                                                       num_workers=6)
        testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                      num_workers=6)
        validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                         shuffle=False,
                                                         num_workers=6)
        net = model.DeepLPFNet()
        net.cuda(0)

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

        optimizer.zero_grad()
        net.train()

        running_loss = 0.0
        examples = 0
        batch_size = 1       # *** WARNING: batch size of > 1 not supported in current version of code ***
        total_examples = 0

        for epoch in range(num_epoch):

            # Train loss
            examples = 0.0
            running_loss = 0.0
            
            for batch_num, data in enumerate(training_data_loader, 0):

                input_img_batch, gt_img_batch, _ = Variable(data['input_img'],
                                                                       requires_grad=False).cuda(), Variable(data['output_img'],
                                                                                                             requires_grad=False).cuda(), data[
                    'name']

                start_time = time.time()
                net_img_batch = net(input_img_batch)
                net_img_batch = torch.clamp(net_img_batch, 0.0, 1.0)

                elapsed_time = time.time() - start_time

                loss = criterion(net_img_batch, gt_img_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.data[0]
                examples += batch_size
                total_examples+=batch_size

                writer.add_scalar('Loss/train', loss.data[0], total_examples)

            logging.info('[%d] train loss: %.15f' %
                         (epoch + 1, running_loss / examples))
            writer.add_scalar('Loss/train_smooth', running_loss / examples, epoch + 1)

            # Valid loss
            '''
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
                total_examples+=batch_size

                writer.add_scalar('Loss/train', loss.data[0], total_examples)

            logging.info('[%d] valid loss: %.15f' %
                         (epoch + 1, running_loss / examples))
            writer.add_scalar('Loss/valid_smooth', running_loss / examples, epoch + 1)

            net.train()
            '''

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
                    torch.save(net.state_dict(), snapshot_path)

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
