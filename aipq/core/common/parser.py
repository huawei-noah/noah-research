# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='IQA')
    parser.add_argument('--color', default='rgb', type=str,
                        choices=['rgb', 'y'],
                        help='color space')
    parser.add_argument('--data-dir', metavar='DIR',
                        help='path to training dataset')
    parser.add_argument('--val-data-dir', metavar='DIR', default='/hdd/data/PIPAL',
                        help='path to PIPAL training set for validation')
    parser.add_argument('--exp-dir', metavar='DIR', default='results/sandbox',
                        help='path to where to save exp')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64)')
    parser.add_argument('--inference-batch-size', default=32, type=int,
                        help='mini-batch size used for inference (default: 32)')
    parser.add_argument('--margin', default=0.2, type=float, metavar='M',
                        help='margin in pairwise ranking loss (default: 0.2)')
    parser.add_argument('--temperature', default=0.01, type=float, metavar='M',
                        help='margin in pairwise ranking loss (default: 0.01)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--factor-lr', default=10, type=float,
                        help='factor to divide to get final learning rate')
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['adam', 'sgd'],
                        help='optimizer')
    parser.add_argument('--backbone', default='vgg', type=str,
                        choices=['resnet', 'alex', 'vgg', 'squeeze'],
                        help='backbone network')
    parser.add_argument('--loss', default='ranking', type=str,
                        choices=['ranking', 'sigmoid', 'bce', 'mse'],
                        help='ranking|sigmoid|bce')
    parser.add_argument('--mode', default='train', type=str,
                        choices=['train', 'all'],
                        help='train set or whole dataset')
    parser.add_argument('--train-dataset', default='pipal', type=str,
                        choices=['pipal', 'kadid', 'noahfaces', 'noahfaces_enhanced'],
                        help='training dataset')
    parser.add_argument('--val-dataset', default='pipal', type=str,
                        choices=['pipal', 'kadid', 'csiq', 'tid2013', 'qads',
                                 'live', 'liu', 'shrq'],
                        help='validation dataset. for live|liu|shrq specify ' +
                        'an inference-batch-size of 1')
    parser.add_argument('--loader', default='pil', type=str,
                        choices=['cv2', 'pil'],
                        help='image loader')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lambdaC', default=1., type=float,
                        help='pairwise comparisons')
    parser.add_argument('--lambdaRP', default=0., type=float,
                        help='Pearson regularization')
    parser.add_argument('--lambdaRS', default=0., type=float,
                        help='Spearman regularization')
    parser.add_argument('--lambdaRK', default=0., type=float,
                        help='Kendall regularization')
    parser.add_argument('--lambdaRI', default=0., type=float,
                        help='Inner-self regularization')
    parser.add_argument('--invert', action='store_true',
                        help='should be raised if we want lower is better')
    parser.add_argument('--lambda-mode', default='abs', type=str,
                        choices=['abs', 'mse'],
                        help='Regularization formulation')
    parser.add_argument('--sampler', default=None, type=str,
                        choices=['kref', 'kimg', 'mos'],
                        help='advanced sampler')
    parser.add_argument('--sampler-param', default=1, type=int,
                        help='parameter of the sampler')
    parser.add_argument('-p', '--print-freq', default=40, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lpips', dest='lpips', action='store_true',
                        help='use lpips model')
    parser.add_argument('--dists', dest='dists', action='store_true',
                        help='use dists model')
    parser.add_argument('--iqt', dest='iqt', action='store_true',
                        help='use iqt model')
    parser.add_argument('--l2pooling', action='store_true',
                        help='l2pooling in vgg (similar to DISTS)')
    parser.add_argument('--clamping', dest='clamping', action='store_true',
                        help='only positive values in linear modules')
    parser.add_argument('--frozen', dest='frozen', action='store_true',
                        help='freeze the backbone')
    parser.add_argument('--l4p', action='store_true',
                        help='l4p for pearson')
    parser.add_argument('--no-hflip', action='store_true',
                        help='remove horizontal flippling from augmentation')
    parser.add_argument('--no-rot', action='store_true',
                        help='remove rotations from augmentation')
    parser.add_argument('--patch-size', default=64, type=int,
                        help='patch size for cropping')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode; show more information')
    args = parser.parse_args()

    args.hflip = not args.no_hflip
    args.rot = not args.no_rot
    args.pnet_tune = not args.frozen
    return args
