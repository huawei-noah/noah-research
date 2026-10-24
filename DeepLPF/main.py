# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem please see the README.

BATCH SIZE: training supports a batch size greater than one via --batch_size,
which needs --crop_size because FiveK images vary in size. Evaluation and
inference run at a batch size of 1 so per-image PSNR/SSIM are reported and
saved individually. To reproduce the paper's reported results, train with a
batch size of 1.
'''
import datetime
import logging
import os
import random

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import cli
import inference
import train


def main():
    """Entry point for training and inference.

    Parses command-line arguments and either runs inference with a saved
    checkpoint, when both ``--checkpoint_filepath`` and
    ``--inference_img_dirpath`` are supplied, or trains a model from scratch.
    The two paths live in :mod:`inference` and :mod:`train`; this function sets
    up logging, seeds, the device, and dispatches.

    """

    # Parse before creating anything on disk: this ran first, so every --help
    # and every rejected command line left an empty log_* directory and a
    # TensorBoard runs/ directory behind in the working directory.
    parser = cli.build_parser()
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.makedirs(log_dirpath, exist_ok=True)

    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    writer = SummaryWriter()

    # The three training paths are required for training and meaningless for
    # inference, so they are checked here rather than by argparse: marking them
    # required=True made the README's inference command fail on a missing
    # --train_img_list_path.
    if args.valid_every < 1:
        parser.error('--valid_every must be at least 1; it is the number of '
                     'epochs between evaluations, and 0 divided the epoch '
                     'counter by zero once the first epoch finished')

    inference_run = (args.checkpoint_filepath is not None
                     and args.inference_img_dirpath is not None)
    if inference_run and args.inference_img_list_path is None:
        parser.error('inference needs --inference_img_list_path, a plain text '
                     'file listing the ids of the images to enhance')
    if not inference_run:
        missing = [name for name in ('training_img_dirpath', 'train_img_list_path',
                                     'valid_img_list_path', 'test_img_list_path')
                   if getattr(args, name) is None]
        if missing:
            parser.error('training needs %s; for inference pass both '
                         '--checkpoint_filepath and --inference_img_dirpath'
                         % ', '.join('--' + name for name in missing))
    if args.tf32:
        # Ampere and later: run matmuls and convolutions in TF32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    if args.seed is not None:
        # Seed every source the training loop draws on: weight init, the
        # DataLoader's shuffle, and the augmentation flips in data.py.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(args.num_epoch))
    logging.info('Seed: ' + (str(args.seed) if args.seed is not None else 'unseeded (runs are not comparable)'))
    logging.info('TF32: ' + str(args.tf32) + ', torch.compile: ' + str(args.use_compile)
                 + ', CUDA graphs: ' + str(args.cuda_graphs) + ', amp: ' + str(args.amp))
    logging.info('Learned filter count: ' + str(args.learn_filter_count)
                 + (', gate weight ' + str(args.gate_weight)
                    if args.learn_filter_count else ''))
    logging.info('Colour head: ' + (str(args.colour_knots) + ' curve knots'
                                    if args.colour_head else 'off'))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(args.valid_every))
    logging.info('Training image directory: ' + str(args.training_img_dirpath))
    logging.info('List of images to inference: ' + str(args.inference_img_list_path))
    logging.info('List of test images: ' + str(args.test_img_list_path))
    logging.info('List of validation images: ' + str(args.valid_img_list_path))
    logging.info('List of training images: ' + str(args.train_img_list_path))

    logging.info('##############################')

    # Select the best available device: CUDA GPU, then Apple Silicon (MPS), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info('Using device: ' + str(device))

    # Training batch size is configurable; evaluation and inference always run
    # at a batch size of 1 so per-image PSNR/SSIM are reported and saved.
    logging.info('Training batch size: ' + str(args.batch_size)
                 + (', crop ' + str(args.crop_size) if args.crop_size else ''))

    if inference_run:
        inference.run_inference(args, device, log_dirpath)
    else:
        train.run_training(args, device, log_dirpath, writer)


if __name__ == "__main__":
    main()
