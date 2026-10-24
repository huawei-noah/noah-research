# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Command-line interface: argument definitions and validation.

Kept apart from the training and inference code so both can be driven from a
test or another script without going through argparse.
"""
import argparse
import random

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for training and inference."""
    parser = argparse.ArgumentParser(
        description="Train the CURL neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, help="Number of epochs to train for", default=10000)
    parser.add_argument(
        "--valid_every", type=int, default=10,
        help="Number of epochs between validation passes")
    parser.add_argument(
        "--checkpoint_filepath", default=None,
        help="Checkpoint to load: resumes training, or with "
             "--inference_img_dirpath runs inference")
    parser.add_argument(
        "--inference_img_dirpath", default=None,
        help="Directory of images to run through a saved CURL model. Supply "
             "with --checkpoint_filepath to select inference instead of training")
    parser.add_argument(
        "--training_img_dirpath", default=None,
        help="Directory holding the input/ and output/ image folders and the "
             "images_train.txt, images_valid.txt and images_test.txt splits")
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Adam learning rate")
    parser.add_argument(
        "--resume_lr", type=float, default=None,
        help="Learning rate to use when resuming from --checkpoint_filepath; "
             "defaults to the rate stored in the checkpoint")
    parser.add_argument(
        "--weight_decay", type=float, default=1e-10, help="Adam weight decay")
    parser.add_argument(
        "--lr_schedule", choices=["none", "cosine"], default="none",
        help="Learning-rate schedule over --num_epoch. 'none' keeps --lr "
             "fixed for the whole run; 'cosine' anneals it to "
             "--lr_min following a cosine, one step per epoch")
    parser.add_argument(
        "--lr_min", type=float, default=1e-6,
        help="Floor the cosine schedule anneals --lr down to")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="DataLoader worker processes")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Training batch size. Above 1 needs --crop_size, because FiveK "
             "images vary in size. Evaluation always runs at 1 so per-image "
             "PSNR and SSIM are reported")
    parser.add_argument(
        "--crop_size", type=int, default=None,
        help="Random square crop taken from each training image")
    parser.add_argument(
        "--tf32", action="store_true",
        help="Ampere and later: run matmuls and convolutions in TF32")
    parser.add_argument(
        "--compile", dest="use_compile", action="store_true",
        help="Wrap the network in torch.compile")
    parser.add_argument(
        "--amp", choices=["off", "bf16", "fp16"], default="off",
        help="Mixed-precision autocast for the training step")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Seed torch, numpy and random; runs are not comparable without it")
    parser.add_argument(
        "--dump_curves_every", type=int, default=0,
        help="Record the predicted curves every N epochs, into curves.jsonl "
             "in the log directory. 0 disables it")
    parser.add_argument(
        "--monotone", action="store_true",
        help="Read the curve heads as monotone tone mappings rather than "
             "scaling curves, so the realised mapping cannot invert tones. "
             "Changes the meaning of the head outputs, so checkpoints are not "
             "interchangeable with the default parameterisation")
    parser.add_argument(
        "--reg_weight", type=float, default=1e-6,
        help="Weight on the curve smoothness penalty. At the default it is "
             "about 0.001%% of the loss; 3e-3 to 1e-2 makes it 3-9%%")
    parser.add_argument(
        "--ssim_window_size", type=int, default=5,
        help="Window size for the SSIM term of the training loss")

    return parser


def validate(args, parser) -> bool:
    """Checks the arguments and seeds the run. Returns True for an inference run.

    The training paths are meaningless for inference, so they are checked here
    rather than by argparse: marking them required would make the documented
    inference command fail on a missing training argument.
    """
    inference_run = (args.checkpoint_filepath is not None
                     and args.inference_img_dirpath is not None)
    if not inference_run and args.training_img_dirpath is None:
        parser.error('training needs --training_img_dirpath; for inference pass '
                     'both --checkpoint_filepath and --inference_img_dirpath')
    if args.valid_every < 1:
        parser.error('--valid_every must be at least 1')
    if args.batch_size > 1 and args.crop_size is None:
        parser.error('--batch_size above 1 needs --crop_size: FiveK images '
                     'vary in size and cannot be stacked without cropping')
    import curves
    curves.set_monotone(args.monotone)
    if args.tf32:
        # Ampere and later: run matmuls and convolutions in TF32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    if args.seed is not None:
        # Seed every source the loop draws on: weight init, the DataLoader
        # shuffle, and the augmentation flips in data.py.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    return inference_run
