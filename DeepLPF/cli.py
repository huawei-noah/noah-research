# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Command-line arguments for training and inference.

Split out of ``main.py``: the flag definitions are most of what a reader
scrolls past, so they sit apart from what the program does with them.
"""
import argparse


def build_parser():
    """Build the argument parser for :func:`main.main`.

    :returns: the parser, with every training and inference flag registered
    :rtype: argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False,
        help="Number of epochs (default 100000)", default=100000)
    parser.add_argument(
        "--batch_size", type=int, required=False, default=1,
        help="Training batch size (default 1). Evaluation and inference always "
             "run at 1 so per-image PSNR/SSIM are reported individually. To "
             "reproduce the paper's reported results, train at 1.")
    parser.add_argument(
        "--crop_size", type=int, required=False, default=0,
        help="If >0, randomly crop training images to this square size. "
             "Required for --batch_size>1 because FiveK images vary in size "
             "and the default collation needs a uniform size.")
    parser.add_argument(
        "--valid_every", type=int, required=False,
        help="Number of epochs after which to compute validation accuracy",
        default=25)
    parser.add_argument(
        "--checkpoint_filepath", required=False,
        help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance",
        default=None)
    parser.add_argument(
        "--training_img_dirpath", required=False,
        help="Directory containing images to train a DeepLPF model instance",
        default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument(
        "--inference_img_list_path", required=False,
        help="Plain text file containing the names of the images to inference")
    parser.add_argument(
        "--train_img_list_path", required=False,
        help="Plain text file containing the names of the training images")
    parser.add_argument(
        "--valid_img_list_path", required=False,
        help="Plain text file containing the names of the validation images")
    parser.add_argument(
        "--test_img_list_path", required=False,
        help="Plain text file containing the names of the test images")

    parser.add_argument(
        "--checkpoint_every", type=int, required=False, default=None,
        help="Also save a checkpoint every N validation rounds regardless of "
             "whether validation improved. Saving only on improvement leaves "
             "late training almost unsampled, so there is nothing to average "
             "over and no way to estimate run-to-run variability.")
    parser.add_argument(
        "--seed", type=int, required=False, default=None,
        help="Seed for torch, numpy and Python RNGs. Without it every run "
             "starts from a different initialisation and shuffle order, so "
             "arms of an ablation differ by luck as well as by the change "
             "under test. Set it for any comparison between runs.")

    parser.add_argument(
        "--tf32", action="store_true",
        help="Allow TF32 matmul/conv on Ampere+ GPUs. Faster, with a small "
             "loss of mantissa precision. Off by default so the replication "
             "path keeps full fp32.")
    parser.add_argument(
        "--compile", action="store_true", dest="use_compile",
        help="Wrap the network in torch.compile. Fuses kernels, which is "
             "where this model's time goes at batch size 1. Costs a one-off "
             "compilation at startup.")
    parser.add_argument(
        "--cuda_graphs", action="store_true",
        help="Capture the whole training step (forward, loss, backward, Adam) "
             "in one CUDA graph per image shape and replay it. Removes the "
             "per-kernel launch cost the model is bound by at batch size 1. "
             "Same kernels and arithmetic as eager; Adam runs capturable, "
             "which computes its bias corrections in fp32 on the device. "
             "Requires CUDA. See trainstep.py.")
    parser.add_argument(
        "--amp", choices=("bf16",), default=None,
        help="Run the network's forward pass under bf16 autocast; the loss "
             "stays fp32. Changes the numerics, so it is off by default.")
    parser.add_argument(
        "--learn_filter_count", action="store_true",
        help="Predict one gate per filter instance and penalise the mean gate, "
             "so the network learns how many of the three instances per branch "
             "an image needs instead of always using all three. Off by "
             "default; the default architecture is unchanged.")
    parser.add_argument(
        "--gate_weight", type=float, required=False, default=3e-3,
        help="Weight on the L1 gate penalty of --learn_filter_count. Too small "
             "and every gate pins at 1; too large and the branches collapse to "
             "the identity.")
    parser.add_argument(
        "--colour_head", action="store_true",
        help="Apply a global colour mixer and per-channel tone curve before the "
             "cubic filter. Every filter head is diagonal, so without this no "
             "filter can express white balance, saturation or a hue shift. Off "
             "by default; identity at initialisation when on.")
    parser.add_argument(
        "--colour_knots", type=int, required=False, default=16,
        help="Knots in the tone curve of --colour_head; 0 keeps the mixer alone.")

    return parser
