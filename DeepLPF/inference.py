# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Running a saved checkpoint over a directory of images."""
import logging
import os

import matplotlib.pyplot as plt
import torch

import metric
import model
from data import Adobe5kDataLoader, Dataset
from util import ImageProcessing


def run_inference(args, device, log_dirpath):
    """Score a saved checkpoint over the images named in a list file.

    ``--inference_img_dirpath`` should hold an ``input`` directory, with a text
    file naming one image per line in the level above it.

    :param args: parsed command-line arguments
    :param device: device to run on
    :param log_dirpath: directory for per-image scores and enhanced images
    :returns: N/A

    """
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath
    inference_img_list_path = args.inference_img_list_path

    # ``inference_img_dirpath`` holds an ``input`` directory; the list file
    # names one image id per line, an id being the filename up to the first
    # "-". A retouched ``output`` directory beside it is optional: with one,
    # every image is scored; without one, the enhanced images are written out
    # and there is nothing to score against.
    loader = Adobe5kDataLoader(data_dirpath=inference_img_dirpath,
                               img_ids_filepath=inference_img_list_path)
    data_dict = loader.load_data(require_output=False)
    have_targets = all(entry.get('output_img') is not None
                       for entry in data_dict.values())

    dataset = Dataset(data_dict=data_dict, normaliser=1, is_inference=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=False, num_workers=0)

    logging.info("Performing inference with images in directory: "
                 + inference_img_dirpath)

    net = model.DeepLPFNet(
        learn_filter_count=args.learn_filter_count,
        colour_knots=args.colour_knots if args.colour_head else None)
    net.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
    net.to(device)
    net.eval()

    if have_targets:
        evaluator = metric.Evaluator(model.DeepLPFLoss(), data_loader, "test",
                                     log_dirpath)
        evaluator.evaluate(net, epoch=0)
        return

    logging.info("No retouched targets found, so no PSNR/SSIM is computed; "
                 "writing the enhanced images to " + log_dirpath)
    out_dirpath = os.path.join(log_dirpath, "inference")
    os.makedirs(out_dirpath, exist_ok=True)

    with torch.no_grad():
        for data in data_loader:
            img = torch.clamp(data['input_img'].to(device), 0, 1)
            enhanced = torch.clamp(net(img), 0, 1)
            rgb = enhanced.squeeze(0)[0:3, :, :].cpu().numpy()
            name = os.path.splitext(data['name'][0])[0] + "_enhanced.png"
            plt.imsave(os.path.join(out_dirpath, name),
                       ImageProcessing.swapimdims_3HW_HW3((rgb * 255).astype('uint8')))
            logging.info("Wrote " + name)
