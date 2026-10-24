# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Running a saved checkpoint over a directory of images."""
import logging

import torch
import torchvision.transforms as transforms

import metric
import model
from data import Adobe5kDataLoader, Dataset
from device import DEVICE


def run_inference(args, log_dirpath):
    """Enhances every image listed in images_inference.txt under the given directory.

    The directory holding the images must have "input" somewhere in its path,
    and the targets, if any, "output"; the ids are listed one per line in
    images_inference.txt one level above.
    """
    EVAL_BATCH_SIZE = 1


    '''
    The directory should have "input" in the path, and one level above it 
    for inference are located, there should be a file "images_inference.txt with each image filename as one line i.e."

    images_inference.txt    ../
                            a1000.tif
                            a1242.tif
                            etc
    '''
    inference_data_loader = Adobe5kDataLoader(data_dirpath=args.inference_img_dirpath,
                                              img_ids_filepath=args.inference_img_dirpath+"/images_inference.txt")
    inference_data_dict = inference_data_loader.load_data()
    inference_dataset = Dataset(data_dict=inference_data_dict,
                                transform=transforms.Compose([transforms.ToTensor()]), normaliser=1,
                                is_inference=True)

    inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False,
                                                        num_workers=args.num_workers)

    '''
    Performs inference on all the images in the directory
    '''
    logging.info(
        "Performing inference with images in directory: " + args.inference_img_dirpath)

    net = model.CURLNet()
    checkpoint = torch.load(args.checkpoint_filepath, map_location=DEVICE)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    criterion = model.CURLLoss(ssim_window_size=args.ssim_window_size,
                                reg_weight=args.reg_weight)

    inference_evaluator = metric.Evaluator(
        criterion, inference_data_loader, "test", log_dirpath)

    inference_evaluator.evaluate(net, epoch=0)


