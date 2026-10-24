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

'''
import matplotlib
matplotlib.use('agg')
import numpy as np
import sys
import os
import torch
import matplotlib.pyplot as plt
from util import ImageProcessing
import logging

np.set_printoptions(threshold=sys.maxsize)


class Evaluator():
    """Evaluates a trained network on a dataset split.

    Runs the network over every image in a data loader, computes the loss, PSNR
    and SSIM, and writes the first 30 enhanced images to disk (annotated with
    their per-image PSNR/SSIM) for qualitative inspection.
    """

    def __init__(self, criterion, data_loader, split_name, log_dirpath):
        """Initialisation function for the data loader

        :param criterion: loss function
        :param data_loader: an instance of the DataLoader class for the dataset of interest
        :param split_name: name of the split e.g. "test", "validation"
        :param log_dirpath: logging directory
        :returns: N/A
        :rtype: N/A

        """
        super().__init__()
        self.criterion = criterion
        self.data_loader = data_loader
        self.split_name = split_name
        self.log_dirpath = log_dirpath

    def evaluate(self, net, epoch=0):
        """Evaluates a network on a specified split of a dataset e.g. test, validation

        Per-image PSNR and SSIM are written to ``<split>_per_image.csv`` in the
        log directory, one row per image per evaluation. The mean alone cannot
        support an error bar: comparing two arms needs the *paired* per-image
        differences, since all arms see the same images and image difficulty is
        the dominant source of variance. See ``tools/paired_bootstrap.py``.

        :param net: PyTorch neural network data structure
        :param epoch: current epoch
        :returns: average loss, average PSNR, average SSIM
        :rtype: float, float, float

        """
        per_image = []
        
        psnr_avg = 0.0
        ssim_avg = 0.0
        examples = 0  # images seen; evaluation always runs one image at a time
        running_loss = 0
        num_batches = 0
        batch_size = 1

        out_dirpath = self.log_dirpath + "/" + self.split_name.lower()
        if not os.path.isdir(out_dirpath):
            os.mkdir(out_dirpath)

        # switch model to evaluation mode; run on whichever device the model
        # already lives on (set by the caller)
        net.eval()
        device = next(net.parameters()).device

        with torch.no_grad():
            for batch_num, data in enumerate(self.data_loader, 0):

                input_img_batch, output_img_batch, name = data['input_img'].to(device), data['output_img'].to(device), \
                    data['name']
                input_img_batch = input_img_batch.unsqueeze(0)

                for i in range(0, input_img_batch.shape[0]):

                    img = input_img_batch[i, :, :, :]
                    img = torch.clamp(img, 0, 1)

                    net_output_img_example = net(img)

                    if net_output_img_example.shape[2]!=output_img_batch.shape[2]:
                        net_output_img_example=net_output_img_example.transpose(2,3)

                    loss = self.criterion(net_output_img_example[:, 0:3, :, :],
                                          output_img_batch[:, 0:3, :, :])

                    # Metrics are computed in numpy on 1x3xHxW float arrays in [0, 1]
                    output_img_batch_rgb = np.expand_dims(
                        output_img_batch.squeeze(0).data.cpu().numpy(), axis=0)

                    net_output_img_example_rgb = np.expand_dims(
                        net_output_img_example.squeeze(0).data.cpu().numpy(), axis=0)
                    net_output_img_example_rgb = np.clip(
                        net_output_img_example_rgb, 0, 1)

                    running_loss += loss.item()
                    examples += batch_size
                    num_batches += 1

                    psnr_example = ImageProcessing.compute_psnr(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32), 1.0)
                    ssim_example = ImageProcessing.compute_ssim(output_img_batch_rgb.astype(np.float32),
                                                                net_output_img_example_rgb.astype(np.float32))

                    psnr_avg += psnr_example
                    ssim_avg += ssim_example
                    per_image.append((name[0], psnr_example, ssim_example))

                    # Only the first ~30 enhanced images are written to disk, to save time.
                    if batch_num <= 30:
                        net_output_img_example = (
                            net_output_img_example_rgb[0, 0:3, :, :] * 255).astype('uint8')

                        plt.imsave(out_dirpath + "/" + name[0].split(".")[0] + "_" + self.split_name.upper() + "_" + str(epoch + 1) + "_" + str(
                            examples) + "_PSNR_" + str("{0:.3f}".format(psnr_example)) + "_SSIM_" + str(
                            "{0:.3f}".format(ssim_example)) + ".jpg",
                            ImageProcessing.swapimdims_3HW_HW3(net_output_img_example))

        psnr_avg = psnr_avg / num_batches
        ssim_avg = ssim_avg / num_batches

        logging.info('loss_%s: %.5f psnr_%s: %.3f ssim_%s: %.3f' % (
            self.split_name, (running_loss / examples), self.split_name, psnr_avg, self.split_name, ssim_avg))

        # One row per image, appended across evaluations. Written after the
        # means are logged so a crash mid-eval cannot leave a partial epoch
        # looking complete.
        csv_path = os.path.join(self.log_dirpath,
                                '%s_per_image.csv' % self.split_name.lower())
        new_file = not os.path.isfile(csv_path)
        with open(csv_path, 'a') as handle:
            if new_file:
                handle.write('epoch,image,psnr,ssim\n')
            for image_name, psnr, ssim in per_image:
                handle.write('%d,%s,%.6f,%.6f\n' % (epoch + 1, image_name, psnr, ssim))

        loss = (running_loss / examples)

        return loss, psnr_avg, ssim_avg
