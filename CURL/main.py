# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""CURL: Neural Curve Layers for Global Image Enhancement.

Entry point for training and inference. Parses the command line, sets up
logging and the device, and dispatches to :mod:`train` or :mod:`inference`.

    python3 main.py --training_img_dirpath=./data/          # train
    python3 main.py --checkpoint_filepath=model.pt \
                    --inference_img_dirpath=./images/       # inference

Training supports a batch size above one via --batch_size, which needs
--crop_size because the images vary in size. Evaluation and inference run at a
batch size of one so per-image PSNR and SSIM are reported and saved.
"""
import datetime
import logging
import os

from torch.utils.tensorboard import SummaryWriter

import cli
import inference
import train
from device import DEVICE


def main():
    """Parses arguments, then trains or runs inference."""
    # Parse before creating anything on disk, so --help and a rejected command
    # line leave no empty log directory behind.
    parser = cli.build_parser()
    args = parser.parse_args()
    inference_run = cli.validate(args, parser)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.makedirs(log_dirpath, exist_ok=True)

    handlers = [logging.FileHandler(log_dirpath + "/curl.log"),
                logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(args.num_epoch))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(args.valid_every))
    logging.info('Training image directory: ' + str(args.training_img_dirpath))
    logging.info('Seed: ' + (str(args.seed) if args.seed is not None
                             else 'unseeded (runs are not comparable)'))
    logging.info('Learning rate: ' + str(args.lr))
    logging.info('Batch size: ' + str(args.batch_size))
    logging.info('Curve parameterisation: '
                 + ('monotone mapping' if args.monotone else 'scaling curve'))
    logging.info('Device: ' + str(DEVICE))
    logging.info('##############################')

    if inference_run:
        inference.run_inference(args, log_dirpath)
    else:
        writer = SummaryWriter()
        try:
            train.run_training(args, log_dirpath, writer)
        finally:
            writer.close()


if __name__ == "__main__":
    main()
