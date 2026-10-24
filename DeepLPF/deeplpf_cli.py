# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The ``deeplpf`` command: enhance image files with a trained model.

``main.py`` is the research entry point, and it asks for a dataset directory
laid out in ``input``/``output`` folders, a text file of image ids, and a
checkpoint path. That is the right shape for training and for scoring a split,
and the wrong shape for "enhance this photograph", which is what most people
arrive wanting. This takes file paths.

    deeplpf enhance photo.jpg
    deeplpf enhance ~/photos --out ~/enhanced --checkpoint my_model.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

import model

#: Files we will try to open. Anything else in a directory is skipped.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')

#: The bundled model, used when --checkpoint is not given. Present in a git
#: checkout; a wheel installed without it will not find one.
DEFAULT_CHECKPOINT_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pretrained_models', '*', '*.pt')

#: Below this the reflection padding in the U-Net has nothing to reflect.
MINIMUM_EDGE = 32


def default_checkpoint():
    """The bundled checkpoint, or None if this install has no weights.

    :returns: path to the bundled Adobe-DPE checkpoint
    :rtype: str or None

    """
    matches = sorted(glob.glob(DEFAULT_CHECKPOINT_GLOB))
    return matches[0] if matches else None


def gather_images(paths):
    """Expand the given files and directories into a list of image paths.

    :param paths: files or directories named on the command line
    :returns: image filepaths, sorted within each argument
    :rtype: list

    """
    images = []
    for path in paths:
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                if entry.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(os.path.join(path, entry))
        else:
            images.append(path)
    return images


def select_device(requested):
    """Pick the device to run on.

    :param requested: 'auto', or a device string such as 'cpu' or 'cuda'
    :returns: the device
    :rtype: torch.device

    """
    if requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_network(checkpoint_filepath, device, learn_filter_count=False,
                 colour_knots=None):
    """Build the network and load a checkpoint into it.

    :param checkpoint_filepath: state dict to load
    :param device: device to move the network to
    :param learn_filter_count: the checkpoint was trained with learned gates
    :param colour_knots: tone-curve knots the checkpoint was trained with
    :returns: the network, in eval mode
    :rtype: model.DeepLPFNet

    """
    net = model.DeepLPFNet(learn_filter_count=learn_filter_count,
                           colour_knots=colour_knots)
    net.load_state_dict(torch.load(checkpoint_filepath, map_location='cpu'))
    net.to(device)
    net.eval()
    return net


def enhance_image(net, device, image_filepath):
    """Run one image through the network.

    :param net: the network
    :param device: device to run on
    :param image_filepath: image to read
    :returns: the enhanced image as HxWx3 uint8
    :rtype: numpy.ndarray

    """
    img = np.array(Image.open(image_filepath))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    height, width = img.shape[:2]
    if min(height, width) < MINIMUM_EDGE:
        raise ValueError('%s is %dx%d; the network needs both edges to be at '
                         'least %d pixels' % (image_filepath, width, height,
                                              MINIMUM_EDGE))

    tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced = torch.clamp(net(torch.clamp(tensor, 0, 1)), 0, 1)

    out = enhanced.squeeze(0)[0:3, :, :].cpu().numpy()
    return (np.transpose(out, (1, 2, 0)) * 255).astype(np.uint8)


def run_enhance(args):
    """Enhance every image named, writing the results to the output directory.

    :param args: parsed arguments for the ``enhance`` subcommand
    :returns: process exit status
    :rtype: int

    """
    checkpoint = args.checkpoint or default_checkpoint()
    if checkpoint is None:
        print('no checkpoint given and none bundled with this install; pass '
              '--checkpoint', file=sys.stderr)
        return 2

    images = gather_images(args.images)
    if not images:
        print('no images found in: %s' % ', '.join(args.images), file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    device = select_device(args.device)
    net = load_network(checkpoint, device, args.learn_filter_count,
                       args.colour_knots)
    print('%s on %s' % (os.path.basename(checkpoint), device))

    failures = 0
    for image_filepath in images:
        name = os.path.splitext(os.path.basename(image_filepath))[0]
        destination = os.path.join(args.out, name + args.suffix + '.png')
        try:
            Image.fromarray(enhance_image(net, device, image_filepath)).save(destination)
        except (OSError, ValueError) as exc:
            # One unreadable or undersized file should not abandon the rest of
            # a directory the user asked for.
            print('  %s: %s' % (image_filepath, exc), file=sys.stderr)
            failures += 1
            continue
        print('  %s -> %s' % (image_filepath, destination))

    return 1 if failures and failures == len(images) else 0


def build_parser():
    """Build the ``deeplpf`` argument parser.

    :returns: the parser
    :rtype: argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(
        prog='deeplpf',
        description='Enhance photographs with a trained DeepLPF model.')
    subparsers = parser.add_subparsers(dest='command', required=True)

    enhance = subparsers.add_parser(
        'enhance', help='enhance image files or a directory of them')
    enhance.add_argument('images', nargs='+',
                         help='image files, or directories of images')
    enhance.add_argument('--out', default='enhanced',
                         help='directory to write to (default: enhanced/)')
    enhance.add_argument('--checkpoint', default=None,
                         help='model to use (default: the bundled Adobe-DPE model)')
    enhance.add_argument('--device', default='auto',
                         help="'auto' (default), 'cpu', 'cuda', 'mps'")
    enhance.add_argument('--suffix', default='_enhanced',
                         help="appended to each output filename (default: '_enhanced')")
    enhance.add_argument('--learn-filter-count', action='store_true',
                         dest='learn_filter_count',
                         help='the checkpoint was trained with a learned filter '
                              'count (one gate per filter instance)')
    enhance.add_argument('--colour-knots', type=int, default=None,
                         dest='colour_knots',
                         help='the checkpoint was trained with a colour head '
                              'using this many tone-curve knots')
    enhance.set_defaults(func=run_enhance)

    return parser


def main(argv=None):
    """Entry point for the ``deeplpf`` console script.

    :param argv: argument list, defaulting to ``sys.argv[1:]``
    :returns: process exit status
    :rtype: int

    """
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
