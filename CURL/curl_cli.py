# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""The ``curl`` command: enhance image files with a trained model.

``main.py`` is the research entry point: it wants a dataset directory laid out
in ``input``/``output`` folders, a text file of image ids, and a checkpoint
path. That is the right shape for training and for scoring a split, and the
wrong shape for "enhance this photograph". This takes file paths.

    curl-enhance photo.jpg
    curl-enhance ~/photos --out ~/enhanced --checkpoint my_model.pt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

#: Files we will try to open. Anything else in a directory is skipped.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')

#: The bundled model, used when --checkpoint is not given. Present in a git
#: checkout; a wheel installed without it will not find one.
DEFAULT_CHECKPOINT_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'pretrained_models', '*', '*.pt')

#: Below this the pooling in the backbone has nothing left to pool.
MINIMUM_EDGE = 48


def default_checkpoint():
    """Returns the bundled checkpoint path, or None when none is present."""
    found = sorted(glob.glob(DEFAULT_CHECKPOINT_GLOB))
    # Prefer a weights-only file if both are shipped: it loads faster.
    for path in found:
        if path.endswith('_weights.pt'):
            return path
    return found[0] if found else None


def gather_images(paths):
    """Expands the given files and directories into a list of image paths."""
    images = []
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if name.lower().endswith(IMAGE_EXTENSIONS):
                    images.append(os.path.join(path, name))
        elif os.path.isfile(path):
            images.append(path)
        else:
            print('not found, skipping: %s' % path, file=sys.stderr)
    return images


def select_device(requested):
    """Picks CUDA, then Apple Silicon, then CPU, unless one was requested."""
    if requested and requested != 'auto':
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_network(checkpoint_filepath, device):
    """Loads a checkpoint into a CURLNet and returns it in eval mode."""
    import device as device_module
    device_module.DEVICE = device          # colour ops read this at call time
    import model

    net = model.CURLNet()
    checkpoint = torch.load(checkpoint_filepath, map_location=device,
                            weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    net.load_state_dict(state)
    net.to(device)
    net.eval()
    return net


def enhance_image(net, device, image_filepath):
    """Enhances one image and returns it as a uint8 HWC array."""
    image = Image.open(image_filepath).convert('RGB')
    if min(image.size) < MINIMUM_EDGE:
        raise ValueError('both edges must be at least %d px, this is %dx%d'
                         % (MINIMUM_EDGE, image.size[0], image.size[1]))
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        enhanced, _ = net(tensor)
    enhanced = torch.clamp(enhanced[:, 0:3, :, :], 0, 1)
    out = (enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255)
    return out.round().astype(np.uint8)


def run_enhance(args):
    """Enhances every image given, writing PNGs into the output directory."""
    checkpoint_filepath = args.checkpoint or default_checkpoint()
    if checkpoint_filepath is None:
        print('no checkpoint found; pass --checkpoint', file=sys.stderr)
        return 1

    images = gather_images(args.paths)
    if not images:
        print('no images to enhance', file=sys.stderr)
        return 1

    device = select_device(args.device)
    net = load_network(checkpoint_filepath, device)
    os.makedirs(args.out, exist_ok=True)
    print('enhancing %d image%s on %s'
          % (len(images), '' if len(images) == 1 else 's', device))

    failures = 0
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0] + '.png'
        try:
            result = enhance_image(net, device, path)
        except Exception as exc:                  # one bad file must not stop
            print('  %s: %s' % (os.path.basename(path), exc), file=sys.stderr)
            failures += 1
            continue
        Image.fromarray(result).save(os.path.join(args.out, name))
        print('  %s -> %s' % (os.path.basename(path), name))

    print('wrote %d to %s' % (len(images) - failures, args.out))
    return 1 if failures and failures == len(images) else 0


def build_parser():
    """Builds the argument parser for the enhance command."""
    parser = argparse.ArgumentParser(
        prog='curl-enhance',
        description='Enhance photographs with a trained CURL model.')
    sub = parser.add_subparsers(dest='command')

    enhance = sub.add_parser('enhance', help='enhance images')
    enhance.add_argument('paths', nargs='+',
                         help='image files, or directories of them')
    enhance.add_argument('--out', default='enhanced',
                         help='output directory (default: enhanced)')
    enhance.add_argument('--checkpoint', default=None,
                         help='model to load (default: the bundled one)')
    enhance.add_argument('--device', default='auto',
                         help='cuda, mps, cpu, or auto (default: auto)')
    enhance.set_defaults(func=run_enhance)
    return parser


def main(argv=None):
    """Entry point for the ``curl-enhance`` command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'command', None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
