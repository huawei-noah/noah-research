#!/usr/bin/env python3
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Gallery: input, the curves CURL predicted, the Expert C retouch, the output.

    python3 curve_gallery.py <repo-dir> <checkpoint> <out.jpg>
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

REPO = sys.argv[1]
sys.path.insert(0, REPO)
import device
device.DEVICE = torch.device('cpu')
import curves as curve_mod
curve_mod.set_monotone(False)
import model

CKPT = sys.argv[2]
OUT = sys.argv[3]

# name, caption down the left, dB shown on the result
ROWS = [
    ('a4522-Duggan_050712_1693', 'a backlit bird and the hills behind it', 33.4),
    ('a4959-DSC_0129',           'a bust lifted out of deep shadow',        35.8),
    ('a4857-_DSC0008-1',         'flat foliage given its reds back',        32.8),
    ('a4553-Duggan_090331_6590', 'a washed-out plate warmed up',            34.3),
]

# which curves to draw, and how. The three RGB curves do most of the work;
# L carries the tone, S the saturation.
DRAW = [
    ('fc_rgb', 0, 'R',    '#d1495b'),
    ('fc_rgb', 1, 'G',    '#2a9d3f'),
    ('fc_rgb', 2, 'B',    '#3a6ea5'),
    ('fc_lab', 0, 'L',    '#555555'),
    ('fc_hsv', 2, 'S',    '#b5179e'),
]
HEAD_COUNT = {'fc_lab': 3, 'fc_rgb': 3, 'fc_hsv': 4}


def predicted_curves(net, path):
    grabbed = {}
    handles = [getattr(net.curllayer, h).register_forward_hook(
        lambda m, i, o, n=h: grabbed.__setitem__(n, o.detach()))
        for h in HEAD_COUNT]
    img = Image.open(path).convert('RGB')
    x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.
    with torch.no_grad():
        net(x)
    for h in handles:
        h.remove()
    out = {}
    for head, count in HEAD_COUNT.items():
        raw = grabbed[head][0, :count * 16].reshape(count, 16)
        out[head] = torch.exp(raw).numpy()
    return out


def main():
    net = model.CURLNet()
    sd = torch.load(CKPT, map_location='cpu', weights_only=False)
    net.load_state_dict(sd.get('model_state_dict', sd))
    net.eval()

    n = len(ROWS)
    fig, axes = plt.subplots(n, 4, figsize=(17.5, 3.3 * n))
    fig.patch.set_facecolor('white')

    d = os.path.join(REPO, 'adobe5k_dpe')
    for r, (name, caption, db) in enumerate(ROWS):
        inp = os.path.join(d, 'curl_example_test_input', name + '.png')
        tgt = os.path.join(d, 'curl_example_test_output', name + '.png')
        res = glob.glob(os.path.join(d, 'curl_example_test_inference',
                                     name + '_TEST_*.jpg'))[0]

        axes[r][0].imshow(Image.open(inp))
        axes[r][0].set_ylabel(caption, fontsize=10, color='0.35')

        ax = axes[r][1]
        cur = predicted_curves(net, inp)
        x = np.arange(16) / 15.0
        for head, k, label, colour in DRAW:
            ax.plot(x, cur[head][k], lw=2.0, color=colour, label=label)
        ax.axhline(1.0, color='0.8', lw=1.0, ls='--')
        ax.set_xlim(0, 1)
        ax.set_xlabel('input value', fontsize=8, labelpad=1)
        ax.set_ylabel('scaling', fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7)
        ax.set_facecolor('#fbfbfb')
        for s in ax.spines.values():
            s.set_color('0.85')

        axes[r][2].imshow(Image.open(tgt))
        axes[r][3].imshow(Image.open(res))
        axes[r][3].text(0.97, 0.05, '%.1f dB' % db, transform=axes[r][3].transAxes,
                        ha='right', va='bottom', color='white', fontsize=12)

        for c in (0, 2, 3):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            for s in axes[r][c].spines.values():
                s.set_visible(False)

    for c, title in enumerate(['Input', 'The curves it predicted',
                               'Expert C retouch', 'CURL']):
        axes[0][c].set_title(title, fontsize=13, pad=10)

    handles = [plt.Line2D([], [], color=c, lw=2.5,
                          label={'R': 'RGB red', 'G': 'RGB green', 'B': 'RGB blue',
                                 'L': 'CIELab lightness', 'S': 'HSV saturation'}[l])
               for _, _, l, c in DRAW]
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(OUT, dpi=110, facecolor='white')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
