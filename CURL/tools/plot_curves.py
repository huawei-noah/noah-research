#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This is a PyTorch implementation of CURL: Neural Curve Layers for Global
# Image Enhancement: https://arxiv.org/pdf/1911.13175.pdf
#
# Please cite paper if you use this code.
#
# Authors: Sean Moran (sean.j.moran@gmail.com), 2020
"""Plot the curves a training run predicted, from curves.jsonl.

    python3 tools/plot_curves.py curves.jsonl -o curves.png

Draws one panel per curve head. Each panel shows the predicted knots at a
selection of epochs, oldest faint and newest solid, so the shape's trajectory
over the run is visible in one image. A second figure tracks curvature, which
is zero for a straight line: the smoothness penalty drives the knots there when
nothing else trains them.
"""
import argparse
import json

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

HEADS = ('fc_lab', 'fc_rgb', 'fc_hsv')
LABELS = {'fc_lab': 'CIELab', 'fc_rgb': 'RGB', 'fc_hsv': 'HSV'}


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda r: r['epoch'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('jsonl')
    ap.add_argument('-o', '--out', default='curves.png')
    ap.add_argument('--curve', type=int, default=0,
                    help='which curve within each head to draw (default 0)')
    ap.add_argument('--snapshots', type=int, default=6,
                    help='how many epochs to overlay')
    args = ap.parse_args()

    rows = load(args.jsonl)
    if not rows:
        raise SystemExit('no rows in %s' % args.jsonl)
    print('%d snapshots, epochs %d to %d'
          % (len(rows), rows[0]['epoch'], rows[-1]['epoch']))

    picks = [rows[i] for i in
             np.linspace(0, len(rows) - 1, min(args.snapshots, len(rows))).astype(int)]
    x = np.arange(16) / 15.0

    fig, axes = plt.subplots(1, len(HEADS), figsize=(13, 3.8))
    for ax, head in zip(np.atleast_1d(axes), HEADS):
        for i, row in enumerate(picks):
            if head not in row:
                continue
            knots = np.array(row[head][args.curve])
            shade = 0.15 + 0.85 * (i / max(len(picks) - 1, 1))
            ax.plot(x, knots, marker='o', markersize=3, linewidth=1.4,
                    color=plt.cm.viridis(shade),
                    label='epoch %d' % row['epoch'])
        ax.axhline(1.0, color='0.6', linewidth=0.8, linestyle='--')
        ax.set_title('%s, curve %d' % (LABELS[head], args.curve))
        ax.set_xlabel('channel value')
        ax.set_ylabel('scaling factor')
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print('wrote', args.out)

    # Curvature over the run: the quantity that separates a curve that is being
    # fitted to images from one relaxing under the smoothness penalty alone.
    fig2, ax = plt.subplots(figsize=(7, 3.4))
    epochs = [r['epoch'] for r in rows]
    for head in HEADS:
        key = head + '_curvature'
        if key in rows[0]:
            ax.plot(epochs, [r.get(key, np.nan) for r in rows],
                    label=LABELS[head], linewidth=1.6)
    ax.set_xlabel('epoch')
    ax.set_ylabel('max |second difference|')
    ax.set_title('Curve curvature over training (0 = a straight line)')
    ax.legend(frameon=False)
    fig2.tight_layout()
    out2 = args.out.replace('.png', '_curvature.png')
    fig2.savefig(out2, dpi=140)
    print('wrote', out2)


if __name__ == '__main__':
    main()
