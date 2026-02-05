# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

import torchvision

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import pandas as pd
import json
import time
import pickle
from torchvision.utils import save_image
import json
import random


def get_ind(vid, index, ds):
    if ds == "ego4d":
        return torchvision.io.read_image(f"{vid}/{index:06}.jpg")
    else:
        raise NameError("Invalid Dataset")


## Data Loader for Ego4D
class R3MBuffer(IterableDataset):
    def __init__(
        self, ego4dpath, num_workers, source1, source2, alpha, datasources, doaug="none"
    ):
        self._num_workers = max(1, num_workers)
        self.alpha = alpha
        self.curr_same = 0
        self.data_sources = datasources
        self.doaug = doaug

        # Augmentations
        if doaug in ["rc", "rctraj"]:
            self.aug = torch.nn.Sequential(
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            )
        else:
            self.aug = lambda a: a

        # Load Data
        if "ego4d" in self.data_sources:
            print("Ego4D")
            self.manifest = pd.read_csv(f"{ego4dpath}manifest.csv")
            print(self.manifest)
            self.ego4dlen = len(self.manifest)
        else:
            raise NameError("Invalid Dataset")

    def _sample(self):
        t0 = time.time()
        ds = random.choice(self.data_sources)

        vidid = np.random.randint(0, self.ego4dlen)
        m = self.manifest.iloc[vidid]
        vidlen = m["len"]
        txt = m["txt"]
        label = txt[2:]  ## Cuts of the "C " part of the text
        vid = m["path"]

        start_ind = np.random.randint(1, 2 + int(self.alpha * vidlen))
        end_ind = np.random.randint(int((1 - self.alpha) * vidlen) - 1, vidlen)
        s1_ind = np.random.randint(2, vidlen)
        s0_ind = np.random.randint(1, s1_ind)
        s2_ind = np.random.randint(s1_ind, vidlen + 1)

        if self.doaug == "rctraj":
            ### Encode each image in the video at once the same way
            im0 = get_ind(vid, start_ind, ds)
            img = get_ind(vid, end_ind, ds)
            imts0 = get_ind(vid, s0_ind, ds)
            imts1 = get_ind(vid, s1_ind, ds)
            imts2 = get_ind(vid, s2_ind, ds)
            allims = torch.stack([im0, img, imts0, imts1, imts2], 0)
            allims_aug = self.aug(allims / 255.0) * 255.0

            im0 = allims_aug[0]
            img = allims_aug[1]
            imts0 = allims_aug[2]
            imts1 = allims_aug[3]
            imts2 = allims_aug[4]
        else:
            ### Encode each image individually
            im0 = self.aug(get_ind(vid, start_ind, ds) / 255.0) * 255.0
            img = self.aug(get_ind(vid, end_ind, ds) / 255.0) * 255.0
            imts0 = self.aug(get_ind(vid, s0_ind, ds) / 255.0) * 255.0
            imts1 = self.aug(get_ind(vid, s1_ind, ds) / 255.0) * 255.0
            imts2 = self.aug(get_ind(vid, s2_ind, ds) / 255.0) * 255.0

        im = torch.stack([im0, img, imts0, imts1, imts2])
        return (im, label)

    def __iter__(self):
        while True:
            yield self._sample()
