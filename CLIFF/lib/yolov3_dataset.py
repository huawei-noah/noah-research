# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from torch.utils.data import Dataset
from lib.pytorch_yolo_v3_master.preprocess import letterbox_image


class DetectionDataset(Dataset):
    def __init__(self, img_bgr_list, inp_dim):
        self.img_bgr_list = img_bgr_list
        self.inp_dim = inp_dim

    def __len__(self):
        return len(self.img_bgr_list)

    def __getitem__(self, idx):
        item = {}

        img_bgr = self.img_bgr_list[idx]
        norm_img = (letterbox_image(img_bgr, (self.inp_dim, self.inp_dim)))
        norm_img = norm_img[:, :, ::-1].transpose((2, 0, 1)).copy()
        norm_img = norm_img / 255.0

        dim = np.array([img_bgr.shape[1], img_bgr.shape[0]])

        item["norm_img"] = norm_img
        item["dim"] = dim

        return item
