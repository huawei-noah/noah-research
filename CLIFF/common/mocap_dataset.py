# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from torch.utils.data import Dataset

from common.imutils import process_image
from common.utils import estimate_focal_length


class MocapDataset(Dataset):
    def __init__(self, img_bgr_list, detection_list):
        self.img_bgr_list = img_bgr_list
        self.detection_list = detection_list

    def __len__(self):
        return len(self.detection_list)

    def __getitem__(self, idx):
        """
        bbox: [batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id]
        :param idx:
        :return:
        """
        item = {}
        img_idx = int(self.detection_list[idx][0].item())
        img_bgr = self.img_bgr_list[img_idx]
        img_rgb = img_bgr[:, :, ::-1]
        img_h, img_w, _ = img_rgb.shape
        focal_length = estimate_focal_length(img_h, img_w)

        bbox = self.detection_list[idx][1:5]
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(img_rgb, bbox)

        item["norm_img"] = norm_img
        item["center"] = center
        item["scale"] = scale
        item["crop_ul"] = crop_ul
        item["crop_br"] = crop_br
        item["img_h"] = img_h
        item["img_w"] = img_w
        item["focal_length"] = focal_length
        return item
