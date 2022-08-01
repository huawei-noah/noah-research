# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
from os.path import join
import sys
import torch

curr_dir = os.path.dirname(os.path.abspath(__file__))
yolo_path = join(curr_dir, "pytorch_yolo_v3_master")
sys.path.append(yolo_path)
from util import write_results
from darknet import Darknet


class HumanDetector:
    def __init__(self):
        # Set up the neural network
        cfgfile = join(yolo_path, "cfg/yolov3.cfg")
        weightsfile = join(curr_dir, "../data/ckpt/yolov3.weights")
        # Input resolution of the network. Increase to increase accuracy. Decrease to increase speed
        reso = 416
        self.num_classes = 80
        self.confidence = 0.5
        self.nms_thesh = 0.4

        print("Load the yolo_v3 checkpoint from path:", weightsfile)
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightsfile)

        self.model.net_info["height"] = reso
        self.in_dim = reso
        assert self.in_dim % 32 == 0
        assert self.in_dim > 32

        # If there's a GPU availible, put the model on GPU
        self.model.cuda()
        # Set the model in evaluation mode
        self.model.eval()

    @torch.no_grad()
    def detect_batch(self, img_bgr_batch, img_dim_batch):
        output = self.model(img_bgr_batch, True)  # GPU available
        output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thesh)

        img_dim_batch = img_dim_batch.repeat(1, 2)
        img_dim_batch = torch.index_select(img_dim_batch, 0, output[:, 0].long())
        scale = torch.min(self.in_dim / img_dim_batch, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (self.in_dim - scale * img_dim_batch[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.in_dim - scale * img_dim_batch[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scale
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim_batch[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim_batch[i, 1])

        # get the human category
        human_idx = output[:, -1] == 0
        human_result = output[human_idx, :]
        # batch_id, min_x, min_y, max_x, max_y, det_conf, nms_conf, category_id
        return human_result
