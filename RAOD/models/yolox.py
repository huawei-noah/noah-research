#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# 
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import torch
import torch.nn as nn
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
# 2023-04-19 Modified by Huawei, import adaptive adjustment module
from .adaptive_module import Adaptive_Module


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    def __init__(self, backbone=None, head=None, nf=16, gamma_range=[1.,4.]):
        super().__init__()
        # 2023-04-19 Modified by Huawei, definition of adaptive adjustment module
        self.TMM = Adaptive_Module(in_ch=3, nf=nf, gamma_range=gamma_range)

        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        # 2023-04-19 Modified by Huawei, apply adaptive adjustment module on RAW inputs
        x_tm = self.TMM(x)
        x_tm = torch.clamp(x_tm, 0, 1) * 255.0
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x_tm)
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)
            outputs = {"total_loss": loss, "iou_loss": iou_loss, "l1_loss": l1_loss, "conf_loss": conf_loss, "cls_loss": cls_loss, "num_fg": num_fg}
        else:
            outputs = self.head(fpn_outs)
        return outputs
