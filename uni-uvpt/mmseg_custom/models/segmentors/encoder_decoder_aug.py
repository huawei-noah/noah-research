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
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import random
from collections import OrderedDict

from mmseg.core import add_prefix
from mmseg.ops import resize
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder


def load_state_dict(state_dict, key='generator'):
    clean_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(key + '.'):
            clean_state_dict[k[len(key)+1:]] = v

    return clean_state_dict

@SEGMENTORS.register_module()
class EncoderDecoderAug(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                 backbone,
                 decode_head,
                 decode_head0,
                 decode_head1,
                 decode_head2,
                 decode_head3,
                 decode_head4,
                 decode_head5,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(EncoderDecoderAug, self).__init__(
            backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, pretrained, init_cfg)

        self.k2head = {0:2,1:1,2:0,3:0,4:3,5:3,6:4}
        self.num_heads = 6 # global+K

        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.decode_head_aug0 = builder.build_head(decode_head0)
        self.decode_head_aug1 = builder.build_head(decode_head1)
        self.decode_head_aug2 = builder.build_head(decode_head2)
        self.decode_head_aug3 = builder.build_head(decode_head3)
        self.decode_head_aug4 = builder.build_head(decode_head4)
        self.decode_head_aug5 = builder.build_head(decode_head5)

        # if pretrained is not None:
        #     para = torch.load(pretrained)["state_dict"]
        #     self.backbone.load_state_dict(load_state_dict(para, 'backbone'), strict=True)
        #     self.decode_head.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug0.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug1.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug2.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug3.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug4.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     self.decode_head_aug5.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
        #     print('successfully Loading pretrained model from {}.............',format(pretrained))
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

        self.decode_head_aug = [self.decode_head_aug0, self.decode_head_aug1, self.decode_head_aug2,
                                self.decode_head_aug3, self.decode_head_aug4, self.decode_head_aug5]

        self.decode_head = self.decode_head_aug[-1]
        

    def _decode_head_aug_forward_train(self, x, img_metas, gt_semantic_seg, k):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        head_id = list(range(self.num_heads))
        head_id.remove(self.k2head[k])

        for i in head_id:
            loss_decode = self.decode_head_aug[i].forward_train(x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, 'decode_{}'.format(i)))

        return losses

    def forward_train(self, img, img_metas, gt_semantic_seg, 
                        img_adain, img_styleaug, img_snow, img_frost, img_cartoon, img_fda_random, img_fda):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            XXXX
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        img_list = [img_adain, img_styleaug, img_fda_random, img_fda, img_snow, img_frost, img_cartoon]
        k = random.randint(0, len(img_list) - 1)

        x = self.extract_feat(img_list[k])
        losses = dict()
        loss_decode = self._decode_head_aug_forward_train(x, img_metas, gt_semantic_seg, k)
        losses.update(loss_decode)

        return losses

