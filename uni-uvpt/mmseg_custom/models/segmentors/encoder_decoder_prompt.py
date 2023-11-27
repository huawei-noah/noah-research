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

from collections import OrderedDict


from mmseg.ops import resize
from mmseg.models.segmentors import EncoderDecoder
from mmseg.models.builder import SEGMENTORS
from mmcv.runner import auto_fp16

from ..losses import calc_jsd_multiscale, get_mixture_label


def load_state_dict(state_dict, key='generator'):
    clean_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k.startswith(key + '.'):
            clean_state_dict[k[len(key)+1:]] = v

    return clean_state_dict


@SEGMENTORS.register_module()
class EncoderDecoderPrompt(EncoderDecoder):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, pretrained, init_cfg)
        
        if pretrained is not None:
            para = torch.load(pretrained)["state_dict"]
            self.backbone.load_state_dict(load_state_dict(para, 'backbone'), strict=False) # loading pretrained model and prompt module is not in state_dict
            self.decode_head.load_state_dict(load_state_dict(para, 'decode_head'), strict=True) 
            print('successfully Loading pretrained model from {}.............',format(pretrained))

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        losses = self(**data_batch)
        if "prev_pred" in losses:
            prev_pred = losses.pop("prev_pred")
            max_prob = losses.pop("max_prob")
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(data_batch['img_metas']),
                prev_pred=prev_pred,
                max_prob=max_prob)
        else:
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss,
                log_vars=log_vars,
                num_samples=len(data_batch['img_metas']))

        return outputs


@SEGMENTORS.register_module()
class MultiscaleEncoderDecoderPrompt(EncoderDecoderPrompt):
    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(
            backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg, pretrained, init_cfg)

        if pretrained is not None:
            para = torch.load(pretrained)["state_dict"]
            self.backbone.load_state_dict(load_state_dict(para, 'backbone'),
                                          strict=False)  # loading pretrained model and prompt module is not in state_dict
            self.decode_head.load_state_dict(load_state_dict(para, 'decode_head'), strict=True)
            print('successfully Loading pretrained model from {}.............', format(pretrained))
        self.weight = nn.Parameter(torch.Tensor(3))
        self.weight.data.fill_(1)
        self.weight.to(0)
        self.backbone_type = self.backbone._get_name()

        if "Swin" in self.backbone_type:
            self.output_num = len(self.backbone.stages)  
        elif "mit" in self.backbone_type:
            self.output_num = len(self.backbone.block)
        elif "ResNetV1c" in self.backbone_type:
            self.output_num = len(self.backbone.res_layers)
        elif "VGG" in self.backbone_type:
            self.output_num = len(self.backbone.stage_blocks)-1
        else:
            raise EOFError()

        self.weight_prompt = [nn.Parameter(torch.Tensor(3)) for _ in range(self.output_num)]

        for weight in self.weight_prompt:
            weight.data.fill_(1)
            weight.to(0)

        self.prediction_consistency_loss_weight = 1.0
        self.feature_consistency_loss_weight = 0.001
        self.no_update_pseudo = False
        self.no_feature_consistency_loss = False
        self.no_prediction_consistency_loss = False

    def forward_train(self, img, img_metas, gt_semantic_seg):
        losses = dict()
        inputs_small = F.interpolate(img, scale_factor=0.5, mode='bilinear', align_corners=True)
        inputs_large = F.interpolate(img, scale_factor=2, mode='bilinear', align_corners=True)
        x1 = self.extract_feat(img)
        out1 = x1['outs'] if type(x1) is dict else x1
        pred1 = self.decode_head.forward(out1)
        # input to be scaled e.g 0.7
        x2 = self.extract_feat(inputs_small)
        out2 = x2['outs'] if type(x2) is dict else x2
        pred2 = self.decode_head.forward(out2)
        # # input to be scaled e.g 1.5
        x3 = self.extract_feat(inputs_large)
        out3 = x3['outs'] if type(x3) is dict else x3
        pred3 = self.decode_head.forward(out3)

        if not self.no_feature_consistency_loss:

            for i in range(self.output_num):
                prompt = x1["prompts"][i].permute(0, 2, 3, 1) # b h w c
                size = prompt.shape[1:3]
                prompt_small = F.interpolate(x2["prompts"][i], size=size, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
                prompt_large = F.interpolate(x3["prompts"][i], size=size, mode='bilinear', align_corners=True).permute(0, 2, 3, 1)

                prompt_consistency_loss = calc_jsd_multiscale(F.softmax(self.weight_prompt[i], dim=0),
                                                              [prompt.flatten(1, 2), prompt_small.flatten(1, 2),
                                                               prompt_large.flatten(1, 2)])
                losses["prompt_consistency_loss_" + str(i)] = \
                    prompt_consistency_loss["consistency_loss"] * self.feature_consistency_loss_weight

        loss_decode = calc_jsd_multiscale(F.softmax(self.weight, dim=0), [pred1, pred2, pred3],
                                          self.decode_head.losses, gt_semantic_seg, threshold=0.8)
        if not self.no_prediction_consistency_loss:
            loss_decode["consistency_loss"] *= self.prediction_consistency_loss_weight
        else:
            del loss_decode["consistency_loss"]
        losses.update(loss_decode)
        
        # online pseudo update
        if not self.no_update_pseudo:
            size = gt_semantic_seg.shape[2:]
            seg_logits = [pred1, pred2, pred3]
            seg_logits = [resize(seg_logit, size=size, mode='bilinear', align_corners=self.align_corners,
                                 warning=False) for seg_logit in seg_logits]
            probs = [F.softmax(logits, dim=1) for logits in seg_logits]
            result = get_mixture_label(probs, F.softmax(self.weight, dim=0))
            output = torch.argmax(result, dim=1)
            max_prob, _ = torch.max(result, dim=1)
            losses["prev_pred"] = output.detach().cpu()
            losses["max_prob"] = max_prob.detach().cpu()

        return losses

   
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x["outs"], img_metas) if type(x) is dict else \
            self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

