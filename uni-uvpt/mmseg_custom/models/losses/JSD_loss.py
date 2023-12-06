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
# Modified from https://github.com/lhoyer/DAFormer
# 

import torch
import torch.nn.functional as F


def get_mixture_label(probs, weight):
    weighted_probs = [weight[i] * prob for i, prob in enumerate(probs)]
    mixture_label = (torch.stack(weighted_probs)).sum(axis=0)
    mixture_label = torch.clamp(mixture_label, 1e-3, 1 - 1e-3)  # h,c,h,w
    return mixture_label


def calc_jsd_multiscale(weight, preds, criterion=None, gt_semantic_seg=None, threshold=0.8):
    if gt_semantic_seg is not None and criterion is not None:
        # loss = [criterion(pred, gt_semantic_seg) for pred in preds]
        Mask_label255 = (gt_semantic_seg != 255).float()
        preds = [F.interpolate(pred, size=gt_semantic_seg.shape[2:], mode='bilinear', align_corners=True)
                 for pred in preds]
        probs = [F.softmax(logits, dim=1) for logits in preds]
        mixture_label = get_mixture_label(probs, weight)
        max_probs = torch.amax(mixture_label * Mask_label255, dim=(0, 1), keepdim=True)
        mask = max_probs.ge(threshold).float()
        logp_mixture = mixture_label.log()
        log_probs = [torch.sum(F.kl_div(logp_mixture, prob, reduction='none') * mask, dim=1) for prob in probs]
        consistency = sum(log_probs)
        # losses = dict()
        # losses["acc_seg"] = torch.tensor([0.0]).to(0)
        # # for i, item in enumerate(loss):
        # #     if "loss_seg" in item:
        # #         losses["loss_seg_" + str(i)] = item["loss_seg"] * weight[i]
        # #     elif "loss_GtA_selftraining_losss" in item:
        # #         losses["loss_GtA_selftraining_losss_" + str(i)] = item["loss_GtA_selftraining_losss"] * weight[i]
        # #     elif "loss_ce" in item:
        # #         losses["loss_ce_" + str(i)] = item["loss_ce"] * weight[i]
        # #     losses["acc_seg"] += item["acc_seg"]
        # losses["acc_seg"] /= 3
        # TODO new try
        loss = criterion(preds[0], gt_semantic_seg)
        losses = loss
        ##
        losses["consistency_loss"] = torch.mean(consistency)
    else:
        # preds = [(logits - torch.min(logits)) / (torch.max(logits) - torch.min(logits)) for logits in preds]
        preds = [torch.clamp(pred, 1e-3, 1 - 1e-3) for pred in preds]
        mixture_label = get_mixture_label(preds, weight)
        # logp_mixture = mixture_label.log()
        # log_probs = [F.kl_div(logp_mixture, prob, reduction='mean') for prob in preds]
        log_probs = [F.mse_loss(prob, mixture_label, reduction='mean') for prob in preds]
        losses = dict()
        losses["consistency_loss"] = torch.mean(torch.stack(log_probs))
    return losses
