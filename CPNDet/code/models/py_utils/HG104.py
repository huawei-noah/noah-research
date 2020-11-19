# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pdb
import torch
import numpy as np
from . import builder
import torch.nn as nn
from mmcv import Config
import torch.nn.functional as F

from .utils import convolution, residual
from .utils import make_layer, make_layer_revr
from .kp_utils import make_pool_layer, make_unpool_layer
from .bbox import build_assigner, build_sampler, bbox2roi
from .kp_utils import make_merge_layer, make_inter_layer, make_cnv_layer
from .kp_utils import _sigmoid, _ae_loss, _regr_loss, _neg_loss, bbox_overlaps
from .kp_utils import make_tl_layer, make_br_layer, make_region_layer, make_kp_layer, _regr_l1_loss
from .kp_utils import _tranpose_and_gather_feat, _decode, _generate_bboxes, _htbox2roi, _htbox2roi_test, _filter_bboxes

class kp_module(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(kp_module, self).__init__()

        self.n   = n

        curr_mod = modules[0]
        next_mod = modules[1]

        curr_dim = dims[0]
        next_dim = dims[1]

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod, 
            layer=layer, **kwargs
        )  
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = kp_module(
            n - 1, dims[1:], modules[1:], layer=layer, 
            make_up_layer=make_up_layer, 
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)

class hg104(nn.Module):
    def __init__(
        self, db, n, nstack, dims, modules, out_dim, pre=None, cnv_dim=256, 
        make_tl_layer=make_tl_layer, make_br_layer=make_br_layer,
        make_cnv_layer=make_cnv_layer, make_heat_layer=make_kp_layer,
        make_grouping_layer = make_region_layer, make_regr_layer=make_kp_layer,
        make_region_layer = make_region_layer, make_up_layer=make_layer, make_low_layer=make_layer, 
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, make_inter_layer=make_inter_layer, 
        kp_layer=residual
    ):
        super(hg104, self).__init__()

        self.nstack            = nstack
        self._decode           = _decode
        self._generate_bboxes  = _generate_bboxes
        self._db               = db
        self.K                 = self._db.configs["top_k"]
        self.input_size        = db.configs["input_size"]
        self.output_size       = db.configs["output_sizes"][0]
        self.kernel            = self._db.configs["nms_kernel"]
        self.gr_threshold      = self._db.configs["gr_threshold"]
        self.categories        = self._db.configs["categories"]
        
        self.grouping_roi_extractor = builder.build_roi_extractor(Config(self._db._model['grouping_roi_extractor']).item)
        self.region_roi_extractor   = builder.build_roi_extractor(Config(self._db._model['region_roi_extractor']).item)
        
        self.roi_out_size   = Config(self._db._model['grouping_roi_extractor']).item.roi_layer.out_size
        self.iou_threshold  = self._db.configs["iou_threshold"]
        self.train_cfg      = Config(self._db._model['train_cfg'])
        self.bbox_head      = builder.build_bbox_head(Config(self._db._model['bbox_head']).item)
        
        curr_dim = dims[0]

        self.pre = nn.Sequential(
            convolution(7, 3, 128, stride=2),
            residual(3, 128, 256, stride=2)
        ) if pre is None else pre

        self.kps  = nn.ModuleList([
            kp_module(
                n, dims, modules, layer=kp_layer,
                make_up_layer=make_up_layer,
                make_low_layer=make_low_layer,
                make_hg_layer=make_hg_layer,
                make_hg_layer_revr=make_hg_layer_revr,
                make_pool_layer=make_pool_layer,
                make_unpool_layer=make_unpool_layer,
                make_merge_layer=make_merge_layer
            ) for _ in range(nstack)
        ])
        self.cnvs = nn.ModuleList([
            make_cnv_layer(curr_dim, cnv_dim) for _ in range(nstack)
        ])
        
        self.tl_cnvs = nn.ModuleList([
            make_tl_layer(cnv_dim) for _ in range(nstack)
        ])
        self.br_cnvs = nn.ModuleList([
            make_br_layer(cnv_dim) for _ in range(nstack)
        ])

        ## keypoint heatmaps
        self.tl_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            make_heat_layer(cnv_dim, curr_dim, out_dim) for _ in range(nstack)
        ])

        self.regions = nn.ModuleList([
            make_region_layer(cnv_dim, curr_dim) for _ in range(nstack)
         ])
        
        self.region_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(curr_dim, curr_dim, (self.roi_out_size, self.roi_out_size), bias=False),
                              nn.BatchNorm2d(curr_dim),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(curr_dim, out_dim, (1, 1))
                          ) for _ in range(nstack)
                       ])
        
        self.groupings = nn.ModuleList([
            make_grouping_layer(cnv_dim, 32) for _ in range(nstack)
         ])
        
        self.grouping_reduces = nn.ModuleList([
                          nn.Sequential(
                              nn.Conv2d(32, 32, (self.roi_out_size, self.roi_out_size), bias=False),
                              nn.BatchNorm2d(32),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 1, (1, 1))
                          ) for _ in range(nstack)
                       ])

        for tl_heat, br_heat, region_reduce, grouping_reduce in zip \
           (self.tl_heats, self.br_heats, self.region_reduces, self.grouping_reduces):
            tl_heat[-1].bias.data.fill_(-2.19)
            br_heat[-1].bias.data.fill_(-2.19)
            region_reduce[-1].bias.data.fill_(-2.19)
            grouping_reduce[-1].bias.data.fill_(-2.19)

        self.inters = nn.ModuleList([
            make_inter_layer(curr_dim) for _ in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(curr_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])
        self.cnvs_   = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, curr_dim, (1, 1), bias=False),
                nn.BatchNorm2d(curr_dim)
            ) for _ in range(nstack - 1)
        ])

        self.tl_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        self.br_regrs = nn.ModuleList([
            make_regr_layer(cnv_dim, curr_dim, 2) for _ in range(nstack)
        ])
        
        self.relu = nn.ReLU(inplace=True)
        
        self.init_weights()

    def init_weights(self):
        self.grouping_roi_extractor.init_weights()
        self.region_roi_extractor.init_weights()
        
    def _train(self, *xs):
        image         = xs[0]
        tl_inds       = xs[1]
        br_inds       = xs[2]
        gt_detections = xs[3]
        tag_lens      = xs[4]
        
        num_imgs      = image.size(0)

        outs             = []
        grouping_feats   = []
        region_feats     = []
        decode_inputs    = []
        grouping_list    = []
        gt_list          = []
        gt_labels        = []
        sampling_results = []
        grouping_outs    = []
        region_outs      = []
        
        inter = self.pre(image)
        
        layers = zip(
            self.kps,         self.cnvs,
            self.tl_cnvs,     self.br_cnvs, 
            self.tl_heats,    self.br_heats,
            self.tl_regrs,    self.br_regrs,    
            self.regions,     self.groupings
        )
        for ind, layer in enumerate(layers):
            kp_,        cnv_             = layer[0:2]
            tl_cnv_,     br_cnv_         = layer[2:4]
            tl_heat_,    br_heat_        = layer[4:6]
            tl_regr_,    br_regr_        = layer[6:8]
            region_,     grouping_       = layer[8:10]

            kp = kp_(inter)
            cnv = cnv_(kp)
            
            tl_cnv = tl_cnv_(cnv)
            br_cnv = br_cnv_(cnv)
            
            region_feat    = region_(cnv)
            grouping_feat  = grouping_(cnv)
            
            region_feats   += [region_feat]
            grouping_feats += [grouping_feat]

            tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
            tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
            
            decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]
            
            tl_regr = _tranpose_and_gather_feat(tl_regr, tl_inds)
            br_regr = _tranpose_and_gather_feat(br_regr, br_inds)
            
            outs += [tl_heat, br_heat, tl_regr, br_regr]
            
            if ind == self.nstack - 1:
                ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:])
                for i in range(num_imgs):
                    gt_box     = gt_detections[i][:tag_lens[i]][:,:4] 
                    ht_box     = ht_boxes[i]
                    score_inds = ht_box[:,4] > 0 
                    ht_box     = ht_box[score_inds, :4]
                    
                    if ht_box.size(0) == 0:
                        grouping_list += [gt_box] 
                    else:
                        grouping_list += [ht_box]
                        
                    gt_list   += [gt_box]
                    gt_labels += [(gt_detections[i,:tag_lens[i], -1]+ 1).long()]
                
                bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
                bbox_sampler  = build_sampler(self.train_cfg.rcnn.sampler, context=self)
                
                gt_list_ignore = [None for _ in range(num_imgs)]
                
                for i in range(num_imgs):
                    assign_result = bbox_assigner.assign(grouping_list[i], gt_list[i], gt_list_ignore[i], gt_labels[i])
                    sampling_result = bbox_sampler.sample(self.categories, assign_result, grouping_list[i], gt_list[i], gt_labels[i])
                    sampling_results.append(sampling_result)
                
                grouping_rois = bbox2roi([res.bboxes for res in sampling_results]) 
                box_targets   = self.bbox_head.get_target(sampling_results, gt_list, gt_labels, self.train_cfg.rcnn)
                roi_labels = box_targets[0]
                gt_label_inds = roi_labels > self.categories
                roi_labels[gt_label_inds] -= self.categories
                grouping_inds = roi_labels > 0
                grouping_labels = grouping_rois.new_full((grouping_rois.size(0), 1, 1, 1), 0, dtype=torch.float).cuda()
                grouping_labels[grouping_inds] = 1
                region_labels = grouping_rois.new_full((grouping_rois.size(0), self.categories+1), 0, dtype=torch.float).cuda()
                region_labels = region_labels.scatter_(1, roi_labels.unsqueeze(-1), 1)
                region_labels = region_labels[:,1:].unsqueeze(-1).unsqueeze(-1)
                
                grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois)
                
                for grouping_reduce, grouping_roi_feat in zip(self.grouping_reduces, grouping_roi_feats):
                    grouping_outs += [_sigmoid(grouping_reduce(grouping_roi_feat))]
                 
                grouping_scores = grouping_outs[-1][:,0,0,0].clone().detach()
                grouping_scores[gt_label_inds] = 1
                select_inds = grouping_scores >= self.gr_threshold
                region_rois = grouping_rois[select_inds].contiguous()
                region_labels = region_labels[select_inds]
                    
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois)
                for region_reduce, region_roi_feat in zip(self.region_reduces, region_roi_feats):
                     region_outs += [_sigmoid(region_reduce(region_roi_feat))]
                    
                outs += [grouping_outs, grouping_labels, region_outs, region_labels]
                    
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
        return outs

    def _test(self, *xs, **kwargs):
        image     = xs[0]
        no_flip   = kwargs.pop('no_flip')
        image_idx = kwargs['image_idx'] 
        kwargs.pop('image_idx')
        
        num_imgs = image.size(0)
        
        inter    = self.pre(image)

        outs            = []
        region_feats    = []
        grouping_feats  = []
        decode_inputs   = []
        grouping_list   = []
        score_inds_list = []

        layers = zip(
            self.kps,           self.cnvs,
            self.tl_cnvs,       self.br_cnvs, 
            self.tl_heats,      self.br_heats,     
            self.tl_regrs,      self.br_regrs,       
            self.regions,       self.groupings
        )
        
        for ind, layer in enumerate(layers):
            kp_,           cnv_      = layer[0:2]
            tl_cnv_,       br_cnv_   = layer[2:4]
            tl_heat_,      br_heat_  = layer[4:6]
            tl_regr_,      br_regr_  = layer[6:8]
            region_,       grouping_ = layer[8:10]

            kp = kp_(inter)
            cnv = cnv_(kp)
            
            if ind == self.nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                
                region_feat    = region_(cnv)
                grouping_feat  = grouping_(cnv)
                
                region_feats   += [region_feat]
                grouping_feats += [grouping_feat]

                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)
                
                decode_inputs += [tl_heat.clone().detach(), br_heat.clone().detach()]

                outs += [tl_regr, br_regr]
                
                ht_boxes, tlbr_inds, tlbr_scores, tl_clses = self._generate_bboxes(decode_inputs[-2:])
                all_groupings = ht_boxes[:,:, -1].new_full(ht_boxes[:,:, -1].size(), 0, dtype=torch.float)
                
                for i in range(num_imgs):
                    ht_box     = ht_boxes[i]
                    score_inds = ht_box[:,4] > 0
                    ht_box     = ht_box[score_inds, :4]
                    
                    grouping_list += [ht_box]
                    score_inds_list+= [score_inds.unsqueeze(0)]
                        
                grouping_rois = _htbox2roi_test(grouping_list)
                grouping_roi_feats = self.grouping_roi_extractor(grouping_feats, grouping_rois.float())
                
                grouping_scores = self.grouping_reduces[-1](grouping_roi_feats[-1])
                grouping_scores = _sigmoid(grouping_scores)
                
                grouping_inds = grouping_scores[:,0,0,0] >= self.gr_threshold
                
                if grouping_inds.float().sum() > 0:
                    region_rois = grouping_rois[grouping_inds].contiguous().float()
                else:
                    region_rois = grouping_rois
                
                region_roi_feats = self.region_roi_extractor(region_feats, region_rois)
                region_scores = self.region_reduces[-1](region_roi_feats[-1])
                region_scores = _sigmoid(region_scores)
                
                if grouping_inds.float().sum() > 1:
                     _filter_bboxes(ht_boxes, tl_clses, region_scores, grouping_scores, self.gr_threshold)
                        
                if no_flip:
                    all_groupings[score_inds_list[0]] = grouping_scores[:,0,0,0]
                else:
                    all_groupings[torch.cat((score_inds_list[0], score_inds_list[1]), 0)] = grouping_scores[:,0,0,0]
                
                outs += [ht_boxes, all_groupings, tlbr_inds, tlbr_scores, tl_clses, self.gr_threshold]
                
            if ind < self.nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = self.relu(inter)
                inter = self.inters[ind](inter)
                
        return self._decode(*outs[-8:], **kwargs)
    
    def forward(self, *xs, **kwargs):
        if len(xs) > 1:
            return self._train(*xs, **kwargs)
        return self._test(*xs, **kwargs)

class AELoss(nn.Module):
    def __init__(self, pull_weight=1, push_weight=1, regr_weight=1, focal_loss=_neg_loss):
        super(AELoss, self).__init__()

        self.pull_weight   = pull_weight
        self.push_weight   = push_weight
        self.regr_weight   = regr_weight
        self.focal_loss    = focal_loss
        self.ae_loss       = _ae_loss
        self.regr_loss     = _regr_loss
        self._regr_l1_loss = _regr_l1_loss

    def forward(self, outs, targets):
        region_labels   = outs.pop(-1)
        region_outs     = outs.pop(-1)
        grouping_labels = outs.pop(-1)
        grouping_outs   = outs.pop(-1)
        
        stride = 4

        tl_heats = outs[0::stride]
        br_heats = outs[1::stride]
        tl_regrs = outs[2::stride]
        br_regrs = outs[3::stride]
        
        gt_tl_heat = targets[0]
        gt_br_heat = targets[1]
        gt_mask    = targets[2]
        gt_tl_regr = targets[3]
        gt_br_regr = targets[4]
        
        # keypoints loss
        focal_loss = 0

        tl_heats = [_sigmoid(t) for t in tl_heats]
        br_heats = [_sigmoid(b) for b in br_heats]

        focal_loss += self.focal_loss(tl_heats, gt_tl_heat)
        focal_loss += self.focal_loss(br_heats, gt_br_heat)
        
        # grouping loss
        grouping_loss = 0
        grouping_loss+= self.focal_loss(grouping_outs, grouping_labels)
        
        # region loss
        region_loss = 0
        region_loss+= self.focal_loss(region_outs, region_labels)

        regr_loss = 0
        for tl_regr, br_regr in zip(tl_regrs, br_regrs):
            regr_loss += self.regr_loss(tl_regr, gt_tl_regr, gt_mask)
            regr_loss += self.regr_loss(br_regr, gt_br_regr, gt_mask)
        regr_loss = self.regr_weight * regr_loss
        
        loss = (focal_loss + grouping_loss + region_loss + regr_loss) / len(tl_heats)
        
        return loss.unsqueeze(0), (focal_loss / len(tl_heats)).unsqueeze(0), (grouping_loss / len(tl_heats)).unsqueeze(0), \
                          (region_loss / len(tl_heats)).unsqueeze(0), (regr_loss / len(tl_heats)).unsqueeze(0)
