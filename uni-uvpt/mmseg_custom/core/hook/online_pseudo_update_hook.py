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


import os.path as osp

import os
import numpy as np
from mmcv.runner import HOOKS, Hook
from PIL import Image
from mmcv.runner.hooks import IterTimerHook
import torch
from mmcv.parallel import MMDataParallel
from tqdm import tqdm
from mmseg.core.evaluation import intersect_and_union
import pandas as pd
import copy


@HOOKS.register_module()
class OnlineSimplePseudoUpdateHook(IterTimerHook):
    def __init__(self, thre):
        super(OnlineSimplePseudoUpdateHook, self).__init__()
        self.thre = thre

    def before_train_iter(self, runner):
        model = runner.model
        model.module.inference_thre = self.thre
        model.eval()
        model = MMDataParallel(model, device_ids=[0])
        with torch.no_grad():
            result = model(runner.data_batch["img"].data, runner.data_batch["img_metas"].data,
                           return_loss=False, rescale=False)
        # runner.data_batch["gt_semantic_seg"].data[0] = runner.data_batch["gt_semantic_seg"].data[0].repeat(2, 1, 1, 1)
        flip = runner.data_batch["img_metas"].data[0][0]["flip"]
        flip_direction = runner.data_batch["img_metas"].data[0][0]['flip_direction']
        for i in range(len(result)):
            if not flip:
                runner.data_batch["gt_semantic_seg"].data[0][i] = torch.tensor(result[i]).unsqueeze(0)
            else:
                if flip_direction == 'horizontal':
                    runner.data_batch["gt_semantic_seg"].data[0][i] = torch.tensor(result[i]).flip(1).unsqueeze(0)
                elif flip_direction == 'vertical':
                    runner.data_batch["gt_semantic_seg"].data[0][i] = torch.tensor(result[i]).flip(0).unsqueeze(0)
        model.train()
        model.module.inference_thre = None


@HOOKS.register_module()
class ADELEPseudoUpdateHook(Hook):
    def __init__(self, total_iter, curv, threshold=0.9):
        epoch = np.arange(1, total_iter + 1)
        self.update_iter = []
        for cate in curv:
            a, b, c, _ = curv[cate]
            relative_change = abs(abs(self.derivation(epoch, a, b, c)) -
                                  abs(self.derivation(1, a, b, c))) / abs(self.derivation(1, a, b, c))
            # relative_change[relative_change > 1] = 0
            print(cate, curv[cate], (np.sum(relative_change <= threshold) + 1) * 100)
            if len(np.where(relative_change > threshold)[0]) != 0:
                self.update_iter.append((np.where(relative_change > threshold)[0][0] + 1) * 100)  # 100 is pretrained model eval_iter
            else:
                self.update_iter.append((len(relative_change) + 1) * 100)
        self.updated_class_list = []
        # self.update_iter = [1000] * 19
        # self.updated_class_list = [i for i in range(19)]
        self.prev_pred = dict()
        self.max_prob = dict()
        self.prev_crop_bbox = dict()
        self.gt = dict()

    def derivation(self, x, a, b, c):
        x = x + 1e-6  # numerical robustness
        return 3 * a * x ** 2 + 2 * b * x + c

    def update_label(self, seg_label, prev_pred, max_prob, mask_threshold=0.8):
        b, h, w = seg_label.size()
        prev_pred = prev_pred.unsqueeze(0)
        seg_change_indx = (seg_label != prev_pred) & (max_prob > mask_threshold)
        class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
        for element in self.updated_class_list:
            class_indx_seg_argmax = class_indx_seg_argmax | (prev_pred == element)
        seg_change_indx = seg_change_indx & class_indx_seg_argmax
        seg_label[seg_change_indx] = prev_pred[seg_change_indx]
        return seg_label

    def before_train_iter(self, runner):
        img_metas = runner.data_batch["img_metas"].data[0]
        data_num = len(img_metas)
        for i in range(data_num):
            img_name = img_metas[i]["ori_filename"].split("/")[1]
            if img_name not in self.gt:
                self.gt[img_name] = img_metas[i]["gt_full"]

        if runner.data_loader.epoch <= 1:
            return
        for class_idx in range(len(self.update_iter)):
            if class_idx not in self.updated_class_list:
                if runner.iter >= self.update_iter[class_idx]:
                    self.updated_class_list.append(class_idx)
        # if runner.iter > 0:
        if len(self.updated_class_list) > 0:
            for i in range(data_num):
                img_name = img_metas[i]["ori_filename"].split("/")[1]
                if img_name not in self.prev_pred:
                    continue
                flip = img_metas[i]["flip"]
                flip_direction = img_metas[i]["flip_direction"]
                up, down, left, right = self.prev_crop_bbox[img_name]
                seg_label = self.gt[img_name][:, up:down, left:right]
                if not set(np.unique(seg_label)).isdisjoint(self.updated_class_list[1:]):
                    prev_pred = self.prev_pred[img_name]
                    max_prob = self.max_prob[img_name]
                    new_label = self.update_label(seg_label, prev_pred, max_prob)
                    self.gt[img_name][:, up:down, left:right] = new_label
                up, down, left, right = img_metas[i]["crop_bbox"]
                new_label = self.gt[img_name][:, up:down, left:right]
                if flip:
                    new_label = new_label.flip(2) if flip_direction == "horizontal" else new_label.flip(1)
                runner.data_batch["gt_semantic_seg"].data[0][i] = new_label

    def after_train_iter(self, runner):
        img_metas = runner.data_batch["img_metas"].data[0]
        data_num = len(img_metas)
        prev_pred = runner.outputs["prev_pred"]
        max_prob = runner.outputs["max_prob"]
        for i in range(data_num):
            img_name = img_metas[i]["ori_filename"].split("/")[1]
            flip = img_metas[i]["flip"]
            flip_direction = img_metas[i]["flip_direction"]
            if flip:
                self.prev_pred[img_name] = prev_pred[i].flip(1) if flip_direction == "horizontal" else \
                    prev_pred[i].flip(0)
                self.max_prob[img_name] = max_prob[i].flip(1) if flip_direction == "horizontal" else max_prob[i].flip(0)
            else:
                self.prev_pred[img_name] = prev_pred[i]
                self.max_prob[img_name] = max_prob[i]
            self.prev_crop_bbox[img_name] = img_metas[i]["crop_bbox"]


@HOOKS.register_module()
class OnlineADELEPseudoUpdateHook(Hook):
    def __init__(self, img_num, batch_size, class_num, output_dir, train_eval_interval=200,
                 queue_length=1000, threshold=0.9, ratio=0.66):
        self.prev_pred = dict()
        self.max_prob = dict()
        self.prev_crop_bbox = dict()
        self.gt = dict()
        self.area_intersect = []
        self.area_union = []
        self.IoU = []
        self.queue_length = queue_length
        self.train_eval_interval = train_eval_interval
        self.updated_class_list = [255]
        # self.updated_class_list = [255] + [i for i in range(19)]
        self.curv_ratio_threshold = threshold
        self.outdir = output_dir
        self.one_epoch_iter = img_num / batch_size
        self.last_update_iter = [0 for _ in range(class_num)]
        self.last_update_IoU_index = [0 for _ in range(class_num)]
        self.label_threshold = []
        self.class_num = class_num
        self.update_ratio = ratio

    def derivation(self, x, a, b, c):
        x = x + 1e-6  # numerical robustness
        return 3 * a * x ** 2 + 2 * b * x + c

    def update_label(self, seg_label, prev_pred, max_prob, mask_threshold=0.8):
        b, h, w = seg_label.size()
        prev_pred = prev_pred.unsqueeze(0)
        seg_change_indx = torch.zeros((b, h, w), dtype=torch.bool)
        for i in range(self.class_num):
            seg_change_indx = seg_change_indx | ((prev_pred == i) & (max_prob >= self.label_threshold[i]))
            # prev_pred[(max_prob < self.label_threshold[i]) * (prev_pred == i)] = 255
        # seg_change_indx = (seg_label != prev_pred) & (max_prob > mask_threshold)
        class_indx_seg_argmax = torch.zeros((b, h, w), dtype=torch.bool)
        for element in self.updated_class_list:
            class_indx_seg_argmax = class_indx_seg_argmax | (prev_pred == element)
        seg_change_indx = seg_change_indx & class_indx_seg_argmax
        seg_label[seg_change_indx] = prev_pred[seg_change_indx]
        return seg_label

    def before_train_iter(self, runner):
        img_metas = runner.data_batch["img_metas"].data[0]
        data_num = len(img_metas)
        for i in range(data_num):
            img_name = img_metas[i]["ori_filename"].split("/")[1]
            if img_name not in self.gt:
                self.gt[img_name] = img_metas[i]["gt_full"]

        if runner.data_loader.epoch < 1:  # ensure correct the label after one whole epoch
            return

        # if runner.iter > 0:
        if len(self.updated_class_list) > 1:
            for i in range(data_num):
                img_name = img_metas[i]["ori_filename"].split("/")[1]
                if img_name not in self.prev_pred:
                    continue
                flip = img_metas[i]["flip"]
                flip_direction = img_metas[i]["flip_direction"]
                up, down, left, right = self.prev_crop_bbox[img_name]
                seg_label = self.gt[img_name][:, up:down, left:right]
                if not set(np.unique(seg_label)).isdisjoint(self.updated_class_list[1:]):
                    prev_pred = self.prev_pred[img_name]
                    max_prob = self.max_prob[img_name]
                    new_label = self.update_label(seg_label, prev_pred, max_prob)
                    self.gt[img_name][:, up:down, left:right] = new_label
                up, down, left, right = img_metas[i]["crop_bbox"]
                new_label = self.gt[img_name][:, up:down, left:right]
                if flip:
                    new_label = new_label.flip(2) if flip_direction == "horizontal" else new_label.flip(1)
                runner.data_batch["gt_semantic_seg"].data[0][i] = new_label
        else:
            for i in range(data_num):
                flip = img_metas[i]["flip"]
                flip_direction = img_metas[i]["flip_direction"]
                up, down, left, right = img_metas[i]["crop_bbox"]
                img_name = img_metas[i]["ori_filename"].split("/")[1]
                if flip:
                    gt = self.gt[img_name][:, up:down, left:right].flip(2) if flip_direction == "horizontal" else \
                        self.gt[img_name][:, up:down, left:right].flip(1)
                else:
                    gt = self.gt[img_name][:, up:down, left:right]
                runner.data_batch["gt_semantic_seg"].data[0][i] = gt

    def after_train_iter(self, runner):
        img_metas = runner.data_batch["img_metas"].data[0]
        data_num = len(img_metas)
        prev_pred = runner.outputs["prev_pred"]
        max_prob = runner.outputs["max_prob"]
        num_classes = len(runner.data_loader.iter_loader._dataset.CLASSES)
        ignore_index = runner.data_loader.iter_loader._dataset.ignore_index
        for i in range(data_num):
            img_name = img_metas[i]["ori_filename"].split("/")[1]
            flip = img_metas[i]["flip"]
            flip_direction = img_metas[i]["flip_direction"]
            if flip:
                self.prev_pred[img_name] = prev_pred[i].flip(1) if flip_direction == "horizontal" else \
                    prev_pred[i].flip(0)
                self.max_prob[img_name] = max_prob[i].flip(1) if flip_direction == "horizontal" else max_prob[i].flip(0)
            else:
                self.prev_pred[img_name] = prev_pred[i]
                self.max_prob[img_name] = max_prob[i]
            self.prev_crop_bbox[img_name] = img_metas[i]["crop_bbox"]
            up, down, left, right = img_metas[i]["crop_bbox"]
            gt = self.gt[img_name][:, up:down, left:right].numpy()
            area_intersect, area_union, _, _ = intersect_and_union(self.prev_pred[img_name].unsqueeze(0).numpy(),
                                                                   gt, num_classes=num_classes,
                                                                   ignore_index=ignore_index)
            if len(self.area_intersect) < self.queue_length:
                self.area_intersect.append(area_intersect)
                self.area_union.append(area_union)
            else:
                del self.area_intersect[0]
                del self.area_union[0]
                self.area_intersect.append(area_intersect)
                self.area_union.append(area_union)

        if (len(self.area_intersect) == self.queue_length) and runner.iter % self.train_eval_interval == 0:
            IoU = sum(self.area_intersect) / sum(self.area_union)
            self.IoU.append(IoU)
            if not osp.exists(os.path.join(self.outdir, "PseudoIoURecord.csv")):
                Cate_IoU = {str(i): [IoU[i].item()] for i in range(len(IoU))}
                pd.DataFrame(Cate_IoU).to_csv(os.path.join(self.outdir, "PseudoIoURecord.csv"), index=False)
            else:
                Cate_IoU = {str(i): IoU[i].item() for i in range(len(IoU))}
                data = pd.read_csv(os.path.join(self.outdir, "PseudoIoURecord.csv"))
                data.loc[len(data)] = Cate_IoU
                data.to_csv(os.path.join(self.outdir, "PseudoIoURecord.csv"), index=False)
        if runner.data_loader.epoch < 1:  # ensure correct the label after one whole epoch
            return

        if runner.iter % self.train_eval_interval == 0:
            backup = copy.copy(self.updated_class_list)
            for cate in self.updated_class_list[1:]:
                if runner.iter >= self.last_update_iter[cate] + self.one_epoch_iter:  # update at least one whole epoch
                    backup.remove(cate)
            self.updated_class_list = copy.copy(backup)
            for cate in range(self.class_num):
                if cate in self.updated_class_list:
                    continue
                x = [i for i in range(1, len(self.IoU) + 1 - self.last_update_IoU_index[cate])]
                y = [self.IoU[i][cate] for i in range(self.last_update_IoU_index[cate], len(self.IoU))]
                # x = [i for i in range(1, len(self.IoU) + 1)]
                # y = [self.IoU[i][cate] for i in range(len(self.IoU))]
                z = np.polyfit(x, y, 3)
                a, b, c, _ = z
                relative_change = abs(abs(self.derivation(len(self.IoU) - self.last_update_IoU_index[cate], a, b, c)) -
                                      abs(self.derivation(1, a, b, c))) / abs(self.derivation(1, a, b, c))
                # relative_change = abs(abs(self.derivation(len(self.IoU), a, b, c)) -
                #                       abs(self.derivation(1, a, b, c))) / abs(self.derivation(1, a, b, c))
                if self.curv_ratio_threshold < relative_change < 1:
                    self.updated_class_list.append(cate)
                    self.last_update_iter[cate] = runner.iter
                    self.last_update_IoU_index[cate] = len(self.IoU)
            runner.logger.info("update class list at iter: %s is: %s" % (runner.iter, self.updated_class_list))
            # update the threshold for each category
            prev_preds = []
            max_probs = []
            for key in self.prev_pred:
                prev_preds.append(self.prev_pred[key])
                max_probs.append(self.max_prob[key])
            prev_preds = torch.stack(prev_preds)
            max_probs = torch.stack(max_probs)
            self.label_threshold = []
            for i in range(self.class_num):
                x = max_probs[prev_preds == i].numpy()
                if len(x) == 0:
                    self.label_threshold.append(0)
                    continue
                x = np.sort(x)
                if np.int(np.round(len(x) * self.update_ratio)) < len(x):
                    self.label_threshold.append(x[np.int(np.round(len(x) * self.update_ratio))])
                else:
                    self.label_threshold.append(x[-1])
