#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
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


import os
import random
import torch
import torch.nn as nn
import torch.distributed as dist
from yolox.exp import Exp as YoloXBaseExp


class Exp(YoloXBaseExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- model config ---------------- #
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.25
        self.act = 'silu'
        self.gamma_range = [7.0, 10.5]
        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 16
        self.input_size = (1280, 1280)  # (height, width)
        self.multiscale_range = 5       # actual multiscale ranges: [1280-5*64, 1280+5*64]
        # 2023-04-19 Modified by Huawei, specify data path
        self.data_dir = '/home/dataset'
        self.train_ann = '00Train.json'
        self.val_ann = '01Valid.json'
        self.train_ims = self.val_ims = 'raws_debayer_awb_fp32_1280x1280'
        # ---------------- transform config ---------------- #
        self.enable_mixup = False
        self.mosaic_prob = 0.5
        self.mosaic_scale = (0.5, 1.5)

        # self.hsv_prob = 1.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.5
        self.degrees = 10.0
        self.translate = 0.1
        self.shear = 2.0
        # ---------------- training config ---------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200 #500
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        self.no_aug_epochs = 15
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 50    # iter
        self.eval_interval = 10     # epoch
        self.output_dir = os.path.dirname(__file__)
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split('.')[0]

        # ---------------- testing config ---------------- #
        self.test_size = (1280, 1280)
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]

            # use depthwise = True
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, strides=[16, 32, 64], in_channels=in_channels, act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head, nf=16, gamma_range=self.gamma_range)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 64) - self.multiscale_range
                max_size = int(self.input_size[0] / 64) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(64 * size), 64 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import COCORawDataset, TrainTransformRaw, YoloBatchSampler
        from yolox.data import DataLoader, InfiniteSampler, MosaicDetectionRaw, worker_init_reset_seed
        from yolox.utils import wait_for_the_master, get_local_rank

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):
            dataset = COCORawDataset(data_dir=self.data_dir, json_file=self.train_ann, name=self.train_ims, img_size=self.input_size,
                preproc=TrainTransformRaw(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob), cache=cache_img)

        dataset = MosaicDetectionRaw(dataset, mosaic=not no_aug, img_size=self.input_size,
            preproc=TrainTransformRaw(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
            degrees=self.degrees, translate=self.translate, mosaic_scale=self.mosaic_scale, mixup_scale=self.mixup_scale,
            shear=self.shear, enable_mixup=self.enable_mixup, mosaic_prob=self.mosaic_prob, mixup_prob=self.mixup_prob)

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCORawDataset, ValTransformRaw

        valdataset = COCORawDataset(data_dir=self.data_dir, json_file=self.val_ann if not testdev else self.test_ann,
            name=self.val_ims if not testdev else self.test_ims, img_size=self.test_size, preproc=ValTransformRaw(legacy=legacy))
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler}
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        return val_loader


if __name__ == '__main__':
    e = Exp()
    d  = e.get_data_loader(batch_size=4, is_distributed=False, no_aug=False, cache_img=False)
    # d = e.get_eval_loader(batch_size=4, is_distributed=False)

    for (iteration, a) in enumerate(d):
        print(len(a))
        for item in a[:2]:
            print(type(item), torch.max(item), item.shape)
