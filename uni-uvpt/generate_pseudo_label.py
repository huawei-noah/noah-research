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
# Modified from: https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0


import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import numpy as np
from tqdm import tqdm
from PIL import Image

import mmseg_custom   


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', default='configs/Uni-UVPT/generate_pseudo_label_X_to_cityscapes_fcn8s_vgg16.py', help='test config file path')
    parser.add_argument('--checkpoint', default='model/pretrained_models/SSS_fcn8s_vgg16_Synthia_Cityscapes_31.40.pth', help='checkpoint file')
    parser.add_argument('--data-root', type=str, help='the dir to datasets')
    parser.add_argument('--pseudo_label_dir', default="/cache/cityscapes/pretrain/SSS_fcn8s_vgg16_Synthia_Cityscapes_31.40/train/", help='Path to save pseudo label')
    parser.add_argument('--pseudo_thre', type=float)
    parser.add_argument('--gpus', type=int, help='gpu numbers')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    # build the dataloader

    if args.data_root is not None:
        cfg.data.train["data_root"] = args.data_root
        cfg.data.test["data_root"] = args.data_root
        cfg.data.val["data_root"] = args.data_root
    cfg.data.test.img_dir = cfg.data.train.img_dir
    cfg.data.test.ann_dir = cfg.data.train.ann_dir
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.pseudo_thre is not None:
        model.inference_thre = args.pseudo_thre
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        map_location='cpu',
        revise_keys=[(r'^module\.', ''), ('model.', '')])
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    model.eval()
    cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    model = MMDataParallel(model, device_ids=[0])
    count = 0
    for i, data in tqdm(enumerate(data_loader)):
        with torch.no_grad():
            result = model(return_loss=False, **data)
        for j in range(data["img"][0].size()[0]):
            path, img_name = os.path.split(data["img_metas"][0]._data[0][j]["filename"])
            city_name = path.split("/")[-1]
            pseudo_label_name = img_name.replace('_leftImg8bit.png', '_gtFine_labelTrainIds.png')
            save_path = os.path.join(args.pseudo_label_dir, city_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Image.fromarray(result[j].astype(np.uint8), mode='L').save(os.path.join(save_path, pseudo_label_name))
            count += 1


if __name__ == '__main__':
    main()
