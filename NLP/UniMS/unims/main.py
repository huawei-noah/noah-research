# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from os.path import join
import yaml
import logging
import datetime
import argparse
from pprint import pprint

from transformers import BartTokenizer

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import MMSum
from datamodule import MMSumDataModule

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Deal With "RuntimeError: unable to write to file" Error
import sys
import torch
from torch.utils.data import dataloader
# from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate
torch.multiprocessing.set_sharing_strategy("file_system")


def default_collate_override(batch):
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, "default_collate", default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


def set_datamodule(args, image_preprocess):
    # Set Tokenizer
    tokenizer_path = join(args["pretrained_model_path"], args["backbone"])
    tokenizer = BartTokenizer.from_pretrained(tokenizer_path)

    # Set DataModule
    datamodule = MMSumDataModule(
        args, tokenizer=tokenizer, image_preprocess=image_preprocess
    )

    return datamodule, tokenizer


def configure_training(args):
    seed_everything(args["seed"])

    # Set Logger
    log_name = f"{args['backbone']}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    logger.info(f'Save Log to Wandb Project {log_name}')
    wandb_logger = WandbLogger(
        project=args["project"],
        name=log_name,
        offline=True,
        save_dir=args["train_params"]["default_root_dir"],
    )

    # Set Callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_abs_RL',
        save_top_k=3,
        mode='max',
    )

    # Set Trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        **args["train_params"]
    )

    return trainer, log_name


def train(args):
    if args["stage_resume"]:
        model = MMSum.load_from_checkpoint(args["stage_resume_path"], args=args, strict=False)
    else:
        model = MMSum(args)

    datamodule, tokenizer = set_datamodule(args, model.image_preprocess)
    model.tokenizer = tokenizer

    trainer, _ = configure_training(args)
    trainer.fit(model, datamodule)


def test(args):
    logger.info(args['checkpoint_path'])
    state_dict = torch.load(args['checkpoint_path'])['state_dict']

    args['decode_path'] = join(args['decode_path'], datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    logger.info(f'Save Decode Results to {args["decode_path"]}')
    os.mkdir(args['decode_path'])

    model = MMSum(args=args)
    model.load_state_dict(state_dict, strict=False)

    datamodule, tokenizer = set_datamodule(args, model.image_preprocess)
    trainer, log_name = configure_training(args)
    model.tokenizer = tokenizer
    model.freeze()

    datamodule.setup('predict')
    trainer.predict(model=model, datamodule=datamodule, return_predictions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="running config path")
    args = parser.parse_args()

    args = yaml.load(open(args.config, "r", encoding="utf-8").read(), Loader=yaml.FullLoader)
    pprint(args)
    assert args["mode"] in ["train", "test"]
    if args["mode"] == "train":
        train(args)
    elif args["mode"] == "test":
        test(args)
