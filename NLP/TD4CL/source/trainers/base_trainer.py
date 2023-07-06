# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import logging
import os
import sys
import torch
from transformers import AdamW, get_scheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import math

logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, config, processor, datasets=None, raw_data=None):
        self.config = config
        logger.info("******* Running with the following arguments *********")
        for a in sys.argv[1:]:
            logger.info(a)
        logger.info("******************************************************")

        if not os.path.exists(self.config['model_dir']):
            os.makedirs(self.config['model_dir'])

        self.processor = processor
        self.datasets = datasets
        self.raw_data = raw_data
        self.best_epoch = 0
        self.best_step = 0
        self.current_best = -1
        self.patience = 0
        self.total_steps = None
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.current_epoch = 0

        self.device = torch.device("cuda:{}".format(config['device']) if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.loaders = self.init_loaders()

    def train_one_epoch(self, *args, **kwargs):
        """ Basic code for training 1 epoch """
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        """ Basic code for running inference """
        raise NotImplementedError

    def is_best(self, *args, **kwargs):
        """ Checking which metric to use for selecting a checkpoint """
        raise NotImplementedError

    def load_model(self, *args, **kwargs):
        """ Code for loading the model """
        raise NotImplementedError

    def write_predictions(self, *args, **kwargs):
        """ Write predictions to file """
        raise NotImplementedError

    def run_plain(self):
        """
        Classic training, no curriculum.
        """
        if self.config['mode'] != 'eval':

            if self.config['total_steps']:
                self.total_steps = self.config['total_steps']
            else:
                self.total_steps = len(self.loaders['train']) * \
                                   self.config['num_training_epochs'] // self.config['gradient_accumulation_steps']

            self.optimizer = self.init_optimizer()
            self.scheduler = self.init_scheduler()

        if self.config['evals_per_epoch']:
            minibatches = len(self.loaders['train'])
            self.config['log_interval'] = min(int(minibatches / self.config['evals_per_epoch']), 500)

        logger.info(f'====== Start Training ======')
        logger.info(" Num examples = %d", len(self.datasets['train']))
        logger.info(" Num Epochs = %d", self.config['num_training_epochs'])
        logger.info(" Train batch size = %d", self.config['batch_size'])
        logger.info(" Total optimization steps = %d", self.total_steps)
        logger.info(" Warmup steps = %d", math.floor(self.total_steps * self.config['warmup']))
        logger.info(" Gradient accumulation steps = %d", self.config['gradient_accumulation_steps'])
        logger.info(" Learning rate = {}".format(self.config['lr']))
        logger.info(" Weight decay = {}".format(self.config['weight_decay']))
        logger.info(" Gradient clip = {}".format(self.config['grad_clip']))
        logger.info(" Log interval = {}".format(self.config['log_interval']))

        iters = len(self.loaders['train']) // self.config['gradient_accumulation_steps']
        for _ in range(0, self.config['num_training_epochs']):
            self.current_epoch += 1
            self.train_one_epoch(self.loaders['train'], iters)

            if self.config['save_epochs'] or self.config['save_steps_epochs']:
                self.evaluate(track=True)
            else:
                self.evaluate()

            print()
        self.time2stop()

    def init_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.config['weight_decay']},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['eps'])
        return optimizer

    def init_scheduler(self):
        warmup_steps = math.floor(self.config['warmup'] * self.total_steps)

        scheduler = get_scheduler(
            name=self.config['lr_scheduler_type'],
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        return scheduler

    def init_loaders(self):
        loaders = {}
        for k in self.datasets.keys():
            shuffling = 'train' in k
            logger.info(f'{k:<20} ({len(self.datasets[k])}) -> Shuffling = {shuffling}')

            data_sampler = RandomSampler(self.datasets[k]) if shuffling else SequentialSampler(self.datasets[k])
            loaders[k] = DataLoader(
                dataset=self.datasets[k],
                batch_size=self.config['batch_size'] if 'train' in k else self.config['eval_batch_size'],
                sampler=data_sampler,
                collate_fn=self.processor.collate_fn
            )
        return loaders

    def save_model(self):
        self.model.save_pretrained(self.config['model_dir'])
        logger.info(f"Saved model checkpoint to {self.config['model_dir']}")

    def time2stop(self):
        logger.info(f'====== Training ended ======')
        logger.info(f'*** Best Epoch = {self.best_epoch} ***')
        logger.info(f'*** Best Step = {self.best_step} ***')
        logger.info(f'*** Best score = {self.current_best} ***')
        exit(0)

    def track_best_model(self, results):
        if self.is_best(results):
            self.best_step = self.global_step
            self.best_epoch = self.current_epoch
            self.patience = 0
            self.save_model()
        else:
            self.patience += 1

        if self.config['early_stop'] and (self.patience == self.config['max_patience']):
            logger.info('*** Stopping early! Ran out of patience ... *** ')
            self.time2stop()

    def evaluate(self, track=False):
        for split_n in self.loaders.keys():
            if split_n == 'val':
                self.eval(track=track, eval_split='val')
            else:
                if split_n != 'train':
                    self.eval(track=False, eval_split=split_n)
