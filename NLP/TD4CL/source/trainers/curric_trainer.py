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

from samplers import *
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def get_curriculum_trainer(base):
    class CurriculumTrainer(base):
        """
        Trainer for Curriculum Learning in shards (Baby Step, One pass)
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def run_sharding(self, sample=False, selection=None):
            """
            Perform training on shards
            1. One-pass: Train on each shard individually
            2. Baby-step: Train on the combination of the already seen shards
            3. Annealing: Train on the current shared + 1/N of the already seen shards
            """
            logger.info(f"**** Using the {self.config['curriculum']} curriculum ! ****")

            # First, train the Curriculum for a fixed number of steps
            sampler = BatchShardingSampler(self.datasets['train'],
                                           metric1=self.config['metric1'],
                                           metric2=self.config['metric2'],
                                           curric=self.config['curriculum'],
                                           batch_size=self.config['batch_size'],
                                           sample=sample,
                                           selection=selection)
            loader = DataLoader(dataset=self.datasets['train'],
                                batch_sampler=sampler,
                                collate_fn=self.processor.collate_fn)

            curric_steps = len(loader) // self.config['gradient_accumulation_steps']
            epochs2steps = (len(self.loaders['train']) * self.config['num_training_epochs']) // self.config['gradient_accumulation_steps']
            remaining_steps = epochs2steps - curric_steps
            logger.info(f'Total steps: {curric_steps}')
            logger.info(f'Remaining steps: {remaining_steps}')

            if curric_steps > epochs2steps:
                self.total_steps = epochs2steps
            else:
                self.total_steps = curric_steps + remaining_steps

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

            self.train_one_epoch(loader, len(loader) // self.config['gradient_accumulation_steps'])  # train curriculum
            self.evaluate(track=True)

            # Continue training as normal for the remaining number of steps
            iters = len(self.loaders['train']) // self.config['gradient_accumulation_steps']
            for _ in range(0, self.config['num_training_epochs']):
                self.current_epoch += 1
                self.train_one_epoch(self.loaders['train'], iters)
                if self.config['save_epochs'] or self.config['save_steps_epochs']:
                    self.evaluate(track=True)
                else:
                    self.evaluate()
            self.time2stop()

        def run_competence(self, sample=False, selection=None):
            """
            Run a curriculum where the number of data processed at each step is defined by a
            pacing function.
            Based on Platanios et al.(2019) - https://www.aclweb.org/anthology/N19-1119.pdf
            Additional: Instead of sampling examples uniformly, sample based on model variability
            """
            logger.info(f"**** Using the {self.config['curriculum']} curriculum ! ****")

            # First, train the Curriculum for a fixed number of steps
            sampler = BatchPacingSampler(self.datasets['train'],
                                         self.config,
                                         metric1=self.config['metric1'],
                                         metric2=self.config['metric2'],
                                         batch_size=self.config['batch_size'],
                                         c0=self.config['curric_start'],
                                         total_steps=self.config['curric_steps'],
                                         sample=sample,
                                         selection=selection)
            loader = DataLoader(dataset=self.datasets['train'],
                                batch_sampler=sampler,
                                collate_fn=self.processor.collate_fn)

            curric_steps = math.ceil(len(loader) / self.config['gradient_accumulation_steps'])
            epochs2steps = math.ceil((len(self.loaders['train']) * self.config['num_training_epochs']) / self.config['gradient_accumulation_steps'])
            remaining_steps = epochs2steps - curric_steps
            logger.info(f'Total steps: {curric_steps}')
            logger.info(f'Remaining steps: {remaining_steps}')

            if curric_steps > epochs2steps:
                self.total_steps = epochs2steps
            else:
                self.total_steps = curric_steps + remaining_steps

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

            # train one epoch
            self.train_one_epoch(loader, math.ceil(len(loader) / self.config['gradient_accumulation_steps']))
            self.eval(track=True)

            # Continue training as normal for the remaining number of steps
            iters = math.ceil(len(self.loaders['train']) / self.config['gradient_accumulation_steps'])
            for epoch in range(0, self.config['num_training_epochs']):
                self.current_epoch += 1
                self.train_one_epoch(self.loaders['train'], iters)
                if self.config['save_epochs'] or self.config['save_steps_epochs']:
                    self.evaluate(track=True)
                else:
                    self.evaluate()
            self.time2stop()

    return CurriculumTrainer
