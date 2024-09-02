# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
# ============================================================================

from transformers import Trainer
from typing import Optional, Dict, List, Union, Tuple, Any
import logging
import math
import collections
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers.deepspeed import deepspeed_init
from transformers.trainer_utils import has_length
import copy
from optimization import get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)

from transformers.trainer_pt_utils import *


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.
        Args:
            num_training_steps (int): The number of training steps to do.
        """
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer if optimizer is None else optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            min_lr=self.args.min_learning_rate,
            init_lr=self.args.learning_rate
        )
        return self.lr_scheduler

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.
        Subclass and override this method to inject custom behavior.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        # Log steps as well
        logs["step"] = self.state.global_step
        output = {**logs, **{"step": self.state.global_step}}

        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
