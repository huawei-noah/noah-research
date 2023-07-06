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

from utils import *
from tqdm import tqdm
from time import time
from trainers.base_trainer import BaseTrainer
from models import ModifiedRobertaForMultipleChoice, ModifiedXLMRobertaForMultipleChoice
from transformers import AutoConfig

logger = logging.getLogger(__name__)


class MultiChoiceTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def print_results(self, results, mode):
        if 'train' in mode:
            print()
            logger.info(f"Epoch {self.current_epoch:<2}  | {mode:<12} | "
                        f"ACCURACY = {results['accuracy']:.2f} | "
                        f"Loss = {results['loss']:.5f} | "
                        f"{results['time']}")
        else:
            logger.info(f"step {self.global_step:<6} | {mode:<12} | "
                        f"ACCURACY = {results['accuracy']:.2f} | "
                        f"Loss = {results['loss']:.5f} | "
                        f"{results['time']}")

    def load_model(self):
        """
        Load Multiple Choice Model
        """
        if 'train' in self.config['mode']:
            model_path = self.config['model_name']
            logger.info(f"*** Loading Model from {model_path} ***")

            model_config = AutoConfig.from_pretrained(
                model_path
            )
            if self.config['model_name'].startswith('xlm-roberta-'):
                model = ModifiedXLMRobertaForMultipleChoice.from_pretrained(
                    model_path,
                    config=model_config
                )
            else:
                model = ModifiedRobertaForMultipleChoice.from_pretrained(
                    model_path,
                    config=model_config
                )

        else:
            model_path = self.config['model_dir']
            logger.info(f"*** Loading Model from {model_path} ***")

            if not os.path.exists(model_path):
                raise Exception(f"Non-existent Model path: {model_path}")
            else:
                if self.config['model_name'].startswith('xlm-roberta-'):
                    model = ModifiedXLMRobertaForMultipleChoice.from_pretrained(model_path)
                else:
                    model = ModifiedRobertaForMultipleChoice.from_pretrained(model_path)

        model.to(self.device)
        logger.info(f"*** Loaded model from {model_path} ***")
        return model

    def train_one_epoch(self, dataloader, iters, split_name='train'):
        """
        Main training loop for 1 epoch only and a classification task.

        Args:
            dataloader: Iterable with batches to process
            iters: The total number of iterations
            split_name: useful fo sharding only
        """
        results = {}
        info = []
        total_loss = 0
        predictions, ground_truth = [], []

        t0 = time()
        self.model.zero_grad()

        loop = tqdm(range(iters), leave=False, disable=True, mininterval=10, ncols=100)
        for step, batch in enumerate(dataloader):
            self.model.train()  # <--- Very important !!!

            inputs = {'input_ids': batch['input_ids'],
                      'attention_mask': batch['attention_mask'],
                      'labels': batch['labels']}
            inputs = {k: b.to(self.device) for k, b in inputs.items()}

            outputs = self.model(**inputs)
            loss = outputs.loss

            predictions += [outputs[1].argmax(dim=-1).detach().cpu().numpy()]
            ground_truth += [batch['labels']]

            loss = loss / self.config['gradient_accumulation_steps']
            loss.backward()
            loss_float = loss.item()
            total_loss += loss_float

            info += [{'logits': outputs[1].detach().cpu().numpy().tolist(),
                      'label': batch['labels'].detach().cpu().numpy().tolist(),
                      'id': batch['id']}]

            if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
                if self.config['grad_clip'] > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['grad_clip'])
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                self.global_step += 1

                loop.set_description(f'Epoch {self.current_epoch}, step {self.global_step}, '
                                     f'lr {self.scheduler.get_last_lr()[0]:.8f}, '
                                     f'loss {loss_float:.4f}')
                loop.update(1)

                if self.config['log_interval'] and (self.global_step % self.config['log_interval'] == 0) and (
                        self.global_step > 0):
                    if self.config['save_steps'] or self.config['save_steps_epochs']:
                        self.evaluate(track=True)
                    else:
                        self.evaluate()

                if not self.config['baseline']:
                    if self.global_step % self.total_steps == 0 and (self.global_step > 0):  # if total_steps completed, finish training
                        self.evaluate(track=True)
                        self.time2stop()

        t1 = time()
        results['time'] = humanized_time(t1 - t0)
        results['loss'] = total_loss / iters

        train_score = calculate_scores(self.config, predictions, ground_truth)
        results.update(train_score)
        self.print_results(results, split_name)

        # training dynamics
        if self.config['log_dynamics']:
            logger.info(f'Saving training dynamics for epoch {self.current_epoch}')
            out_file = os.path.join(self.config['model_dir'],
                                    self.config['dataset_name'] + f'_epoch{self.current_epoch}.json')
            self.log_training_dynamics(out_file, info)

    def eval(self, eval_split='val', track=False, return_dict=False, write_preds=False):
        """
        Evaluation for a classification task.

        Args:
            eval_split: Data split to evaluate one (e.g. 'val')
            track: Perform only evaluation or track learning (e.g. check for early stopping)
            write_preds: Write predictions into file
            return_dict: Return a dictionary with the results after evaluation

        Returns:
            Dictionary with results if specified
        """
        total_loss = 0
        results = {}
        predictions, ground_truth = [], []
        eval_info = []

        t0 = time()
        for batch in self.loaders[eval_split]:
            self.model.eval()

            with torch.no_grad():
                inputs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}

                if 'labels' in batch:
                    inputs.update({'labels': batch['labels']})

                inputs = {k: b.to(self.device) for k, b in inputs.items()}
                outputs = self.model(**inputs)

                if 'labels' in batch:
                    loss = outputs[0]
                    total_loss += loss.item()
                    ground_truth += [batch['labels'].detach().cpu().numpy()]
                    predictions += [outputs[1].argmax(dim=-1).detach().cpu().numpy()]

                    eval_info += [{'logits': outputs[1].detach().cpu().numpy().tolist(),
                                   'label': batch['labels'].detach().cpu().numpy().tolist(),
                                   'id': batch['id']}]
                else:
                    predictions += [outputs[0].argmax(dim=-1).detach().cpu().numpy()]
                    eval_info += [{'preds': predictions[-1].tolist()}]

        t1 = time()
        results['time'] = humanized_time(t1 - t0)
        results['loss'] = total_loss / len(self.loaders[eval_split])

        if 'labels' in next(iter(self.loaders[eval_split])):
            eval_score = calculate_scores(self.config, predictions, ground_truth)
            results.update(eval_score)
            self.print_results(results, eval_split)

        if (eval_split == 'val_m' or eval_split == 'val') and track:
            self.track_best_model(results)

        if write_preds:
            self.write_predictions(eval_info, eval_split)

        if return_dict:
            return results

    def is_best(self, results):
        if results['accuracy'] > self.current_best:
            self.current_best = results['accuracy']
            return True
        else:
            return False

    @staticmethod
    def log_training_dynamics(filef, info):
        """
        Save training dynamics to a .json file
        Each line contains a dictionary with keys: id, logits, preds, labels
        """
        info = group_dicts(info)

        with open(filef, 'w') as outfile:
            for i, id_ in enumerate(info['id']):  # for each item
                out_dict = {}
                for key in info.keys():  # id, logits, preds, labels
                    if type(info[key][i]) is np.ndarray:
                        out_dict[key] = np.squeeze(info[key][i]).tolist()
                    else:
                        out_dict[key] = info[key][i]

                outfile.write(json.dumps(out_dict) + '\n')

    def write_predictions(self, info, eval_split):
        """
        Write predictions
        """
        filepath = os.path.join(self.config['model_dir'], self.config['dataset_name'])
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        out_dir = os.path.join(filepath, f"{eval_split}.json")
        logger.info(' Writing predictions to {}'.format(out_dir))

        # inverse_label_map = {v: k for k, v in self.processor.label_map.items()}
        with open(out_dir, 'w') as outfile:
            for dict_ in info:
                for id_, p, l in zip(dict_['id'], dict_['logits'], dict_['label']):
                    tmpd = {'id': id_, 'pred': int(np.argmax(p)), 'label': l}
                    outfile.write(json.dumps(tmpd) + '\n')