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

import transformers.utils.logging
from transformers import AutoTokenizer
from datasets import set_caching_enabled
from trainers.curric_trainer import get_curriculum_trainer
from trainers.mchoice_trainer import MultiChoiceTrainer as Trainer
from utils import show_random_examples, setup_logger, init_seed
from data_processors import *
from data_files import *
import argparse
import logging
import sys
import math

datasets.set_caching_enabled(False)
logger = logging.getLogger(__name__)
datasets.utils.logging.set_verbosity_error()
transformers.utils.logging.set_verbosity_error()


def main(config):
    config = setup_logger(config)
    _ = init_seed(config['seed'])

    ########################
    # Data
    ########################
    dataset_name = config['dataset_name']
    data_files = DataFiles(config)

    # load data
    dataset = data_files.filename(dataset_name)
    logger.info(dataset)

    ########################
    # Setup Model
    ########################
    logger.info('*** Using the {} model ***'.format(config['model_name'].upper()))
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name']
    )

    processor = MultipleChoiceProcessor(config, tokenizer)
    processed_data = dataset.map(processor.convert_to_features, batched=True)
    logger.info(processed_data)

    if 'train' in config['mode']:
        show_random_examples(dataset['train'], num_examples=10, rand=True)

    ########################
    # Set up Trainer
    ########################
    if ('competence-bias' in config['curriculum']) or ('competence-most' in config['curriculum']):
        _, sample, start, end, total = config['curriculum'].split('-')
        config['curric_start'] = float(start)
        config['curric_steps'] = math.ceil(float(end) * int(total))

    elif 'competence' in config['curriculum']:
        _, start, end, total = config['curriculum'].split('-')
        config['curric_start'] = float(start)
        config['curric_steps'] = math.ceil(float(end) * int(total))

    curric_trainer = get_curriculum_trainer(Trainer)
    trainer = curric_trainer(config, processor, datasets=processed_data)

    ################################
    # Train/Eval Mode + Curriculum
    ################################
    if 'train' in args.mode:
        if config['curriculum'] == 'none':
            trainer.run_plain()

        elif config['curriculum'] in ['baby-step', 'one-pass']:
            trainer.run_sharding(sample='none', selection=args.selection)

        elif 'annealing' in config['curriculum']:
            if len(config['curriculum'].split('-')) == 1:
                smpl = 'none'
            else:
                smpl = config['curriculum'].split('-')[1]
            trainer.run_sharding(sample=smpl, selection=args.selection)

        elif 'competence' in config['curriculum']:
            if len(config['curriculum'].split('-')) == 5:
                smpl = config['curriculum'].split('-')[1]
            else:
                smpl = 'none'
            trainer.run_competence(sample=smpl, selection=args.selection)

        else:
            print(config['curriculum'])
            logger.info(' Invalid curriculum.')
            exit(0)

    elif args.mode == 'eval':
        trainer.device = torch.device("cuda:{}".format(config['device']) if torch.cuda.is_available() else "cpu")
        logger.info(f'Evaluating on {config["dataset_name"]}')

        res = {}
        for data_split in trainer.loaders.keys():
            if data_split != 'train':
                res[data_split] = trainer.eval(data_split, track=False, return_dict=True, write_preds=True)
        print(res)

    else:
        print(args.mode)
        logger.info(' Invalid training/testing mode.')
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--glue_dir', type=str, default='../glue_data')
    parser.add_argument('--all_data_dir', type=str, default='../data')
    parser.add_argument('--dynamics_dir', type=str, default='../dynamics')
    parser.add_argument('--task', type=str, default='multi_choice')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--show_examples', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=335458169, help='Seed. If -1 initialise with a random one.')
    parser.add_argument('--model_name', type=str, help='Model name from HF Transformers')
    parser.add_argument('--mode', required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Saving directory')
    parser.add_argument('--model_dir', type=str, required=True, help='Directory to save model')
    parser.add_argument('--pretrained_model_dir', type=str, help='Folder with a pretrained model')
    parser.add_argument('--num_training_epochs', type=int, required=True, help='Training in epochs')
    parser.add_argument('--device', type=int, default=0, help='Specify GPU number, please')
    parser.add_argument('--max_seq_length', type=int, required=True, help='Max sequence length allowed')
    parser.add_argument('--warmup', type=float, default=0.0, help='Warmup percentage')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--eval_batch_size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Regularisation')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient Clipping')
    parser.add_argument('--gradient_accumulation_steps', required=True, type=int, default=1)
    parser.add_argument('--max_patience', type=int, default=25, help='Patience (STEPS) before stopping')
    parser.add_argument('--early_stop', action='store_true', help='Do early stopping or not')
    parser.add_argument('--save_steps', action='store_true')
    parser.add_argument('--save_epochs', action='store_true')
    parser.add_argument('--save_steps_epochs', action='store_true')
    parser.add_argument('--log_interval', type=int, help='Evaluate per # batches')
    parser.add_argument('--n_per_epoch', type=int, help='Evaluate number of times per epoch')
    parser.add_argument('--evals_per_epoch', type=int, help='Evaluate number of times per epoch')
    parser.add_argument('--lr_scheduler_type', default="linear", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant",
                                 "constant_with_warmup"])
    parser.add_argument('--curriculum', type=str, help='Type of curriculum to use')
    parser.add_argument('--metric1', type=str, help='Type of curriculum to use')
    parser.add_argument('--metric2', type=str, help='Type of curriculum to use as 2nd metric')
    parser.add_argument('--curric_steps', type=int, help='Number of curriculum steps')
    parser.add_argument('--log_dynamics', action='store_true', help='Log dynamics during training')
    parser.add_argument('--use_dynamics', type=str, help='Model directory to use dynamics from')
    parser.add_argument('--selection', type=float, help='Use data section')
    parser.add_argument('--steps', type=int, help='Train for a certain number of steps only')
    parser.add_argument('--total_steps', type=int, help='Train only for this number of steps')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()

    main(vars(args))
