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

import os
import argparse
import json
import sys
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
import datasets
import torch
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
sys.path.append('../')
import utils

pd.options.mode.chained_assignment = None


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_seq_class(pred_folder, model, tokenizer, name):
    # for each possible model
    model_dynamics = []

    logger.info(pred_folder)
    logger.info('Tokenizing dataset ...')
    dataset = datasets.load_from_disk(pred_folder)['train']
    print(dataset)
    device = torch.device('cuda')

    def score_sentence(model, tokenizer, sentences, mask_token_id=tokenizer.mask_token_id):
        tensor_input = tokenizer.encode(sentences, truncation=True, max_length=128, return_tensors='pt')

        repeat_input = tensor_input.repeat(tensor_input.size(-1) - 2, 1)
        mask = torch.ones(tensor_input.size(-1) - 1).diag(1)[:-2]
        masked_input = repeat_input.masked_fill(mask == 1, mask_token_id)
        # for m in masked_input:
        #     print(tokenizer.convert_ids_to_tokens(m))
        labels = repeat_input.masked_fill(masked_input != mask_token_id, -100).to(device)
        # for l in labels:
        #     print([tokenizer.convert_ids_to_tokens(ll.item()) if ll.item() != -100 else -100 for ll in l])
        # exit(0)

        # print(masked_input.size())
        batch = {'input_ids': masked_input.to(device),
                 'labels': labels.to(device)}
        outputs = model(**batch)
        result = math.exp(outputs.loss.item())
        return result

    model.to(device)
    model.eval()
    total_ppl = 0
    for item in tqdm(dataset):
        with torch.no_grad():
            if args.dataset_name in ['siqa', 'xcopa']:
                ex = {
                    'id': item['id'],
                    'ppl': score_sentence(model, tokenizer, item['premise']) + score_sentence(model, tokenizer, item['question']),
                }
            else:
                ex = {
                    'id': item['id'],
                    'ppl': (score_sentence(model, tokenizer, item['sentence1']) + score_sentence(model, tokenizer, item['sentence2']))
                    if 'sentence2' in item else score_sentence(model, tokenizer, [item['sentence1']])
                }
            model_dynamics.append(ex)

        total_ppl += ex['ppl']

    print('Total PPL', total_ppl / len(dataset))
    return model_dynamics


def main(args):
    logger.info('*** Loading {} tokenizer ***'.format(args.model_name.upper()))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name
    )

    # Load training statistics
    if args.dataset_name in ['pawsx', 'xnli', 'xcopa', 'mldoc', 'siqa', 'mnli', 'qnli', 'rte']:
        model_dynamics = load_seq_class(os.path.join(args.data_dir, args.dataset_name), model, tokenizer,
                                        name=args.dataset_name)
    else:
        model_dynamics = None

    model_dynamics = datasets.Dataset.from_pandas(pd.DataFrame(model_dynamics))
    print(model_dynamics)
    print(model_dynamics[0])

    final = {}
    for m in model_dynamics:
        final[m['id']] = m

    output_file = os.path.join(args.output_dir, f'{args.dataset_name}_{args.model_name}_ppl.json')

    logger.info(f' Writing dynamics statistics to {output_file}')
    with open(output_file, 'w') as outfile:
        json.dump(final, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str, default='../../dynamics/')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--analysis', action='store_true')
    args = parser.parse_args()

    utils.init_seed(42)
    main(args)
