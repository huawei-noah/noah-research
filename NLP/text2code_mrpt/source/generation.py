# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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

import sys
import os
import re
from typing import Optional
from dataclasses import dataclass, field
from utils import write_jsonl, read_problems
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import transformers
from transformers import HfArgumentParser, set_seed, AutoTokenizer
import logging
from pangu_alpha import (
    PanguAlphaConfig,
    PanguAlphaModel,
    PanguAlphaTokenizer
)
from gpt_neo import GPTNeoForCausalLM
import pickle
import copy
import json


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


model2tokenizer = {'pycodegpt': AutoTokenizer, 'pangu': PanguAlphaTokenizer}
model2model = {'pycodegpt': GPTNeoForCausalLM, 'pangu': PanguAlphaModel}


@dataclass
class Arguments:
    local_rank: str = field(default=0)
    model_name_or_path: str = field(default=None)
    prefix_lm: bool = field(default=None)
    max_seq_length: int = field(default=None, metadata={"help": "Max sequence length"})
    max_new_tokens: int = field(default=None, metadata={"help": "Max new tokens, excluding prompt"})
    stop_token: str = field(default="<eot>", metadata={"help": "Stopping token"})
    output_dir: str = field(default="", metadata={"help": "Output directory"})
    dataset_file: str = field(default=None, metadata={"help": "dataset file"})
    temperature: float = field(default=1.0, metadata={"help": ""})
    greedy: bool = field(default=False, metadata={"help": "Do greedy decoding"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "primarily useful for CTRL model; in that case, use 1.2"})
    k: int = field(default=0, metadata={"help": ""})
    p: float = field(default=0.9, metadata={"help": ""})
    batch_size: int = field(default=32, metadata={"help": "Batch size"})
    seed: int = field(default=1234, metadata={"help": "Seed"})
    no_cuda: bool = field(default=False, metadata={"help": ""})
    num_return_sequences: int = field(default=1, metadata={"help": ""})
    mlp_samples: int = field(default=1, metadata={"help": ""})
    torch_dtype: str = field(default="fp32", metadata={"help": "What dtype to load the model with"})
    replicated_tokens_map: bool = field(default=False, metadata={"help": "Allow replicated embeddings"})
    data_path: str = field(default=None, metadata={'help': "data_path"})
    model_type: str = field(default='pangu', metadata={"help": "Model type"})
    incremental: bool = field(default=False, metadata={'help': "Use incremental generations"})
    show_examples: bool = field(default=False, metadata={'help': "Show example generation"})

class PycodegptDataset(Dataset):
    def __init__(self, problems, args=None, tokenizer=None):
        self.problems = []

        pbar = tqdm(problems)
        for task_id in pbar:
            pbar.set_description(f'Processing {task_id}')

            docstring = problems[task_id]["prompt"]
            signature = problems[task_id]["signature"]

            if "\n" in docstring:
                doc = docstring.split('\n')
                new_doc = []
                for d in doc:
                    new_doc.append(f"    {d}")
                new_docstring = '\n'.join(new_doc)
                docstring_ids = tokenizer.encode('\n    \"\"\"\n' + new_docstring + '\n    \"\"\"', add_special_tokens=False)
            else:
                docstring_ids = tokenizer.encode('\n    \"\"\" ' + docstring + ' \"\"\"', add_special_tokens=False)

            code_ids = tokenizer.encode(signature, add_special_tokens=False)
            if args.replicated_tokens_map:
                new_code_ids = []
                for cid in code_ids:
                    if cid in args.replicated_tokens_map:
                        new_code_ids.append(args.replicated_tokens_map[cid])
                    else:
                        new_code_ids.append(cid)
                code_ids = copy.deepcopy(new_code_ids)

            if problems[task_id]['completion'] != '':  # for incremental
                completion_ids = tokenizer.encode(problems[task_id]['completion'], add_special_tokens=False)
                if args.replicated_tokens_map:
                    new_complition_ids = []
                    for cid in completion_ids:
                        if cid in args.replicated_tokens_map:
                            new_complition_ids.append(args.replicated_tokens_map[cid])
                        else:
                            new_complition_ids.append(cid)
                    completion_ids = copy.deepcopy(new_complition_ids)

                comp_len = len(completion_ids)
                encoded_prompt = [tokenizer.convert_tokens_to_ids('<|beginoftext|>')] + \
                                 code_ids + \
                                 [tokenizer.convert_tokens_to_ids('<comments>')] + \
                                 docstring_ids + \
                                 [tokenizer.convert_tokens_to_ids('<python>')] + \
                                 completion_ids

            else:
                comp_len = 0
                encoded_prompt = [tokenizer.convert_tokens_to_ids('<|beginoftext|>')] + \
                                 code_ids + \
                                 [tokenizer.convert_tokens_to_ids('<comments>')] + \
                                 docstring_ids + \
                                 [tokenizer.convert_tokens_to_ids('<python>')]

            things_to_return = {
                'task_id': task_id, #.replace('HumanEval', 'Python'),
                'prompt_length': len(encoded_prompt) - comp_len,
                'encoded_prompt': encoded_prompt,
                'attn_masks': [1] * len(encoded_prompt),
                'original_prompt': problems[task_id]['signature']
            }

            # Prefix-LM option
            if args.prefix_lm:
                things_to_return.update({'prefix_lm_mask': len(encoded_prompt) - 1})

            self.problems.append(things_to_return)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, item):
        return self.problems[item]


class PanguDataset(Dataset):
    def __init__(self, problems, args=None, tokenizer=None):
        self.problems = []

        pbar = tqdm(problems)
        for task_id in pbar:
            pbar.set_description(f'Processing {task_id}')

            comments = problems[task_id]["prompt"]
            code = problems[task_id]["signature"]

            encoded_comments = [tokenizer.convert_tokens_to_ids('<comments>')] + \
                                tokenizer.encode(comments, add_special_tokens=False)

            if problems[task_id]['completion'] != '':  # for incremental
                completion_ids = tokenizer.encode(problems[task_id]['completion'], add_special_tokens=False)
                comp_len = len(completion_ids)
                code_ids = tokenizer.encode(code, add_special_tokens=False) + completion_ids 
            else:
                comp_len = 0
                code_ids = tokenizer.encode(code, add_special_tokens=False)

            if args.replicated_tokens_map:
                new_code_ids = []
                for cid in code_ids:
                    if cid in args.replicated_tokens_map:
                        new_code_ids.append(args.replicated_tokens_map[cid])
                    else:
                        new_code_ids.append(cid)
                code_ids = copy.deepcopy(new_code_ids)

            encoded_code = [tokenizer.convert_tokens_to_ids('<python>')] + code_ids

            encoded_prompt = encoded_comments + encoded_code
            if args.max_seq_length is not None:
                encoded_prompt = encoded_prompt[:args.max_seq_length]

            things_to_return = {
                'task_id': task_id, # .replace('HumanEval', 'Python'),
                'code': code,
                'prompt_length': len(encoded_prompt) - comp_len,
                'encoded_prompt': encoded_prompt,
                'attn_masks': [1] * len(encoded_prompt),
                'original_prompt': problems[task_id]['signature']
            }

            # Prefix-LM option
            if args.prefix_lm:
                things_to_return.update({'prefix_lm_mask': len(encoded_comments)})

            self.problems.append(things_to_return)

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, item):
        return self.problems[item]


class MyCollate:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer

    def __call__(self, batch):
        task_id_batch = [item['task_id'] for item in batch]
        orig_prompt_batch = [item['original_prompt'] for item in batch]

        max_len = max([len(item['encoded_prompt']) for item in batch])

        for item in batch:
            pad_rest = max_len - len(item['encoded_prompt'])

            item['attn_masks'] = ([0] * pad_rest) + item['attn_masks']
            item['encoded_prompt'] = ([self.tokenizer.convert_tokens_to_ids('<pad>')] * pad_rest) + item['encoded_prompt']

            item['prompt_length'] += pad_rest

        attnmasks_batch = torch.tensor([item['attn_masks'] for item in batch], dtype=torch.long)
        problem_batch = torch.tensor([item['encoded_prompt'] for item in batch], dtype=torch.long)
        prompt_length_batch = [item['prompt_length'] for item in batch]

        if 'prefix_lm_mask' in batch[0]:
            prefix_batch = torch.tensor([item['prefix_lm_mask'] for item in batch], dtype=torch.long)
        else:
            prefix_batch = None
        return task_id_batch, prompt_length_batch, problem_batch, attnmasks_batch, prefix_batch, orig_prompt_batch


def post_process_generated_tokens(generated_tokens, indent_spaces=4):
    generated_tokens = ''.join(generated_tokens).replace('\u2581', ' ').replace('[_DUP_]', '')
    generated_tokens = generated_tokens.split("<NEW_LINE>")

    indent = " " * indent_spaces
    num_indents = 0
    generated_tokens_tmp = []
    for line in generated_tokens:
        if "<INDENT>" in line:
            num_indents += 1
        num_indents -= len(re.findall(r"<DEDENT>", line))
        line = line.replace("<INDENT>", "").replace("<DEDENT>", "")
        line = indent * num_indents + line.strip()
        generated_tokens_tmp.append(line)

    generated_tokens = "\n".join(generated_tokens_tmp)
    return generated_tokens


def main():
    transformers.utils.logging.set_verbosity_info()
    logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
    args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]

    if args.replicated_tokens_map:
        with open(os.path.join(args.data_path, 'replicated_tokens_map.pkl'), 'rb') as f:
            args.replicated_tokens_map = pickle.load(f)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)

    #############################
    # LOAD MODEL & TOKENIZER
    #############################
    tokenizer = model2tokenizer[args.model_type].from_pretrained(
        args.model_name_or_path,
        local_files_only=True,
        use_fast=True
    )

    ##############################
    # DeepSpeed things
    ##############################
    if args.torch_dtype == 'fp16':
        logger.info("========== Using FP16 to load the model ==========")
        dtype = torch.float16
    elif args.torch_dtype == 'fp32':
        logger.info("========== Using FP32 to load the model ==========")
        dtype = torch.float32
    else:
        logger.info("========== Using AUTO to load the model ==========")
        dtype = "auto"

    model = model2model[args.model_type].from_pretrained(
        args.model_name_or_path,
        args=args,
        torch_dtype=dtype,
        tokenizer=tokenizer
    )
    model.to('cuda')

    ##########################
    # Example Generation
    ##########################
    if args.show_examples:
        if 'pycodegpt' in args.model_name_or_path:
            example_generation_pycodegpt(tokenizer=tokenizer, args=args, model=model)
        else:
            example_generation_pangu(tokenizer=tokenizer, args=args, model=model)

    ###################################
    # Actually begin Generation
    ###################################
    logger.info(f'Loading dataset: {args.dataset_file}')
    problems = read_problems(args.dataset_file, infer_incremental_completions=args.incremental)

    if 'pycodegpt' in args.model_name_or_path:
        dataset = PycodegptDataset(
            problems,
            tokenizer=tokenizer,
            args=args
        )
    else:
        dataset = PanguDataset(
            problems,
            tokenizer=tokenizer,
            args=args
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=MyCollate(args=args, tokenizer=tokenizer),
        shuffle=False
    )
    generated_sequences = []

    model.eval()

    for sample_no in tqdm(range(args.num_return_sequences), leave=False, desc='Generating samples'):
        for task_ids, prompt_lengths, batch, attn_masks, prefix_idx, orig_prompts in \
                tqdm(dataloader, leave=False, desc=f'For sample #{sample_no} / {args.num_return_sequences}'):

            if not args.no_cuda:
                batch = batch.to('cuda')
                attn_masks = attn_masks.to('cuda')

                if args.prefix_lm:
                    prefix_idx = prefix_idx.to('cuda')

            pad_token_id = tokenizer.convert_tokens_to_ids('<pad>')

            with torch.no_grad():
                output_sequences = model.generate(
                    input_ids=batch,
                    max_length=args.max_seq_length if args.max_seq_length else None,
                    max_new_tokens=args.max_new_tokens if args.max_new_tokens else None,
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=not args.greedy,
                    num_return_sequences=args.mlp_samples,
                    attention_mask=attn_masks,
                    prefix_lm_mask=prefix_idx if args.prefix_lm else None,
                    pad_token_id=pad_token_id,
                    eos_token_id=tokenizer.convert_tokens_to_ids('<eot>'),
                    sample_replacement=True
                )

            for task_id, prompt_length, generated_sequence, orig_prompt in \
                    zip(task_ids, prompt_lengths, output_sequences, orig_prompts):

                generated_sequence = generated_sequence[prompt_length:]

                # Decode text
                answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_sequence))

                # Remove all text after the stop token
                answer = answer[: answer.find(args.stop_token) if args.stop_token else None]

                # post-process
                answer = post_process_generated_tokens(answer)
                
                generated_sequences.append(dict(task_id=task_id, generation=answer, prompt=orig_prompt))

        write_jsonl(
            os.path.join(
                args.output_dir,
                f"samples={args.num_return_sequences}_{args.torch_dtype}_bs={args.batch_size}_t={args.temperature}_k={args.k}_p={args.p}.jsonl"
            ),
            generated_sequences
        )

    return generated_sequences


def example_generation_pangu(tokenizer=None, model=None, args=None):
    logger.info('--- In example generation of PanGu ---')

    fixed_prompt = "Check if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> " \
                   "has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n"
    signature = "def has_close_elements(numbers: List[float], threshold: float):"

    model.eval()

    stop_token = '<eot>'

    encoded_comments = [tokenizer.convert_tokens_to_ids('<comments>')] + \
                        tokenizer.encode(fixed_prompt, add_special_tokens=False)

    code_ids = tokenizer.encode(signature, add_special_tokens=False)
    if args.replicated_tokens_map:
        new_code_ids = []
        for cid in code_ids:
            if cid in args.replicated_tokens_map:
                new_code_ids.append(args.replicated_tokens_map[cid])
            else:
                new_code_ids.append(cid)
        code_ids = copy.deepcopy(new_code_ids)
    encoded_code = [tokenizer.convert_tokens_to_ids('<python>')] + code_ids

    encoded_prompt = encoded_comments + encoded_code

    if args.prefix_lm:
        prefix_lm_mask = len(encoded_comments)
    else:
        prefix_lm_mask = None

    encoded_prompt = torch.LongTensor([encoded_prompt]).to('cuda')

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1024,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=False,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True,
        prefix_lm_mask=torch.LongTensor([prefix_lm_mask]).to('cuda') if prefix_lm_mask is not None else None,
        eos_token_id=tokenizer.convert_tokens_to_ids('<eot>'),
        sample_replacement=True
    )
    generated_sequence = output_sequences.sequences[0]
    generated_sequence = generated_sequence.tolist()
    generated_sequence = generated_sequence[input_ids.size(-1):]

    # Unified tokenizer is ok here, we just need ids to tokens
    text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_sequence))
    text = text[: text.find(stop_token) if stop_token else None]
    text = post_process_generated_tokens(text)
    text = fixed_prompt + signature + text

    logger.info('========== EXAMPLE GENERATION ==========')
    print(text)
    logger.info('========================================')


def example_generation_pycodegpt(tokenizer=None, model=None, args=None):
    logger.info('--- In example generation of PYCODEGPT ---')

    signature = "def has_close_elements(numbers: List[float], threshold: float) -> bool:"
    docstring = "\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each " \
                "other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    " \
                "False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\""

    model.eval()

    stop_token = '<eot>'

    code_ids = tokenizer.encode(signature, add_special_tokens=False)
    if args.replicated_tokens_map:
        new_code_ids = []
        for cid in code_ids:
            if cid in args.replicated_tokens_map:
                new_code_ids.append(args.replicated_tokens_map[cid])
            else:
                new_code_ids.append(cid)
        code_ids = copy.deepcopy(new_code_ids)

    encoded_prompt = [tokenizer.convert_tokens_to_ids('<|beginoftext|>')] + \
                     code_ids + \
                     [tokenizer.convert_tokens_to_ids('<comments>')] + \
                      tokenizer.encode(docstring, add_special_tokens=False)  + \
                     [tokenizer.convert_tokens_to_ids('<python>')]

    if args.prefix_lm:
        prefix_lm_mask = len(encoded_prompt) - 1
    else:
        prefix_lm_mask = None

    encoded_prompt = torch.LongTensor([encoded_prompt]).to('cuda')

    if encoded_prompt.size()[-1] == 0:
        input_ids = None
    else:
        input_ids = encoded_prompt

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1024,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        repetition_penalty=1.0,
        do_sample=False,
        num_return_sequences=1,
        output_scores=True,
        return_dict_in_generate=True,
        prefix_lm_mask=torch.LongTensor([prefix_lm_mask]).to('cuda') if prefix_lm_mask is not None else None,
        eos_token_id=tokenizer.convert_tokens_to_ids('<eot>'),
        sample_replacement=True
    )
    generated_sequence = output_sequences.sequences[0]
    generated_sequence = generated_sequence.tolist()
    generated_sequence = generated_sequence[input_ids.size(-1):]

    # Unified tokenizer is ok here, we just need ids to tokens
    text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(generated_sequence))
    text = text[: text.find(stop_token) if stop_token else None]
    text = post_process_generated_tokens(text)
    text = signature + docstring + text

    logger.info('========== EXAMPLE GENERATION ==========')
    print(text)
    logger.info('========================================')


if __name__ == "__main__":
    generated_sequences = main()