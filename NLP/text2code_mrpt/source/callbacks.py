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

import torch
from transformers import TrainerCallback
from pangu_alpha import PanguAlphaTokenizer
import re
import os
import copy


class ExampleInput(TrainerCallback):
    """
    A callback that prints some info about the first example fed to the model
    """
    def on_train_begin(self, args, state, control, tokenizer=None, train_dataloader=None, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        for b, batch in enumerate(train_dataloader):

            if args.mlm_docstring:
                if state.is_local_process_zero:
                    tmp_inputs_mlm = [tokenizer.convert_ids_to_tokens(t.item()) for t in batch['mlm']['input_ids'][-1]]
                    tmp_inputs_clm = [tokenizer.convert_ids_to_tokens(t.item()) for t in batch['clm']['input_ids'][-1]]
                    tmp_labels_mlm = [tokenizer.convert_ids_to_tokens(t.item()) if t.item() != -100 else '<ign>' for t in
                                      batch['mlm']['labels'][-1]]
                    tmp_labels_clm = [tokenizer.convert_ids_to_tokens(t.item()) if t.item() != -100 else '<ign>' for t in
                                      batch['clm']['labels'][-1]]

                    print("--- MLM ---")
                    for ii, jj, kk in zip(tmp_inputs_mlm, tmp_labels_mlm, batch['mlm']['attention_mask'][-1]):
                        print(repr(ii), repr(jj), kk.item())

                    print("--- CLM ---")
                    for ii, jj, kk in zip(tmp_inputs_clm, tmp_labels_clm, batch['clm']['attention_mask'][-1]):
                        print(repr(ii), repr(jj), kk.item())

            else:
                if state.is_local_process_zero:
                    tmp_inputs = [tokenizer.convert_ids_to_tokens(t.item()) for t in batch['input_ids'][-1]]
                    tmp_labels = [tokenizer.convert_ids_to_tokens(t.item()) if t.item() != -100 else '<ign>' for t in
                                  batch['labels'][-1]]

                    for ii, jj, kk in zip(tmp_inputs, tmp_labels, batch['attention_mask'][-1]):
                        print(repr(ii), repr(jj), kk.item())

            if b == 2:
                break


class GenerationCallback(TrainerCallback):
    """
    A callback that does an example Generation
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @staticmethod
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

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        fixed_prompt = "Check if in given list of numbers, are any two numbers closer to each other than\ngiven threshold.\n>>> " \
                       "has_close_elements([1.0, 2.0, 3.0], 0.5)\nFalse\n>>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\nTrue\n"
        signature = "def has_close_elements(numbers: List[float], threshold: float):"

        model.eval()

        stop_token = '<eot>'
        encoded_comments = [self.tokenizer.convert_tokens_to_ids('<comments>')] + \
                           self.tokenizer.encode(fixed_prompt, add_special_tokens=False)

        code_ids = tokenizer.encode(signature, add_special_tokens=False)
        if args.replicated_tokens_map is not None:
            new_code_ids = []
            for cid in code_ids:
                if cid in args.replicated_tokens_map:
                    new_code_ids.append(args.replicated_tokens_map[cid])
                else:
                    new_code_ids.append(cid)
            code_ids = copy.deepcopy(new_code_ids)

        encoded_code = [self.tokenizer.convert_tokens_to_ids('<python>')] + code_ids

        if len(encoded_comments) + len(encoded_code) > (1024 - 500):
            encoded_comments = encoded_comments[:1024 - 500 - len(encoded_code)]

        encoded_prompt = encoded_comments + encoded_code

        encoded_prompt = torch.LongTensor([encoded_prompt]).to('cuda')

        if encoded_prompt.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_prompt

        if args.prefix_lm:
            prefix_lm_mask = len(encoded_comments)
        else:
            prefix_lm_mask = None

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=500 + len(encoded_prompt[0]),
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=False,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            prefix_lm_mask=torch.LongTensor([prefix_lm_mask]).to('cuda') if prefix_lm_mask is not None else None
        )
        generated_sequence = output_sequences.sequences[0]
        generated_sequence = generated_sequence.tolist()
        generated_sequence = generated_sequence[input_ids.size(-1):]

        # Unified tokenizer is ok here, we just need ids to tokens
        text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(generated_sequence))
        text = self.post_process_generated_tokens(text)

        text = text[: text.find(stop_token) if stop_token else None]
        text = fixed_prompt + signature + text

        if state.is_local_process_zero:
            if args.prefix_lm:
                print('Using prefix LM ...')
            print('========== EXAMPLE GENERATION ==========')
            print(text)
            print('========================================')
            if args.debugging:
                exit(0)


class GenerationCallbackRaw(TrainerCallback):
    """
    A callback that does an example Generation
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @staticmethod
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

    def on_evaluate(self, args, state, control, model=None, tokenizer=None, **kwargs):
        signature = "def has_close_elements(numbers: List[float], threshold: float) -> bool:"
        docstring = "\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each " \
                    "other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    " \
                    "False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\""

        model.eval()

        stop_token = '<eot>'

        code_ids = tokenizer.encode(signature, add_special_tokens=False)
        if args.replicated_tokens_map is not None:
            new_code_ids = []
            for cid in code_ids:
                if cid in args.replicated_tokens_map:
                    new_code_ids.append(args.replicated_tokens_map[cid])
                else:
                    new_code_ids.append(cid)
            code_ids = copy.deepcopy(new_code_ids)

        encoded_prompt = [self.tokenizer.convert_tokens_to_ids('<|beginoftext|>')] + \
                         code_ids + \
                         [self.tokenizer.convert_tokens_to_ids('<comments>')] + \
                         self.tokenizer.encode(docstring, add_special_tokens=False) + \
                         [self.tokenizer.convert_tokens_to_ids('<python>')]
        input_ids = torch.LongTensor([encoded_prompt]).to('cuda')

        if args.prefix_lm:
            prefix_lm_mask = len(encoded_prompt) - 1
        else:
            prefix_lm_mask = None

        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=500 + len(input_ids[0]),
            temperature=1.0,
            top_k=0,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=False,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
            synced_gpus=True,
            prefix_lm_mask=torch.LongTensor([prefix_lm_mask]).to('cuda') if prefix_lm_mask is not None else None
        )
        generated_sequence = output_sequences.sequences[0]
        generated_sequence = generated_sequence.tolist()
        generated_sequence = generated_sequence[input_ids.size(-1):]

        # Unified tokenizer is ok here, we just need ids to tokens
        text = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(generated_sequence))
        text = self.post_process_generated_tokens(text)

        text = text[: text.find(stop_token) if stop_token else None]
        text = signature + docstring + text

        if state.is_local_process_zero:
            if args.prefix_lm:
                print('Using prefix LM ...')
            print('========== EXAMPLE GENERATION ==========')
            print(text)
            print('========================================')
            if args.debugging:
                exit(0)
