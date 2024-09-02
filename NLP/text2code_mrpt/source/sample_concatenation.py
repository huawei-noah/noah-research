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

from transformers import HfArgumentParser, AutoTokenizer
from pangu_alpha import PanguAlphaTokenizer
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import datasets
import os
from tokenization import tokenization_function, tokenization_function_raw
import random
import pickle


@dataclass
class Arguments:
	dataset_dir: str = field(default=None, metadata={"help": "dataset directory"})
	max_seq_length: int = field(default=1024, metadata={"help": "maximum sequence length"})
	tokenizer: str = field(default=None, metadata={"help": "type of tokenizer to use"})
	model_name_or_path: str = field(default=None, metadata={"help": "model name or path"})
	separate_embeds: bool = field(default=False, metadata={"help": "separate all embeddings"})
	separate_some_embeds: str = field(default=None, metadata={"help": "separate some embeddings, based on the given "
																	  "text file"})
	save_name: str = field(default=None, metadata={"help": "save output dir name"})
	main_dir: str = field(default="/nfs/aiml2/nlp_team/fenia/MRPT/", metadata={"help": "cache/saving/load directory"})


def concatenate_examples(tokenized_data, max_seq_length):
	"""
	Approach No1, concatenate examples naively to target for maximum randomization
	"""
	buffers = {}
	concatenated_examples = {}

	ks = list(tokenized_data.keys())
	for key_name in ks:
		if key_name not in buffers:
			buffers[key_name] = []

		if key_name not in concatenated_examples:
			concatenated_examples[key_name] = []

	for ex, example_input in enumerate(tokenized_data['input_ids']):

		if len(buffers['input_ids']) + len(example_input) <= max_seq_length:  # great, actually concatenate this
			for key_name in ks:
				if isinstance(tokenized_data[key_name][ex], list):
					buffers[key_name].extend(tokenized_data[key_name][ex])  # augment buffers
				else:
					buffers[key_name].extend([tokenized_data[key_name][ex]])

		else:
			# if the concatenated example surpasses the length, then return the buffer and add the example in a new one
			for key_name in ks:
				concatenated_examples[key_name].append(buffers[key_name])  # add existing buffers
				buffers[key_name] = []  # empty
				if isinstance(tokenized_data[key_name][ex], list):
					buffers[key_name].extend(tokenized_data[key_name][ex])  # augment buffers
				else:
					buffers[key_name].extend([tokenized_data[key_name][ex]])

	if len(buffers['input_ids']) > 0:
		for key_name in ks:
			concatenated_examples[key_name].append(buffers[key_name])
			buffers[key_name] = []

	return concatenated_examples


def main(args):
	data = datasets.load_dataset(
		"json",
		data_files=os.path.join(args.dataset_dir, "python_data.json"),
		split='train'
	)
	data = data.shuffle(seed=42)

	if args.tokenizer == 'pangu':
		tokenizer = PanguAlphaTokenizer(vocab_file=os.path.join(args.main_dir, "spm/vocab.model"))
	else:
		tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

	special_tokens_dict_code = {
		"additional_special_tokens": ["<eod>", "<eot>", "<pad>", "<java>", "<python>", "<go>", "<php>",
									  "<javascript>", "<ruby>", "<en>", "<cn>", "<comments>",
									  "<NEW_LINE>", "<INDENT>", "<DEDENT>",
									  "<mask>"]
	}
	old_tok_size = len(tokenizer.get_vocab())
	tokenizer.add_special_tokens(special_tokens_dict_code)
	print(tokenizer.all_special_tokens)
	print(tokenizer.additional_special_tokens)
	print(tokenizer.additional_special_tokens_ids)

	print(f"New vocab size with extra special tokens: {len(tokenizer.get_vocab())} vs {old_tok_size}")

	if args.separate_some_embeds is not None:
		print("***** Separating some embeddings *****")
		with open(args.separate_some_embeds, 'r') as infile:
			tmp_extra_tokens = infile.read().splitlines()
			extra_tokens = []
			for et in tmp_extra_tokens:
				extra_tokens.append(et)
				if 'pycodegpt' in args.model_name_or_path:
					extra_tokens.append("Ġ" + et)
				else:
					extra_tokens.append("▁" + et)

	elif args.separate_embeds:
		print("***** Separating the entire space *****")
		extra_tokens = tokenizer.get_vocab()

	else:
		extra_tokens = []

	new_extra_tokens = []
	args.replicated_tokens_map = {}
	tok_vocab = tokenizer.get_vocab()
	extra_count = len(tok_vocab)

	# Extra tokens will *ALWAYS* be appended *AT THE END*
	for e_tok in list(sorted(set(extra_tokens))):
		if e_tok in tokenizer.all_special_tokens:  # if a special token
			continue
		elif e_tok in tok_vocab:
			new_extra_tokens.append(f'[_DUP_]{e_tok}')  # give it another name just to be sure
			# old id -> new id
			args.replicated_tokens_map[tokenizer.convert_tokens_to_ids(e_tok)] = extra_count
			extra_count += 1
		else:
			print(e_tok)
			new_extra_tokens.append(e_tok)
			new_extra_tokens.append(f'[_DUP_]{e_tok}')
			args.replicated_tokens_map[extra_count] = extra_count+1
			extra_count += 2

	print(f"New extra tokens: {len(new_extra_tokens)}")
	print(f"Previous total vocab size: {len(tokenizer.get_vocab())}")
	tokenizer.add_tokens(new_extra_tokens)
	print(f"New total vocab size: {len(tokenizer.get_vocab())}")

	if not os.path.exists(os.path.join(args.main_dir, args.save_name)):
		os.mkdir(os.path.join(args.main_dir, args.save_name))

	with open(os.path.join(args.main_dir, f'{args.save_name}', 'replicated_tokens_map.pkl'), 'wb') as f:
		pickle.dump(args.replicated_tokens_map, f)

	if 'pycodegpt' in args.model_name_or_path:
		tok_func = tokenization_function_raw
	else:
		tok_func = tokenization_function

	tokenized_data = data.map(
		tok_func,
		batched=True,
		remove_columns=['_id', 'docstring', 'code'],
		num_proc=32,
		desc="Running tokenizer on dataset",
		cache_file_name=os.path.join(args.main_dir, "cache/", args.save_name, "tokenized_data.arrow"),
		load_from_cache_file=False,
		fn_kwargs={
			'max_seq_length': args.max_seq_length,
			'tokenizer': tokenizer,
			'new_id_map': args.replicated_tokens_map
		}
	)
	print(tokenized_data)
	for index in random.sample(range(len(tokenized_data)), 3):
		print(f"Sample {index} of the training set")
		print(
			tokenizer.convert_tokens_to_string(
				tokenizer.convert_ids_to_tokens(tokenized_data[index]['input_ids'])).replace('[_DUP_]', '')
		)

	# shuffle
	tokenized_data = tokenized_data.shuffle(seed=42)

	concatenated_data = tokenized_data.map(
		concatenate_examples,
		batched=True,
		batch_size=5000,
		num_proc=32,
		desc="Concatenating examples",
		cache_file_name=os.path.join(args.main_dir, "cache/", args.save_name, "concatenated_data.arrow"),
		load_from_cache_file=False,
		fn_kwargs={
			'max_seq_length': args.max_seq_length
		}
	)
	print(concatenated_data)

	for index in random.sample(range(len(concatenated_data)), 3):
		print(f"Sample {index} of the training set")
		print(
			tokenizer.convert_tokens_to_string(
				tokenizer.convert_ids_to_tokens(concatenated_data[index]['input_ids'])).replace('[_DUP_]', '')
		)

	concatenated_data.save_to_disk(os.path.join(args.main_dir, args.save_name), num_proc=32)


if __name__ == "__main__":
	args = HfArgumentParser(Arguments).parse_args_into_dataclasses()[0]
	main(args)
