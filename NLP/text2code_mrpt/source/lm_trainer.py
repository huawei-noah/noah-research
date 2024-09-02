#!/usr/bin/env python
# coding=utf-8

# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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



"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import torch
import sys
import copy
from dataclasses import dataclass, field
import logging
import datasets
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
)
import random
from callbacks import *
from transformers.trainer_utils import get_last_checkpoint
from pangu_alpha import (
    PanguAlphaModel,
    PanguAlphaTokenizer,
    PanguAlphaConfig
)
from gpt_neo import GPTNeoForCausalLM
from custom_trainer import CustomTrainer
from custom_collator import (
    DataCollatorWithPaddingForCorruptCLM,
    DataCollatorWithPaddingForCLM
)
from tokenization import tokenization_function, tokenization_function_raw
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.utils import see_memory_usage
from optimization import *
import pickle


logger = logging.getLogger(__name__)
datasets.disable_progress_bar()
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("pangu_alpha.generation_utils").setLevel(logging.ERROR)
logging.getLogger("gpt2.modeling_gpt2").setLevel(logging.ERROR)
logging.getLogger("gpt_neo.modeling_gpt_neo").setLevel(logging.ERROR)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The model checkpoint for weights initialization. "
                          "Don't set if you want to train a model from scratch."},
    )
    tokenizer_model: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer 1"}
    )
    num_of_gpus: Optional[int] = field(
        default=1,
        metadata={"help": "Number of GPUs per node"}
    )
    num_of_nodes: Optional[int] = field(
        default=1,
        metadata={"help": "Number of nodes"}
    )
    separate_embeds: Optional[bool] = field(
        default=False,
        metadata={"help": "Duplicate embeddings and treat them as separate"}
    )
    separate_some_embeds: Optional[str] = field(
        default=None,
        metadata={"help": "Text file with tokens to add to the vocabulary"}
    )
    prefix_lm: Optional[bool] = field(
        default=False,
        metadata={"help": "If enabled, used bidirectionality in a region of the input, namely the prefix."}
    )
    min_learning_rate: Optional[float] = field(
        default=0.0,
        metadata={"help": "Minimum Learning rate"}
    )
    debugging: Optional[bool] = field(
        default=False,
        metadata={"help": "Debug or not"}
    )
    predict_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Loss only on code tokens"}
    )
    corrupt_docstring: Optional[bool] = field(
        default=False,
        metadata={"help": "Add masks on the Docstring Only."}
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    data_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum total input sequence length after tokenization."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    load_from_scratch: Optional[bool] = field(
        default=False,
        metadata={"help": "Load model with random weights-training from scratch"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": 'Cache directory'}
    )
    eval_samples: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of samples to use as validation set."}
    )
    validation_percentage: Optional[float] = field(
        default=0.001,
        metadata={"help": "Percentage of data to use as a validation set"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.corrupt_docstring = model_args.corrupt_docstring
    training_args.predict_code = model_args.predict_code
    training_args.dataset_name = data_args.dataset_name
    training_args.prefix_lm = model_args.prefix_lm
    training_args.separate_embeds = model_args.separate_embeds
    training_args.min_learning_rate = model_args.min_learning_rate
    training_args.debugging = model_args.debugging
    training_args.separate_some_embeds = model_args.separate_some_embeds

    # Setup logging
    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    # transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity_debug()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logging.getLogger("transformers.models.gpt2.modeling_gpt2").setLevel(logging.ERROR)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Data hyperparameters: {data_args}")
    logger.info(f"Model hyperparameters: {model_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Handle the repository creation
    if training_args.output_dir is not None:
        os.makedirs(training_args.output_dir, exist_ok=True)

    ##############################
    # TOKENIZER LOADING
    ##############################
    if 'pycodegpt' in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        tokenizer = PanguAlphaTokenizer(vocab_file="spm/vocab.model")

    special_tokens_dict_code = {
        "additional_special_tokens": ["<eod>", "<eot>", "<pad>", "<java>", "<python>", "<go>", "<php>",
                                      "<javascript>", "<ruby>", "<en>", "<cn>", "<comments>",
                                      "<NEW_LINE>", "<INDENT>", "<DEDENT>",
                                      "<mask>"]
    }
    old_tok_size = len(tokenizer.get_vocab())
    tokenizer.add_special_tokens(special_tokens_dict_code)
    logger.info(tokenizer.all_special_tokens)
    logger.info(tokenizer.additional_special_tokens)
    logger.info(tokenizer.additional_special_tokens_ids)

    logger.info(f"New vocab size with extra special tokens: {len(tokenizer.get_vocab())} vs {old_tok_size}")

    if training_args.separate_some_embeds is not None:
        logger.info("***** Separating some embeddings *****")
        with open(training_args.separate_some_embeds, 'r') as infile:
            tmp_extra_tokens = infile.read().splitlines()
            extra_tokens = []
            for et in tmp_extra_tokens:
                extra_tokens.append(et)
                if 'pycodegpt' in model_args.model_name_or_path:
                    extra_tokens.append("Ġ" + et)
                else:
                    extra_tokens.append("▁" + et)

    elif training_args.separate_embeds:
        logger.info("***** Separating the entire space *****")
        extra_tokens = tokenizer.get_vocab()

    else:
        extra_tokens = []

    with training_args.main_process_first():
        new_extra_tokens = []
        training_args.replicated_tokens_map = {}
        tok_vocab = tokenizer.get_vocab()
        extra_count = len(tok_vocab)

        # Extra tokens will *ALWAYS* be appended *AT THE END*
        for e_tok in list(sorted(set(extra_tokens))):
            if e_tok in tokenizer.all_special_tokens:  # if a special token
                continue
            elif e_tok in tok_vocab:
                new_extra_tokens.append(f'[_DUP_]{e_tok}')  # give it another name just to be sure
                # old id -> new id
                training_args.replicated_tokens_map[tokenizer.convert_tokens_to_ids(e_tok)] = extra_count
                extra_count += 1
            else:
                # print(e_tok)
                new_extra_tokens.append(e_tok)
                new_extra_tokens.append(f'[_DUP_]{e_tok}')
                training_args.replicated_tokens_map[extra_count] = extra_count + 1
                extra_count += 2

    logger.info(f"New extra tokens: {len(new_extra_tokens)}")
    logger.info(f"Previous total vocab size: {len(tokenizer.get_vocab())}")
    tokenizer.add_tokens(new_extra_tokens)
    logger.info(f"New total vocab size: {len(tokenizer.get_vocab())}")

    # with open(os.path.join(data_args.dataset_name, 'replicated_tokens_map.pkl'), 'rb') as f:
    #     training_args.replicated_tokens_map = pickle.load(f)

    ##############################
    # MODEL LOADING
    ##############################
    set_seed(training_args.seed)

    if data_args.load_from_scratch:
        logger.info("Initialising model to be trained from scratch !")
        model_config = PanguAlphaConfig.from_pretrained(model_args.model_name_or_path)
        model = PanguAlphaModel(
            config=model_config,
            args=training_args
        )
    elif 'pycodegpt' in model_args.model_name_or_path:
        logger.info(f"Loading model from checkpoint {model_args.model_name_or_path}")
        model, missed_info = GPTNeoForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            output_loading_info=True,
            local_files_only=True,
            args=training_args,
            tokenizer=tokenizer
        )
        logger.info(missed_info)
    else:
        logger.info(f"Loading model from checkpoint {model_args.model_name_or_path}")
        model, missed_info = PanguAlphaModel.from_pretrained(
            model_args.model_name_or_path,
            output_loading_info=True,
            local_files_only=True,
            args=training_args,
            tokenizer=tokenizer
        )
        logger.info(missed_info)

    ##############################################################
    # Extend the embedding layer
    ##############################################################
    model.resize_token_embeddings(len(tokenizer))

    ############################
    # DeepSpeed check
    ############################
    if training_args.deepspeed and 's3' in training_args.deepspeed:
        estimate_zero3_model_states_mem_needs_all_live(
            model,
            num_gpus_per_node=model_args.num_of_gpus,
            num_nodes=model_args.num_of_nodes
        )
        see_memory_usage("Memory used, after model loading .from_pretrained", force=True)

    elif training_args.deepspeed and 's2' in training_args.deepspeed:
        estimate_zero2_model_states_mem_needs_all_live(
            model,
            num_gpus_per_node=model_args.num_of_gpus,
            num_nodes=model_args.num_of_nodes
        )
        see_memory_usage("Memory used, after model loading .from_pretrained", force=True)

    # ----- HACK BEGIN ------ #
    with training_args.main_process_first():
        if model_args.separate_embeds or model_args.separate_some_embeds:
            logger.info('*** INITIALIZING SEPARATE EMBEDS FROM EXISTING ONES ***')

            for tok_id_src, tok_id_tgt in training_args.replicated_tokens_map.items():
                model.transformer.wte.weight.data[tok_id_tgt, :] = \
                    copy.copy(model.transformer.wte.weight.data[tok_id_src, :])

            token_id = tokenizer.convert_tokens_to_ids('return')
            token_id_new = tokenizer.convert_tokens_to_ids('[_DUP_]return')

            assert torch.eq(model.transformer.wte.weight.data[token_id],
                            model.transformer.wte.weight.data[token_id_new]).all()
            logger.info(f"-> Assertion passed: Equal embeds of id = {token_id} and id = {token_id_new}")
            logger.info(f'*** FINISHED INITIALIZATION ***')
    # ----- HACK END ----- #

    # Make sure padding tokens have a zero vector~
    logger.info('*** Making padded tokens have a 0 vector ***')
    if 'pycodegpt' in model_args.model_name_or_path:
        pad_id = tokenizer.convert_tokens_to_ids('<|padoftext|>')
    else:
        pad_id = tokenizer.convert_tokens_to_ids('<pad>')
    model.transformer.wte.weight.data[pad_id, :] = 0

    # Modeling sizing
    model_size = sum(t.numel() for n, t in model.named_parameters())
    model_size_wo_embed = sum(t.numel() for n, t in model.named_parameters() if 'transformer.wte' not in n)

    logger.info(f"Training model -> Total size = {model_size} parameters")
    logger.info(f"Training model -> Total size w/o embedding layer = {model_size_wo_embed} parameters")

    logger.info(f"Total size of embedding layer: {model.transformer.wte.weight.size()}")
    for n, p in model.named_parameters():
        logger.info(f"{n} -> {p.requires_grad}")

    ####################
    # DATA
    ####################
    logger.info(f"Loading data from {data_args.dataset_name} directory")
    data = datasets.load_from_disk(data_args.dataset_name)
    logger.info(data)

    # Split in Train and Validation
    logger.info("Splitting dataset into train and test")
    dataset = data.train_test_split(test_size=data_args.validation_percentage, shuffle=True, seed=42)
    train_dataset, eval_dataset = dataset['train'], dataset['test']
    logger.info(f"Training set -> {train_dataset}")
    logger.info(f"Evaluation set -> {eval_dataset}")

    # Print 3 samples from evaluation set
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set")
        logger.info(
             tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(eval_dataset[index]['input_ids'])).replace('[_DUP_]', '')
        )

    ############################
    # DATA_COLLATORS
    ############################
    if model_args.corrupt_docstring:
        logger.info("Using Dynamic Padding Data Collator for corrupted CLM")
        my_collator = DataCollatorWithPaddingForCorruptCLM(
            predict_code=model_args.predict_code,
            tokenizer=tokenizer,
            prefix_lm=training_args.prefix_lm,
            code_mask=True if 'pycodegpt' in model_args.model_name_or_path else False,
        )
    else:
        logger.info("Using Dynamic Padding Data Collator for CLM")
        my_collator = DataCollatorWithPaddingForCLM(
            predict_code=model_args.predict_code,
            tokenizer=tokenizer,
            prefix_lm=training_args.prefix_lm,
            code_mask=True if 'pycodegpt' in model_args.model_name_or_path else False,
        )

    ############################
    # TRAINER
    ############################
    logger.info("******* Running with the following arguments *********")
    for a in sys.argv[1:]:
        logger.info(f"  {a}")
    logger.info("******************************************************")

    if 'pycodegpt' in model_args.model_name_or_path:
        generation_callback = GenerationCallbackRaw(tokenizer=tokenizer)
    else:
        generation_callback = GenerationCallback(tokenizer=tokenizer)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=my_collator,
        callbacks=[generation_callback]
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()