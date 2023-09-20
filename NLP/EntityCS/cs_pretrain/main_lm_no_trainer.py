#!/usr/bin/env python
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

import argparse
import logging
import math
import random
import os
import time

import sys
import numpy as np
import datasets
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

import transformers
from accelerate import Accelerator, DistributedType
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from transformers.utils.versions import require_version

from utils import get_sampling_probability_from_counts, humanized_time, str2bool
from samplers import BatchLanguageSampler
from my_data_collator import (
    DynamicDataCollatorForEntityMasking,
    DynamicDataCollatorForLanguageModeling
)

logger = logging.getLogger(__name__)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# datasets.set_progress_bar_enabled(False)
# datasets.set_caching_enabled(False)

with open("languages/xlmr_langs.txt") as infile:
    XLMR_LANGS = [line.rstrip() for line in infile]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a Masked Language Modeling task on CS corpus"
    )
    parser.add_argument(
        "--hf_datasets_folder_no_en",
        type=str,
        default=None,
        help="A folder containing the huggingface datasets with parallel sentences",
    )
    parser.add_argument(
        "--hf_datasets_folder_en",
        type=str,
        default=None,
        required=True,
        help="A folder containing the huggingface datasets with English sentences",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=None,
        help="The percentage of the train set used as validation set",
    )
    parser.add_argument(
        "--validation_examples_per_lang",
        type=int,
        default=20,
        help="Number of examples per language in training datasets used as validation set if no validation_split is provided",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the models."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Set patience value for early stopping"
    )
    parser.add_argument(
        "--validation_steps", type=int, default=1000, help="Set validation steps"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="If passed, will use FP16 training."
    )
    parser.add_argument(
        "--max_clip_grad_norm", type=float, default=1.0, help="Set max clip grad norm."
    )
    parser.add_argument(
        "--languages", type=str, default=None, help="File with Languages to load."
    )
    parser.add_argument(
        "--partial_masking",
        type=str2bool,
        help="If set, partially mask subwords in an entity.",
    )
    parser.add_argument(
        "--keep_random",
        type=str2bool,
        help="Valid when partial_masking is True. If set, 10% masking tokens will be replaced with random in an entity.",
    )
    parser.add_argument(
        "--keep_same",
        type=str2bool,
        help="If set, 10% masking tokens will be input and we calculate loss on them.",
    )
    parser.add_argument(
        "--entity_probability",
        type=float,
        default=0.5,
        help="How much percentage of entities (tokens) as masking candidates.",
    )
    parser.add_argument(
        "--insert_lang_tag",
        type=str2bool,
        help="If set, add language tags at the beginning of a sentence, after [CLS]",
    )
    parser.add_argument(
        "--update_partial_layers",
        nargs="*",
        default=None,
        help="If set, only update the layers mentioned.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="If set, models are saved after k evaluations even it is no the best PPL.",
    )
    parser.add_argument(
        "--multi_node", type=str2bool, default=False, help="Multi-node training."
    )
    parser.add_argument(
        "--masking",
        type=str,
        default=False,
        help="Choose mode of masking: Entity Prediction (EP), Masked Language Modeling (MLM), "
             "Entity Predictions and MLM",
        choices=["ep", "mlm", "ep-mlm"],
    )
    args = parser.parse_args()

    # Sanity checks
    if args.hf_datasets_folder_no_en is None and args.hf_datasets_folder_en is None:
        raise ValueError("Training folder for cs corpus is needed.")

    return args


def main():
    patience = 0
    args = parse_args()

    if args.languages is not None:
        with open(args.languages + ".txt") as infile:
            args.languages = [line.rstrip() for line in infile]
        logger.info(args.languages)

        for lang in args.languages:
            if lang not in XLMR_LANGS:
                args.languages.remove(lang)
                logger.warning(
                    f"Dataset for language {lang} does not exist, removing from training languages."
                )

    lang2id = {}
    id2lang = {}
    for i, lang in enumerate(args.languages):
        lang2id[lang] = i
        id2lang[i] = lang

    ########## DISTRIBUTED HACK ##########
    if args.multi_node:
        my_init_method = ""
        my_rank = int(os.getenv("RANK", "0"))
        my_world_size = int(os.getenv("WORLD_SIZE", "1"))
        my_master_ip = os.getenv("MASTER_ADDR", "localhost")
        my_master_port = os.getenv("MASTER_PORT", "6000")
        my_init_method += my_master_ip + ":" + my_master_port
        logger.info(f'Local rank: {os.getenv("LOCAL_RANK", -1)}')
        logger.info(
            f"Rank: {my_rank} || World_size: {my_world_size} || Init: {my_init_method}"
        )
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=my_world_size,
            rank=my_rank,
            init_method=my_init_method,
        )
    ######################################

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator(fp16=args.fp16)
    # Make one log on every process with the configuration for debugging.

    # Setup logging
    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    transformers.utils.logging.set_verbosity_debug()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(
        logging.INFO if accelerator.is_local_main_process else logging.ERROR
    )
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get datasets
    logger.info(f"Loading huggingface datasets")
    raw_datasets_by_lang = {}
    for lang in args.languages:
        logger.info(f"Loading {lang.upper()}")

        raw_datasets_by_lang[lang] = load_dataset(
            "huawei-noah/entity_cs", data_files=f"data/{lang}/", split="train"
        )

        if lang == "en":
            # EN dataset does not have "language" column, so we add it now
            new_col = [lang] * len(raw_datasets_by_lang[lang])
            raw_datasets_by_lang[lang] = raw_datasets_by_lang[lang].add_column(
                "language", new_col
            )
            # rename the column to have the same column name as the Non-EN sentences
            raw_datasets_by_lang[lang] = raw_datasets_by_lang[lang].rename_column(
                "en_sentence", "cs_sentence"
            )
        else:
            # We are not feeding parallel data, so remove en_sentence from the dataset
            raw_datasets_by_lang[lang] = raw_datasets_by_lang[lang].remove_columns(
                "en_sentence"
            )

        # shuffle cs_sentences in all langs
        raw_datasets_by_lang[lang] = raw_datasets_by_lang[lang].shuffle(seed=args.seed)

    # merge langs to one train, validation dataset
    train_datasets = []
    val_datasets = []
    if args.validation_split is not None:
        for lang in args.languages:
            train_val = raw_datasets_by_lang[lang].train_test_split(
                test_size=args.validation_split
            )
            train_datasets.append(train_val["train"])
            val_datasets.append(train_val["test"])
    else:
        # raw_datasets already shuffled, just take the first args.validation_examples_per_lang in each lang as validation
        for lang in args.languages:
            val_datasets.append(
                raw_datasets_by_lang[lang].select(
                    range(args.validation_examples_per_lang)
                )
            )
            train_datasets.append(
                raw_datasets_by_lang[lang].select(
                    range(
                        args.validation_examples_per_lang,
                        len(raw_datasets_by_lang[lang]),
                    )
                )
            )

    raw_datasets = datasets.DatasetDict()
    raw_datasets["validation"] = concatenate_datasets(val_datasets)
    raw_datasets["train"] = concatenate_datasets(train_datasets)

    logger.info(f"dataset by lang: {raw_datasets_by_lang}")
    logger.info(f"merged dataset: {raw_datasets}")

    # calculate number of training samples for each language and sampling probability
    train_lang_nums = [len(data) for data in train_datasets]
    _, sampling_probs_mlm = get_sampling_probability_from_counts(train_lang_nums)

    logger.info(f"train_lang_nums: {train_lang_nums}")
    logger.info(f"sampling_probs_mlm: {sampling_probs_mlm}")

    # start and end idx for each lang in training dataset
    train_lang_idx = {}
    cur_idx = 0
    for lang, train_lang_num in zip(args.languages, train_lang_nums):
        train_lang_idx[lang] = (cur_idx, cur_idx + train_lang_num - 1)
        cur_idx += train_lang_num

    logger.info(
        f"(start_idx, end_idx) for each language in the merged train_datasets: {train_lang_idx}"
    )

    # Sampler
    logger.info("Creating sampling batches")
    t0 = time.time()
    sampler = BatchLanguageSampler(
        batch_size=args.per_device_train_batch_size,
        train_lang_nums=train_lang_nums,
        langs=args.languages,
        lang_idx=train_lang_idx,
    )
    logger.info(f"Sampling bathes created, it took {humanized_time(time.time() - t0)}.")

    config = AutoConfig.from_pretrained(args.model_name_or_path.split("/")[-1],
                                        hidden_dropout_prob=0.0,
                                        attention_probs_dropout_prob=0.0)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=not args.use_slow_tokenizer,
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path, config=config
    )
    print(model)

    if args.masking != "mlm":
        # add lang tags as special tokens
        special_tokens_dict = {
            "additional_special_tokens": [f"<{l}>" for l in args.languages]
            + [f"</{l}>" for l in args.languages]
        }

        if "en" not in args.languages:
            # even "en" is not in args.languages, en lang tags may still appear in cs_sentences, need to add <en>, </en>
            special_tokens_dict["additional_special_tokens"] += ["<en>", "</en>"]

        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(
            f"Added {num_added_toks} special tokens: {tokenizer.additional_special_tokens}"
        )

        model.resize_token_embeddings(len(tokenizer))

    if args.update_partial_layers is not None:
        logger.info(
            f"Only the following layers will be updated: {args.update_partial_layers}"
        )
        # non_frozen_layers = ['embeddings', '10', '11', 'lm_head']
        logger.info("Model layers BEFORE freezing")
        for n, p in model.named_parameters():
            logger.info(f"n: {n}, p: {p.requires_grad}")
            if not any(layer in n for layer in args.update_partial_layers):
                p.requires_grad = False

        logger.info("Model layers AFTER freezing")
        for n, p in model.named_parameters():
            logger.info(f"n: {n}, p: {p.requires_grad}")

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    logger.info(f"Selected MAX_SEQ_LEN: {max_seq_length}")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(
            f"Sample {index} of the training set: {raw_datasets['train'][index]}."
        )

    # Data collator
    if args.masking == "mlm":
        data_collator = DynamicDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=args.mlm_probability,
            id2lang=id2lang,
            max_length=max_seq_length,
        )
    else:  # EP or *EP_MLM
        data_collator = DynamicDataCollatorForEntityMasking(
            tokenizer=tokenizer,
            id2lang=id2lang,
            insert_lang_tag=args.insert_lang_tag,
            max_length=max_seq_length,
            entity_probability=args.entity_probability,
            masking=args.masking,
            partial_masking=args.partial_masking,
            keep_random=args.keep_random,
            keep_same=args.keep_same,
        )

    # Data loader
    train_dataloader = DataLoader(
        raw_datasets["train"],
        batch_sampler=sampler,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        raw_datasets["validation"],
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    for n, p in model.named_parameters():
        if p.requires_grad:
            logger.info(f"n: {n}, p: {p.requires_grad}")

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, eval_dataloader, train_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader, train_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(
            args.warmup_ratio * args.num_train_epochs * num_update_steps_per_epoch
        ),
        num_training_steps=args.max_train_steps,
    )

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("******* Running with the following arguments *********")
    for a in sys.argv[1:]:
        logger.info(f"  {a}")
    logger.info("******************************************************")

    logger.info("******* Running training *******")
    logger.info(f"  Num examples = {len(raw_datasets['train'])}")
    logger.info(f"  Validation examples: {len(raw_datasets['validation'])}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Grad. Norm clip = {args.max_clip_grad_norm}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Validation steps = {args.validation_steps}")
    logger.info(f"  Distributed type is MULTI_GPU = {accelerator.distributed_type == DistributedType.MULTI_GPU}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    counter = 0  # count how many batches processed
    best_step = 0
    best_perplexity = float("inf")
    evaluations = 0

    # ------------------------ Training loop ---------------------------
    # patience and max_train_steps will determine whether break the loop

    sampling_lang_dict = {}
    langs_not_used_dict = {}
    train_losses = []
    t0 = time.time()
    printed_labels = False

    for epoch in range(args.num_train_epochs):  # just 1 epoch for now
        for step, batch in enumerate(train_dataloader):
            model.train()

            if args.masking != "mlm":
                langs_not_used = batch["langs_not_used"]

                for k, v in langs_not_used.items():
                    langs_not_used_dict[k] = langs_not_used_dict.get(k, 0) + v.item()

            batch_lang = batch["language"]
            for l in batch_lang:
                lang = id2lang[l.item()]
                sampling_lang_dict[lang] = (
                    sampling_lang_dict.get(lang, 0) + args.per_device_train_batch_size
                )

            counter += 1

            batch_ = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }

            if not printed_labels:
                for i in range(min(4, args.per_device_train_batch_size)):
                    logger.info(
                        f"input_tokens: {tokenizer.convert_ids_to_tokens(batch['input_ids'][i])}"
                    )
                    logger.info(f"labels: {batch['labels'][i]}")
                    logger.info(f"labels str: {[tokenizer.convert_ids_to_tokens(ii.item()) if ii.item() != -100 else -100 for ii in batch['labels'][i]]}")
                    for tt1, tt2 in zip(tokenizer.convert_ids_to_tokens(batch['input_ids'][i]),
                                        [tokenizer.convert_ids_to_tokens(ii.item()) if ii.item() != -100 else -100 for ii in batch['labels'][i]]):
                        logger.info(f'{str(tt1):<15} -> {str(tt2)}')
                    logger.info('*' * 10)
                printed_labels = True

            outputs = model(**batch_)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            train_losses.append(loss.item())

            if counter % args.gradient_accumulation_steps != 0:
                # Prevent backward from doing gradient all_reduce in every step
                if accelerator.distributed_type == DistributedType.MULTI_GPU:
                    with model.no_sync():
                        accelerator.backward(loss)
                else:
                    accelerator.backward(loss)

            else: # Accumulation is over
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_clip_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

                # evaluate every args.validation_steps, only evaluate if completed_steps has changed value
                if completed_steps > 0 and completed_steps % args.validation_steps == 0:
                    evaluations += 1
                    logger.info(
                        f"{completed_steps} steps completed - Evaluating! - "
                        f"Elapsed time: {humanized_time(time.time() - t0)}."
                    )
                    logger.info(f"Model training: {model.training}")
                    logger.info(f"sampling_lang_dict: {sampling_lang_dict}")

                    if args.masking != "mlm":
                        logger.info(f"langs_not_used_dict: {langs_not_used_dict}")
                        actual_lang_dict = {}
                        for k, v in sampling_lang_dict.items():
                            if k in langs_not_used_dict:
                                actual_lang_dict[k] = (
                                    sampling_lang_dict[k] - langs_not_used_dict[k]
                                )
                            else:
                                actual_lang_dict[k] = sampling_lang_dict[k]
                        logger.info(f"actual_lang_dict: {actual_lang_dict}")

                    model.eval()
                    losses = []

                    for step_val, batch_val in enumerate(eval_dataloader):
                        with torch.no_grad():
                            batch_ = {
                                "input_ids": batch_val["input_ids"],
                                "attention_mask": batch_val["attention_mask"],
                                "labels": batch_val["labels"],
                            }
                            outputs = model(**batch_)
                            loss = outputs.loss

                        losses.append(
                            accelerator.gather(
                                loss.repeat(args.per_device_eval_batch_size)
                            )
                        )

                    losses = torch.cat(losses)
                    losses = losses[: len(raw_datasets["validation"])]
                    try:
                        perplexity = math.exp(torch.mean(losses))
                    except OverflowError:
                        perplexity = float("inf")

                    logger.info(
                        f"Evaluation at step: {completed_steps}, "
                        f"TRAIN perplexity: {math.exp(np.mean(train_losses)):.4f}, "
                        f"EVAL perplexity: {perplexity:.4f}"
                    )
                    train_losses = []

                    # check whether it is the best model up to date and save the best model
                    if perplexity < best_perplexity:
                        patience = 0
                        best_perplexity = perplexity
                        best_step = completed_steps
                        logger.info(
                            f"Saving best model to date at step {completed_steps} with perplexity: {perplexity:.4f}"
                        )
                        # save model and log file
                        output_dir = os.path.join(
                            args.output_dir,
                            f"step_{completed_steps}_ppl_{perplexity:.4f}",
                        )
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir, save_function=accelerator.save
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(output_dir)

                    else:
                        patience += 1
                        # also save the model every 5 evaluations even the PPL is not the best
                        if evaluations % args.save_every == 0:
                            logger.info(
                                f"Saving model at step {completed_steps} after {evaluations} evaluations with perplexity: {perplexity:.4f}"
                            )
                            # save model and log file
                            output_dir = os.path.join(
                                args.output_dir,
                                f"step_{completed_steps}_ppl_{perplexity:.4f}",
                            )
                            accelerator.wait_for_everyone()
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                output_dir, save_function=accelerator.save
                            )
                            if accelerator.is_main_process:
                                tokenizer.save_pretrained(output_dir)

                    if patience == args.patience:
                        logger.info(
                            f"PPL not improved for {args.patience} evaluations. Stopping early!"
                        )
                        break

                    logger.info("Continue Training...")
                    t0 = time.time()

            if completed_steps >= args.max_train_steps:
                logger.info(f"Max train steps reached. Stopping training!")
                break

    # ----------------------End of Training loop ---------------------------

    logger.info(f"Best step: {best_step}, perplexity: {best_perplexity:.4f}")

    # evaluate at the end of training, useful when training exit due to reaching max_train_steps
    logger.info("End of training Evaluating!")

    logger.info(f"sampling_lang_dict: {sampling_lang_dict}")
    if args.masking != "mlm":
        logger.info(f"langs_not_used_dict: {langs_not_used_dict}")
        actual_lang_dict = {}
        for k, v in sampling_lang_dict.items():
            if k in langs_not_used_dict:
                actual_lang_dict[k] = sampling_lang_dict[k] - langs_not_used_dict[k]
            else:
                actual_lang_dict[k] = sampling_lang_dict[k]
        logger.info(f"actual_lang_dict: {actual_lang_dict}")

    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch_ = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"],
            }
            outputs = model(**batch_)
            loss = outputs.loss

        losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(raw_datasets["validation"])]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"End of training: perplexity: {perplexity:.4f}")

    if args.output_dir is not None:
        output_dir = os.path.join(
            args.output_dir, f"step_{completed_steps}_ppl_{perplexity:.4f}"
        )
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
