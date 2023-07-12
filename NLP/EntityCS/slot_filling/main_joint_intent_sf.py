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
import os
import random
import json
import sys

import datasets
import torch
from datasets import ClassLabel, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from modeling import XLMRobertaForTokenAndIntentClassification
from data_collator import DataCollatorForIntentAndTokenClassification
import numpy as np


def str2bool(i):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(i, bool):
        return i
    if i.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif i.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


logger = logging.getLogger(__name__)
require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/token-classification/requirements.txt",
)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on joint intent classification and slot filling task with accelerate library"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the load_from_disk).",
    )
    parser.add_argument(
        "--text_column_name",
        type=str,
        default=None,
        help="The column name of text to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--label_column_name",
        type=str,
        default=None,
        help="The column name of label to input in the file (a csv or JSON file).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
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
        "--output_dir", type=str, default=None, help="Where to store the final model."
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
        "--label_all_tokens",
        action="store_true",
        help="Setting labels of all special tokens to -100 and thus PyTorch will ignore them.",
    )
    parser.add_argument(
        "--return_entity_level_metrics",
        action="store_true",
        help="Indication whether entity level metrics are to be returner.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument(
        "--all_val_langs",
        type=str2bool,
        help="Use all target languages as validation set.",
    )
    parser.add_argument(
        "--eval_languages", default=None, type=str, help="Languages to test."
    )
    parser.add_argument(
        "--add_language_tag",
        type=str2bool,
        help="Add language tag or not in front of the sentence.",
    )
    parser.add_argument(
        "--pretraining_languages",
        default=None,
        type=str,
        help="Languages used during pre-training.",
    )
    parser.add_argument(
        "--xlmr_langs",
        default="../languages/xlmr_langs",
        type=str,
        help="All languages used in xlmr.",
    )
    parser.add_argument("--do_predict", action="store_true", help="Prediction")
    parser.add_argument(
        "--max_clip_grad_norm", type=float, default=1.0, help="Set max clip grad norm."
    )
    parser.add_argument(
        "--evals_per_epoch",
        type=int,
        default=None,
        help="Number of update steps between two evaluations.",
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
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

    accelerator.wait_for_everyone()

    # Pre-training langs in any
    if args.pretraining_languages is not None and args.add_language_tag:
        with open(args.pretraining_languages + ".txt") as infile:
            args.pretraining_languages = [line.rstrip() for line in infile]
            logger.info(f"pretraining_languages: {args.pretraining_languages}")
        with open(args.xlmr_langs + ".txt") as infile:
            XLMR_LANGS = [line.rstrip() for line in infile]
            for l in args.pretraining_languages:
                if l not in XLMR_LANGS:
                    logger.info(
                        f"{l} is not in XLMR_LANGS, removing from pretraining languages"
                    )
                    args.pretraining_languages.remove(l)
        logger.info(f"final pretraining_languages: {args.pretraining_languages}")

    # ======= DATA ======= #
    all_data = {}
    for lang in args.eval_languages.split(","):
        all_data[lang] = datasets.load_from_disk(os.path.join(args.dataset_name, lang))
    logger.info(all_data)

    # ENGLISH only for training!
    raw_datasets = {"train": all_data["en"]["train"]}

    if args.all_val_langs:
        raw_datasets["validation"] = datasets.concatenate_datasets(
            [all_data[lang]["validation"] for lang in args.eval_languages.split(",")]
        )
    else:
        raw_datasets["validation"] = all_data["en"]["validation"]

    # Load all test data for final prediction
    raw_datasets["test"] = datasets.DatasetDict(
        {f"{lang}": all_data[lang]["test"] for lang in args.eval_languages.split(",")}
    )

    # ===================== #

    column_names = raw_datasets["train"].column_names
    features = raw_datasets["train"].features

    # the dataset for joint intent classification and slot filling should have at least the following columns:
    # tokens, slot_tags, intent
    text_column_name = "tokens"
    slot_label_column_name = "slot_tags"
    intent_label_column_name = "intent"

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_all_label_list(labels_list, is_intent=False):
        unique_labels = set()
        for labels in labels_list:
            if is_intent:
                unique_labels = unique_labels | set(labels)
            else:
                for label in labels:
                    unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[slot_label_column_name].feature, ClassLabel):
        slot_label_list = features[slot_label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        slot_label_to_id = {i: i for i in range(len(slot_label_list))}
    else:
        all_slot_label_data = [raw_datasets["train"][slot_label_column_name]]
        if args.all_val_langs:
            for lang in args.eval_languages.split(","):
                all_slot_label_data.append(
                    raw_datasets["validation"][lang][slot_label_column_name]
                )
        else:
            all_slot_label_data.append(
                raw_datasets["validation"][slot_label_column_name]
            )
        for lang in args.eval_languages.split(","):
            all_slot_label_data.append(
                raw_datasets["test"][lang][slot_label_column_name]
            )
        slot_label_list = get_all_label_list(all_slot_label_data)
        slot_label_to_id = {l: i for i, l in enumerate(slot_label_list)}
    num_slot_labels = len(slot_label_list)
    logger.info(f"num_slot_labels: {num_slot_labels}")
    logger.info(f"slot_label_to_id: {slot_label_to_id}")

    all_intent_labels = [raw_datasets["train"][intent_label_column_name]]
    if args.all_val_langs:
        for lang in args.eval_languages.split(","):
            all_intent_labels.append(
                raw_datasets["validation"][lang][intent_label_column_name]
            )
    else:
        all_intent_labels.append(raw_datasets["validation"][intent_label_column_name])
    for lang in args.eval_languages.split(","):
        all_intent_labels.append(raw_datasets["test"][lang][intent_label_column_name])
    intent_label_list = get_all_label_list(all_intent_labels, is_intent=True)
    intent_label_to_id = {l: i for i, l in enumerate(intent_label_list)}
    num_intent_labels = len(intent_label_list)
    logger.info(f"num_intent_labels: {num_intent_labels}")
    logger.info(f"intent_label_to_id: {intent_label_to_id}")

    # Map that sends B-Xxx label to its I-Xxx counterpart
    b_to_i_label = []
    for idx, label in enumerate(slot_label_list):
        if label.startswith("B-") and label.replace("B-", "I-") in slot_label_list:
            b_to_i_label.append(slot_label_list.index(label.replace("B-", "I-")))
        else:
            b_to_i_label.append(idx)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_slot_labels,
        label2id=slot_label_to_id,
        id2labels={i: l for l, i in slot_label_to_id.items()},
        cache_dir=None,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )

    tokenizer_name_or_path = (
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    )
    if config.model_type in {"gpt2", "roberta"}:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=None,
            use_fast=True,
            add_prefix_space=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=None,
            use_fast=True,
        )

    # We are using xlmr-based model
    # adding intent labels here
    model = XLMRobertaForTokenAndIntentClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        num_intent_labels=int(num_intent_labels),
        cache_dir=None,
    )
    print(model)

    # Preprocessing the datasets.
    # ADD LANGUAGE TAGS OR NOT to the target task!
    if args.add_language_tag:
        logger.info("Language Tag option is TRUE")

        # for special_token, special_token_id in zip(list(tokenizer.all_special_tokens), list(tokenizer.all_special_ids)):
        logger.info(tokenizer.all_special_tokens)
        logger.info(tokenizer.all_special_ids)

        # Training set
        raw_datasets["train"] = raw_datasets["train"].map(
            lambda ex: {text_column_name: ["<en>"] + ex[text_column_name]},
            desc="Adding Language Tag in train dataset",
        )

        # Validation set
        if args.all_val_langs:
            raw_datasets["validation"] = raw_datasets["validation"].map(
                lambda ex: {
                    text_column_name: [f"<{ex['langs'][0]}>"] + ex[text_column_name]
                }
                if ex["langs"][0] in args.pretraining_languages
                else {text_column_name: ex[text_column_name]},
                desc="Adding Language Tag in validation dataset",
            )
        else:
            raw_datasets["validation"] = raw_datasets["validation"].map(
                lambda ex: {text_column_name: ["<en>"] + ex[text_column_name]},
                desc="Adding Language Tag in validation dataset",
            )

        # Test set
        for lang in args.eval_languages.split(","):
            raw_datasets["test"][lang] = raw_datasets["test"][lang].map(
                lambda ex: {text_column_name: [f"<{lang}>"] + ex[text_column_name]}
                if lang in args.pretraining_languages
                else {text_column_name: ex[text_column_name]},
                desc="Adding Language Tag in prediction dataset",
            )

    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(
            f"Sample {index} of the training set: {raw_datasets['train'][index]}."
        )

    # Tokenize all texts and align the labels with them.
    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_and_align_labels(examples):

        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_length,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        # for zh an ja it adds extra empty '▁' and we need to remove them
        clean_word_ids = {}
        for i, input_ids in enumerate(tokenized_inputs["input_ids"]):
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            assert len(word_ids) == len(input_tokens)

            clean_word_ids[i] = []
            clean_input_ids = []
            attention_mask = []

            for idx, token in enumerate(input_tokens):
                if not token == "▁":
                    clean_word_ids[i].append(word_ids[idx])
                    clean_input_ids.append(tokenized_inputs["input_ids"][i][idx])
                    attention_mask.append(tokenized_inputs["attention_mask"][i][idx])
            tokenized_inputs["input_ids"][i] = clean_input_ids
            tokenized_inputs["attention_mask"][i] = attention_mask

            assert (
                len(tokenized_inputs["input_ids"][i])
                == len(tokenized_inputs["attention_mask"][i])
                == len(clean_word_ids[i])
            )

        slot_labels = []

        for i, slot_label in enumerate(examples[slot_label_column_name]):

            word_ids = clean_word_ids[i]
            token_ids = tokenized_inputs["input_ids"][i]

            assert len(word_ids) == len(token_ids)

            previous_word_idx = None
            label_ids = []
            for word_idx, token_idx in zip(word_ids, token_ids):

                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(slot_label_to_id[slot_label[word_idx]])

                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if args.label_all_tokens:
                        label_ids.append(
                            b_to_i_label[slot_label_to_id[slot_label[word_idx]]]
                        )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            slot_labels.append(label_ids)
        tokenized_inputs["slot_labels"] = slot_labels
        tokenized_inputs["intent_labels"] = [
            intent_label_to_id[intent] for intent in examples[intent_label_column_name]
        ]
        return tokenized_inputs

    with accelerator.main_process_first():
        train_dataset = raw_datasets["train"].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer Train on dataset",
        )
        eval_dataset = raw_datasets["validation"].map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on Validation dataset",
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # ----- Set number of evaluations ----- #
    if args.evals_per_epoch:
        minibatch = int(len(train_dataset) / args.per_device_train_batch_size)
        args.eval_steps = min(int(minibatch / args.evals_per_epoch), 500)
    # ------------------------------------ #

    # DataLoaders creation:
    # we use `DataCollatorForIntentAndTokenClassification` that applies dynamic padding for us (by padding to the maximum length of
    # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
    # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
    data_collator = DataCollatorForIntentAndTokenClassification(
        tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

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

    # Metrics
    slot_metric = load_metric("seqeval")
    intent_metric = load_metric("accuracy")

    def get_slot_labels(predictions, references):
        # Transform predictions and references tensors to numpy arrays

        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

        # Remove ignored index (special tokens)
        true_slot_predictions = [
            [slot_label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        true_slot_labels = [
            [slot_label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
            for pred, gold_label in zip(y_pred, y_true)
        ]
        return true_slot_predictions, true_slot_labels

    def compute_metrics():
        # compute both slot and intent metrics - once compute() the cached preds and labels should be cleared
        slot_results = slot_metric.compute()
        intent_results = intent_metric.compute()
        if args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in slot_results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            for k, v in intent_results.items():

                final_results[f"intent_{k}"] = v
            return final_results
        else:
            return {
                "intent_accuracy": intent_results["accuracy"],
                "precision": slot_results["overall_precision"],
                "recall": slot_results["overall_recall"],
                "f1": slot_results["overall_f1"],
                "accuracy": slot_results["overall_accuracy"],
            }

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

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    # progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    best_performance = 0

    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()

            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                accelerator.clip_grad_norm_(
                    model.parameters(), max_norm=args.max_clip_grad_norm
                )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                # progress_bar.update(1)
                completed_steps += 1

            # evaluate every args.eval_steps
            # -------------- Evaluation --------------------
            if completed_steps > 0 and completed_steps % args.eval_steps == 0:
                logger.info(
                    f"***** Running Evaluation at step {completed_steps}, current epoch: {epoch} *****"
                )
                model.eval()
                for step, batch in enumerate(eval_dataloader):
                    with torch.no_grad():
                        outputs = model(**batch)
                    # logits is a tuple of (intent_logits, slot_logits)
                    intent_logits, slot_logits = outputs.logits

                    intent_predictions = intent_logits.argmax(dim=-1)

                    intent_metric.add_batch(
                        predictions=accelerator.gather(intent_predictions),
                        references=accelerator.gather(batch["intent_labels"]),
                    )

                    slot_predictions = slot_logits.argmax(dim=-1)
                    slot_labels = batch["slot_labels"]

                    if (
                        not args.pad_to_max_length
                    ):  # necessary to pad predictions and labels for being gathered
                        slot_predictions = accelerator.pad_across_processes(
                            slot_predictions, dim=1, pad_index=-100
                        )
                        slot_labels = accelerator.pad_across_processes(
                            slot_labels, dim=1, pad_index=-100
                        )

                    slot_predictions_gathered = accelerator.gather(slot_predictions)
                    slot_labels_gathered = accelerator.gather(slot_labels)
                    slot_preds, slot_refs = get_slot_labels(
                        slot_predictions_gathered, slot_labels_gathered
                    )
                    slot_metric.add_batch(
                        predictions=slot_preds,
                        references=slot_refs,
                    )  # predictions and preferences are expected to be a nested list of labels, not label_ids

                eval_metric = compute_metrics()
                cur_performance = (
                    eval_metric["overall_f1"] + eval_metric["intent_accuracy"]
                ) / 2
                accelerator.print(
                    f"training loss: {loss}, evaluation ave_acc_f1: {cur_performance}"
                )
                accelerator.print(f"epoch {epoch}:", eval_metric)

                if cur_performance > best_performance:
                    logger.info("Best performance to date, saving the model")

                    # overwrite saved_models based on the best_performance
                    best_performance = cur_performance

                    if args.output_dir is not None:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            args.output_dir, save_function=accelerator.save
                        )
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(args.output_dir)

            # -------------- End of Evaluation --------------------

            if completed_steps >= args.max_train_steps:
                break

    # End of Training loop Evaluation
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        # logits is a tuple of (intent_logits, slot_logits)
        intent_logits, slot_logits = outputs.logits

        intent_predictions = intent_logits.argmax(dim=-1)

        intent_metric.add_batch(
            predictions=accelerator.gather(intent_predictions),
            references=accelerator.gather(batch["intent_labels"]),
        )

        slot_predictions = slot_logits.argmax(dim=-1)
        slot_labels = batch["slot_labels"]

        if (
            not args.pad_to_max_length
        ):  # necessary to pad predictions and labels for being gathered
            slot_predictions = accelerator.pad_across_processes(
                slot_predictions, dim=1, pad_index=-100
            )
            slot_labels = accelerator.pad_across_processes(
                slot_labels, dim=1, pad_index=-100
            )

        slot_predictions_gathered = accelerator.gather(slot_predictions)
        slot_labels_gathered = accelerator.gather(slot_labels)
        slot_preds, slot_refs = get_slot_labels(
            slot_predictions_gathered, slot_labels_gathered
        )
        slot_metric.add_batch(
            predictions=slot_preds,
            references=slot_refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    eval_metric = compute_metrics()
    accelerator.print(f"Final Evaluation:", eval_metric)
    cur_performance = (eval_metric["overall_f1"] + eval_metric["intent_accuracy"]) / 2

    if cur_performance > best_performance:
        # overwrite saved_models based on the best_performance
        logger.info(f"Best performance: {cur_performance}, saving the model")

        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Train and Evaluation done! Best performance: {best_performance}")

    with open(os.path.join(args.output_dir, f"eval_results.json"), "w") as fp:
        json.dump(
            {"eval_overall_performance": best_performance}, fp, cls=NpEncoder, indent=4
        )

    # ***************** prediction *************************
    if args.do_predict:
        logger.info("***** Running Prediction *****")
        with accelerator.main_process_first():
            predict_dataset = raw_datasets["test"].map(
                tokenize_and_align_labels,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on Test dataset",
            )

        # load from the last saved model

        config = AutoConfig.from_pretrained(
            args.output_dir,
            num_labels=num_slot_labels,
            label2id=slot_label_to_id,
            id2labels={i: l for l, i in slot_label_to_id.items()},
            cache_dir=None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.output_dir,
            cache_dir=None,
            use_fast=True,
        )

        # We are using xlmr-based model
        # adding intent labels here
        predict_model = XLMRobertaForTokenAndIntentClassification.from_pretrained(
            args.output_dir,
            from_tf=bool(".ckpt" in args.output_dir),
            config=config,
            num_intent_labels=int(num_intent_labels),
            cache_dir=None,
        )

        predict_model = accelerator.prepare(predict_model)

        predict_model.eval()
        for lang in predict_dataset:

            predict_dataloader = DataLoader(
                predict_dataset[lang],
                collate_fn=data_collator,
                batch_size=args.per_device_eval_batch_size,
            )
            predict_dataloader = accelerator.prepare(predict_dataloader)

            for step, batch in enumerate(predict_dataloader):
                with torch.no_grad():
                    outputs = predict_model(**batch)
                # logits is a tuple of (intent_logits, slot_logits)
                intent_logits, slot_logits = outputs.logits

                intent_predictions = intent_logits.argmax(dim=-1)
                intent_metric.add_batch(
                    predictions=accelerator.gather(intent_predictions),
                    references=accelerator.gather(batch["intent_labels"]),
                )

                slot_predictions = slot_logits.argmax(dim=-1)
                slot_labels = batch["slot_labels"]

                if (
                    not args.pad_to_max_length
                ):  # necessary to pad predictions and labels for being gathered
                    slot_predictions = accelerator.pad_across_processes(
                        slot_predictions, dim=1, pad_index=-100
                    )
                    slot_labels = accelerator.pad_across_processes(
                        slot_labels, dim=1, pad_index=-100
                    )

                slot_predictions_gathered = accelerator.gather(slot_predictions)
                slot_labels_gathered = accelerator.gather(slot_labels)
                slot_preds, slot_refs = get_slot_labels(
                    slot_predictions_gathered, slot_labels_gathered
                )
                slot_metric.add_batch(
                    predictions=slot_preds,
                    references=slot_refs,
                )  # predictions and preferences are expected to be a nested list of labels, not label_ids

            predict_metric_lang = compute_metrics()
            with open(os.path.join(args.output_dir, f"predict-{lang}.json"), "w") as fp:
                json.dump(predict_metric_lang, fp, cls=NpEncoder, indent=4)


if __name__ == "__main__":
    main()
