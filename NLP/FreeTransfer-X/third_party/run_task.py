# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE


import argparse
import glob
import logging
import os
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
SIMPLE_MODELS = ["cnn", "bilstm", "mlp"] # 210918
BATCHNORM_MODELS = ["bilstm"] # 211130: `batchnorm` requires len(batch) > 1

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  BertConfig,
  BertForSequenceClassification,
  BertTokenizer,
  XLMConfig,
  XLMForSequenceClassification,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaForSequenceClassification,
  get_linear_schedule_with_warmup,
  BertForTokenClassification,
  XLMRobertaForTokenClassification,
  XLMForTokenClassification,
)
# from xlm import XLMForTokenClassification # deprecated

from utils.models import (
  BiLSTMForSequenceClassification,
  CNNForSequenceClassification,
  MLPForSequenceClassification,
  BiLSTMForTokenClassification,
  CNNForTokenClassification,
  MLPForTokenClassification,
)

from utils.models_config import (
  BiLSTMConfig,
  CNNConfig,
  MLPConfig,
)

from processors.utils import convert_examples_to_features, convert_examples_to_features_pair, parse_single_batch
from processors.s_cls import MtopSClsProcessor, MatisSClsProcessor
from processors.tag import MtopTagProcessor, MatisTagProcessor

from processors.utils_tag import convert_examples_to_features as convert_examples_to_features_tag
from processors.utils_tag import parse_single_batch_tag, save_predictions
from seqeval.metrics import precision_score, recall_score, f1_score

from utils.tools import show_model_scale

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

# NOTE: transformers==2.0.0, deprecated
# ALL_MODELS = sum(
#   (tuple(conf.pretrained_config_archive_map.keys()) 
#     for conf in (BertConfig, XLMConfig, XLMRobertaConfig)),
#   ()
# )
ALL_MODELS = []

SCLS_MODEL_CLASSES = {
  "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
  "bilstm": (BiLSTMConfig, BiLSTMForSequenceClassification, BertTokenizer),
  "mlp": (MLPConfig, MLPForSequenceClassification, BertTokenizer),
  "cnn": (CNNConfig, CNNForSequenceClassification, BertTokenizer),
}


TAG_MODEL_CLASSES = {
  "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
  "xlm": (XLMConfig, XLMForTokenClassification, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForTokenClassification, XLMRobertaTokenizer),
  "bilstm": (BiLSTMConfig, BiLSTMForTokenClassification, BertTokenizer),
  "mlp": (MLPConfig, MLPForTokenClassification, BertTokenizer),
  "cnn": (CNNConfig, CNNForTokenClassification, BertTokenizer),
}

PROCESSORS = {
  'mtop-s_cls': MtopSClsProcessor,
  'm_atis-s_cls': MatisSClsProcessor,
  'mtop-seq_tag': MtopTagProcessor,
  'm_atis-seq_tag': MatisTagProcessor,
}


def compute_metrics(preds, labels):
  scores = {
    "acc": (preds == labels).mean(), 
    "num": len(
      preds), 
    "correct": (preds == labels).sum()
  }
  return scores


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, lang2id=None, processor=None, label_list=None):
  """Train the model."""
  if args.local_rank in [-1, 0]:
    # tb_writer = SummaryWriter()
    tb_writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard_results")) # 211115

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
    {
      "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
      "weight_decay": args.weight_decay,
    },
    {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

  if args.lr_find_params: # 211118: search lr
    from utils.torch_lr_finder.lr_finder import LRFinder
    num_iter, end_lr = args.lr_find_params.split(',')
    num_iter, end_lr = int(num_iter), float(end_lr)
    logger.info("***** Searching LR *****")
    logger.info(f"  num_iter = {num_iter}, start_lr = {args.learning_rate}, end_lr = {end_lr}")
    # create dataloader
    lr_train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    lr_train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    lr_train_dataloader = DataLoader(train_dataset, sampler=lr_train_sampler, batch_size=lr_train_batch_size)

    lr_eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, processor, split=args.dev_split, language=args.train_language, lang2id=lang2id, evaluate=True)
    lr_eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    lr_eval_sampler = SequentialSampler(lr_eval_dataset)
    lr_eval_dataloader = DataLoader(lr_eval_dataset, sampler=lr_eval_sampler, batch_size=lr_eval_batch_size)

    criterion = None
    lr_finder = LRFinder(model, optimizer, criterion, device=args.device,
        model_type=args.model_type, sent_cls=args.sent_cls, task_type=args.task_type,
        )

    lr_finder.range_test(lr_train_dataloader, val_loader=lr_eval_dataloader, end_lr=end_lr, num_iter=num_iter, step_mode="linear")
    _, suggested_lr = lr_finder.plot(log_lr=False, save_path=os.path.join(args.output_dir, "lr_finder.png")) # save to disk
    lr_finder.reset()
    if suggested_lr > 0:
      prev_lr = args.learning_rate
      args.learning_rate = suggested_lr
      for g in optimizer.param_groups:
        g['lr'] = args.learning_rate
      logger.info(f"  suggested {suggested_lr}, updated lr. {prev_lr} -> {args.learning_rate}")
    else:
      logger.info(f"  suggested {suggested_lr}, failed to search. lr = {args.learning_rate}")

  scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
  )

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    # print(" [run_classify.py:150] loaded optimizer.state_dict() = {}".format(optimizer.state_dict()))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.isfile(args.model_name_or_path):
    # set global_step to gobal_step of last saved checkpoint from model path
    global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    logger.info("  Continuing training from epoch %d", epochs_trained)
    logger.info("  Continuing training from global step %d", global_step)
    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

  best_score = 0
  best_checkpoint = None
  tr_loss, logging_loss = 0.0, 0.0
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  set_seed(args)  # Added here for reproductibility
  for _ in train_iterator:
    # epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    epoch_iterator = train_dataloader # FIXME: 210731
    for step, batch in enumerate(epoch_iterator):
      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      if args.model_type in BATCHNORM_MODELS and len(batch[0]) < 2: # 211130: `batchnorm` requires len(batch[0]) > 1
        logger.info(f" [train] {model_type} skips the {step}th batch: len(batch[0]) = {len(batch[0])}")
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)

      if args.task_type == "s_cls":
        inputs = parse_single_batch(batch, args.model_type, args.sent_cls) # 211118
      elif args.task_type == "seq_tag": # 211202 
        inputs = parse_single_batch_tag(batch, args.model_type)
      
      outputs = model(**inputs)
      loss = outputs[0]

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logger.info(" [seed-{}] global_step = {}, lr = {}, loss = {}".format(
            args.seed, global_step, scheduler.get_lr()[0], (tr_loss - logging_loss) / args.logging_steps
            ))
          logging_loss = tr_loss

          # Only evaluate on single GPU otherwise metrics may not average well
          if (args.local_rank == -1 and args.evaluate_during_training):  
            results = evaluate(args, model, tokenizer, split=args.train_split, language=args.train_language, lang2id=lang2id, processor=processor, label_list=label_list)
            for key, value in results.items():
              tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.task_type == "s_cls":
            metrics = "acc"
          elif args.task_type == "seq_tag": # 211202 
            metrics = "f1"

          if args.eval_test_set:
            output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
            total = total_correct = 0.0
            with open(output_predict_file, 'a') as writer:
              writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
              for language in args.predict_languages.split(','):
                result = evaluate(args, model, tokenizer, split=args.test_split, language=language, lang2id=lang2id, prefix='checkpoint-'+str(global_step), processor=processor, label_list=label_list)
                writer.write('{}={}\n'.format(language, result[metrics]))
                if args.task_type == "s_cls":
                  total += result['num']
                  total_correct += result['correct']
              if args.task_type == "s_cls":
                writer.write('total={}\n'.format(total_correct / total))

          if args.save_only_best_checkpoint:          
            result = evaluate(args, model, tokenizer, split='dev', language=args.train_language, lang2id=lang2id, prefix=str(global_step), processor=processor, label_list=label_list)

            logger.info(" [seed-{}] Dev accuracy {} = {}".format(args.seed, args.train_language, result[metrics]))

            if result[metrics] > best_score:
              logger.info(" result[{}]={} > best_score={}".format(metrics, result[metrics], best_score))
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              best_score = result[metrics]
              # Save model checkpoint
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              model_to_save = (
                model.module if hasattr(model, "module") else model
              )  # Take care of distributed/parallel training
              model_to_save.save_pretrained(output_dir) # ACTUAL: torch.save(model_to_save.state_dict(), os.path.join(output_dir, pytorch_model.bin))
              tokenizer.save_pretrained(output_dir)
              logger.info(" save model: hasattr(model, 'module') = {}".format(hasattr(model, "module")))
              logger.info("   -- model == model_to_save: {}".format(model == model_to_save))
              logger.info("   -- model = {}".format(model))
              logger.info("   -- model_to_save = {}".format(model_to_save))
              # print(" [run_classify.py:302] saved model_to_save.state_dict() = {}".format(model_to_save.state_dict()))

              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving model args to %s", output_dir)

              torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")) # FIXME: `UserWarning: Please also save or load the state of the optimizer when saving or loading the scheduler.`
              # logger.info(" optimizer.state_dict() = {}".format(optimizer.state_dict()))
              torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
              # logger.info(" scheduler = {}".format(scheduler))
              logger.info("Saving optimizer and scheduler states to %s", output_dir)

          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            model_to_save = (
              model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step, best_score, best_checkpoint


def evaluate(args, model, tokenizer, split='train', language='en', lang2id=None, prefix="", output_file=None, label_list=None, output_only_prediction=True, processor=None):
  """Evalute the model."""
  # eval_task_names = (args.task_name,)
  # eval_outputs_dirs = (args.output_dir,)
  # for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
  eval_task = args.task_name
  eval_output_dir = args.output_dir

  results = {}
  eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, processor, split=split, language=language, lang2id=lang2id, evaluate=True)

  if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(eval_output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
  # Note that DistributedSampler samples randomly
  eval_sampler = SequentialSampler(eval_dataset)
  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu eval
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation {} {} *****".format(prefix, language))
  logger.info("  Num examples = %d", len(eval_dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)
  eval_loss = 0.0
  nb_eval_steps = 0
  preds = None
  out_label_ids = None
  sentences = None
  # for batch in tqdm(eval_dataloader, desc="Evaluating"):
  for batch in eval_dataloader: # FIXME: 210731
    model.eval()
    batch = tuple(t.to(args.device) for t in batch)

    with torch.no_grad():
      if args.task_type == "s_cls":
        inputs = parse_single_batch(batch, args.model_type, args.sent_cls) # 211118
      elif args.task_type == "seq_tag": # 211202 
        inputs = parse_single_batch_tag(batch, args.model_type)
      outputs = model(**inputs)
      tmp_eval_loss, logits = outputs[:2]

      eval_loss += tmp_eval_loss.mean().item()

    nb_eval_steps += 1
    if preds is None:
      preds = logits.detach().cpu().numpy()
      out_label_ids = inputs["labels"].detach().cpu().numpy()
      sentences = inputs["input_ids"].detach().cpu().numpy()
    else:
      preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
      out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
      sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)

  eval_loss = eval_loss / nb_eval_steps
  if args.output_mode == "classification":
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)
    if output_file:
      logger.info("***** Save prediction ******")
      with open(output_file, 'w') as fout:
        pad_token_id = tokenizer.pad_token_id
        sentences = sentences.astype(int).tolist()
        sentences = [[w for w in s if w != pad_token_id]for s in sentences]
        sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
        #fout.write('Prediction\tLabel\tSentences\n')
        for p, l, s in zip(list(preds), list(out_label_ids), sentences):
          s = ' '.join(s)
          if label_list:
            p = label_list[p]
            l = label_list[l]
          if output_only_prediction:
            fout.write(str(p) + '\n')
          else:
            fout.write('{}\t{}\t{}\n'.format(p, l, s))

  elif args.output_mode == "tagging":
    preds = np.argmax(preds, axis=2)
    label_map = {i: label for i, label in enumerate(label_list)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
        if out_label_ids[i, j] != args.pad_token_label_id:
          out_label_list[i].append(label_map[out_label_ids[i][j]])
          preds_list[i].append(label_map[preds[i][j]])

    results = {
      "loss": eval_loss,
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list)
    }
    if output_file:
      logger.info("***** Save prediction ******")
      # infile = os.path.join(args.data_dir, lang, "dev.{}".format(args.model_name_or_path))
      # idxfile = infile + '.idx'
      text_file = os.path.join(args.data_dir, split, f"{language}")
      idx_file = text_file + ".idx"
      save_predictions(args, preds_list, output_file, text_file, idx_file)
  else:
    raise ValueError("No other `output_mode`.")

  logger.info("***** Eval results {} {} *****".format(prefix, language))
  for key in sorted(results.keys()):
    logger.info("  %s = %s", key, str(results[key]))

  return results


def load_and_cache_examples(args, task, tokenizer, processor=None, split='train', language='en', lang2id=None, evaluate=False):
  # Make sure only the first process in distributed training process the 
  # dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  # processor = PROCESSORS[task]()
  # output_mode = "classification"
  # Load data features from cache or dataset file
  lc = '_lc' if args.do_lower_case else ''
  model_name = list(filter(None, args.model_name_or_path.split("/"))).pop() # get the last non-null field of a path
  if model_name == "checkpoint-best":
    model_name = list(filter(None, args.model_name_or_path.split("/")))[-2].split(':')[-2]
  logger.info("[load_and_cache_examples] model_name = {}".format(model_name))
  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}-{}_{}_{}_{}_{}{}".format(
      str(args.seed),
      args.ratio_train_examples if 'train' in split else 1.0,
      split,
      # list(filter(None, args.model_name_or_path.split("/"))).pop(), # get the last non-null field of a path
      model_name,
      str(args.max_seq_length),
      str(task),
      str(language),
      lc,
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
    if split == 'train':
      examples = processor.get_train_examples(args.data_dir, language)
    elif split == 'trans-train':
      examples = processor.get_translate_train_examples(args.data_dir, language)
    elif split == 'trans-train_pseudo-label':
      model_split = '_'.join([split, args.model_type]) # NOTE: generated depending on different models
      examples = processor.get_examples(args.data_dir, language, model_split)
    elif split == 'trans-test':
      examples = processor.get_translate_test_examples(args.data_dir, language)
    elif split == 'dev':
      examples = processor.get_dev_examples(args.data_dir, language)
    elif split == 'pseudo_test':
      examples = processor.get_pseudo_test_examples(args.data_dir, language)
    else:
      examples = processor.get_test_examples(args.data_dir, language)

    if args.ratio_train_examples > 0 and 'train' in split : # 210730
      logger.info("Original: %d examples", len(examples))
      # print(examples[1])
      # print(examples[-1])
      examples = examples[:int(len(examples)*args.ratio_train_examples)]
      logger.info("Current: %d examples", len(examples))
      # print(examples[1])
      # print(examples[-1])

    if args.task_type == "s_cls":
      if args.model_type in SIMPLE_MODELS:
        features = convert_examples_to_features_pair(
          examples[1:], # 210728: skip headings
          tokenizer,
          label_list=label_list,
          max_length=args.max_seq_length,
          output_mode=args.output_mode,
          pad_on_left=False,
          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
          pad_token_segment_id=0,
          lang2id=lang2id,
          sent_cls=args.sent_cls, # 211101
        )
      else:
        features = convert_examples_to_features(
          # examples,
          examples[1:], # 210728: skip headings
          tokenizer,
          label_list=label_list,
          max_length=args.max_seq_length,
          output_mode=args.output_mode,
          pad_on_left=False,
          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
          pad_token_segment_id=0,
          lang2id=lang2id,
          sent_cls=args.sent_cls, # 211101
        )
    elif args.task_type == "seq_tag": # 211202
      # TODO: check tokenizer's `cls_token`, `sep_token`, `pad_token`
      # NOTE: w/o the heading line
      # NOTE: `langs` of examples are `lang_id`, instead of str of `language` in classification
      features = convert_examples_to_features_tag(examples, label_list, args.max_seq_length, tokenizer,
                          cls_token_at_end=bool(args.model_type in ["xlnet"]),
                          cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                          pad_on_left=bool(args.model_type in ["xlnet"]),
                          pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                          sep_token_extra=bool(args.model_type in ["roberta", "xlmr"]), # TODO: remove `xlmr`? keep all models the same len(seq)
                          cls_token=tokenizer.cls_token,
                          sep_token=tokenizer.sep_token,
                          pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                          pad_token_label_id=args.pad_token_label_id,
                          lang=language,
                          )

    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process the 
  # dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  if args.task_type == "s_cls": # 211202
    if args.model_type in SIMPLE_MODELS: # 210830
      all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
      if not args.sent_cls: # 211101
        all_input_ids_1 = torch.tensor([f.input_ids_1 for f in features], dtype=torch.long)
    else:
      # Convert to Tensors and build dataset
      all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
      all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
      all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if args.output_mode == "classification":
      all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
      raise ValueError("No other `output_mode` for {}.".format(args.task_name))
    if args.model_type == 'xlm':
      all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_langs)
    elif args.model_type in SIMPLE_MODELS: # 210830
      if args.sent_cls: # 211101
        dataset = TensorDataset(all_input_ids, all_labels)
      else:
        dataset = TensorDataset(all_input_ids, all_input_ids_1, all_labels)
    else:  
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
  elif args.task_type == "seq_tag": # 211202
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if args.model_type == 'xlm' and features[0].langs is not None:
      all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
      logger.info('all_langs[0] = {}'.format(all_langs[0]))
      dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_langs)
    else:
      dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

  return dataset


def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
  )
  parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(SCLS_MODEL_CLASSES.keys()) + ", ".join(TAG_MODEL_CLASSES.keys()),
  )
  parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
  )
  parser.add_argument(
    "--train_language", default="en", type=str, help="Train language if is different of the evaluation language."
  )
  parser.add_argument(
    "--predict_languages", type=str, default="en", help="prediction languages separated by ','."
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
  )
  parser.add_argument(
    "--task_name",
    default="",
    type=str,
    required=True,
    help="The task name",
  )

  # Other parameters
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )
  parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
  )
  parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
  parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
  parser.add_argument("--do_predict_dev", action="store_true", help="Whether to run prediction.")
  parser.add_argument("--init_checkpoint", type=str, default=None, help="initial checkpoint for predicting the dev set")
  parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
  )
  parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
  )
  parser.add_argument("--train_split", type=str, default="train", help="split of training set")
  parser.add_argument("--dev_split", type=str, default="dev", help="split of training set")
  parser.add_argument("--test_split", type=str, default="test", help="split of training set")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
  parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
  )
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
  )
  parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
  )
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

  parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
  parser.add_argument("--log_file", default="train", type=str, help="log file")
  parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
  parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
  )
  parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
  parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
  )
  parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
  )
  parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

  parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--eval_test_set",
    action="store_true",
    help="Whether to evaluate test set durinng training",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument(
    "--save_only_best_checkpoint", action="store_true", help="save only the best checkpoint"
  )
  parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
  parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
  # 210730
  parser.add_argument("--ratio_train_examples", type=float, default=1., help="scale down trainset, for debug")
  # 210813
  parser.add_argument("--rand_init", action="store_true", help="randomly initialize model, instead of `from_pretrained`")
  # 210913
  parser.add_argument(
    "--vocab_path",
    default=None,
    type=str,
    help="Path to vocab txt",
  )
  # 211101
  parser.add_argument("--sent_cls", action="store_true", help="single sentence classification, otherwise pair cls")
  # 211118: search hyper-params
  parser.add_argument("--lr_find_params", type=str, default="", help="comma-splitted `num_iter` and `end_lr`, search learning rate between [`learning_rate`, `end-lr`]")

  args = parser.parse_args()

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Prepare dataset
  if args.task_name not in PROCESSORS:
    raise ValueError("Task not found: %s" % (args.task_name))
  processor = PROCESSORS[args.task_name]()
  label_list = processor.get_labels()
  num_labels = len(label_list)

  dataset, args.task_type = args.task_name.split('-')
  if args.task_type == "s_cls":
    args.output_mode = "classification"
    # label2id, id2label = processor.get_labels_map() # 210731: mapping between readable-label-text & id. unnecessary?
    MODEL_CLASSES = SCLS_MODEL_CLASSES
  elif args.task_type == "seq_tag": # 211202 
    args.output_mode = "tagging"
    # Use cross entropy ignore index as padding label id
    # so that only real label ids contribute to the loss later
    args.pad_token_label_id = CrossEntropyLoss().ignore_index
    MODEL_CLASSES = TAG_MODEL_CLASSES
  else:
    raise TypeError(f" [main] unknown task_type = {args.task_type}, dataset = {dataset}")

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  # config.num_labels = num_labels
  if args.model_type in ["cnn", "mlp"]: # 210823
    if args.model_type == "cnn":
      config.max_sen_len = args.max_seq_length
    # if args.local_rank == -1 or args.no_cuda:
    #   if torch.cuda.is_available() and not args.no_cuda:
    #     config.cuda = True
    #   else:
    #     config.cuda = False
    # else:
    #   config.cuda = True

  # if args.model_type in SIMPLE_MODELS: # 210913: task- & lang-related. # 210918: simple models from .txt vocab
  if args.vocab_path and args.vocab_path != "None": # 211011: adopt `vocab_path` for all model-types, if `vocab_path` is specified
    tokenizer = tokenizer_class(
      args.vocab_path,
      do_lower_case=args.do_lower_case,
      # cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.vocab_size = len(tokenizer.get_vocab())
  else:
    tokenizer = tokenizer_class.from_pretrained(
      args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
      do_lower_case=args.do_lower_case,
      cache_dir=args.cache_dir if args.cache_dir else None,
    )

  logger.info("config = {}".format(config))

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))
  if args.task_type == "seq_tag": # 211203
    processor.set_lang2id(lang2id)

  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  logger.info("Training/evaluation parameters %s", args)

  # Training
  if args.do_train:
    if args.rand_init: # 210813
      logger.info("random initialization {}".format(model_class))
      config.sent_cls = args.sent_cls # 211101
      model = model_class(config=config)
    else:
      if args.init_checkpoint:
        logger.info("loading from folder {}".format(args.init_checkpoint))
        model = model_class.from_pretrained(
          args.init_checkpoint,
          config=config,
          cache_dir=args.init_checkpoint,
          )
      else:
        logger.info("loading from existing model {}".format(args.model_name_or_path))
        model = model_class.from_pretrained(
          args.model_name_or_path,
          from_tf=bool(".ckpt" in args.model_name_or_path),
          cache_dir=args.cache_dir if args.cache_dir else None,
          config=config, # 210731
        )
    show_model_scale(model, logger)
    model.to(args.device)
    logger.info(" init model = {}".format(model))
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, processor, split=args.train_split, language=args.train_language, lang2id=lang2id, evaluate=False)
    global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, tokenizer, lang2id, processor, label_list=label_list)
    logger.info(" [seed-%s] global_step = %s, average loss = %s", args.seed, global_step, tr_loss)
    logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
      model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    logger.info(" model_to_save = {}".format(model_to_save))
    # print(" [run_classify.py:860] model_to_save.state_dict() = {}".format(model_to_save.state_dict()))
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    # config = config_class.from_pretrained(args.output_dir) # 211101: FIXME
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)
    logger.info(" loaded model = {}".format(model))
    # print(" [run_classify.py:872] loaded model.state_dict() = {}".format(model.state_dict()))

  # Evaluation
  if args.task_type == "s_cls":
    metrics = "acc"
  elif args.task_type == "seq_tag": # 211202 
    metrics = "f1"

  results = {}
  if args.init_checkpoint:
    best_checkpoint = args.init_checkpoint
  elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_score = 0
  if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

      # config = config_class.from_pretrained(checkpoint) # 211101: FIXME
      model = model_class.from_pretrained(checkpoint)
      show_model_scale(model, logger)
      model.to(args.device)
      logger.info(" loaded model = {}".format(model))
      # print(" [run_classify.py:899] loaded model.state_dict() = {}".format(model.state_dict()))
      result = evaluate(args, model, tokenizer, split='dev', language=args.train_language, lang2id=lang2id, prefix=prefix, processor=processor, label_list=label_list)
      if result[metrics] > best_score:
        best_checkpoint = checkpoint
        best_score = result[metrics]
      result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
      results.update(result)
    
    output_eval_file = os.path.join(args.output_dir, 'eval_results')
    with open(output_eval_file, 'w') as writer:
      for key, value in results.items():
        writer.write('{} = {}\n'.format(key, value))
      writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
      logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))

  # Prediction
  if args.do_predict and args.local_rank in [-1, 0]:
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
    tokenizer = tokenizer_class.from_pretrained(best_checkpoint, do_lower_case=args.do_lower_case)
    # config = config_class.from_pretrained(best_checkpoint) # 211101: FIXME
    model = model_class.from_pretrained(best_checkpoint)
    show_model_scale(model, logger)
    model.to(args.device)
    logger.info(" loaded model = {}".format(model))
    # print(" [run_classify.py:921] loaded model.state_dict() = {}".format(model.state_dict()))
    output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
    total = total_correct = 0.0
    with open(output_predict_file, 'a') as writer:
      writer.write('======= Predict using the model from {} for {}:\n'.format(best_checkpoint, args.test_split))
      for language in args.predict_languages.split(','):
        output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
        result = evaluate(args, model, tokenizer, split=args.test_split, language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file, label_list=label_list, processor=processor)
        writer.write('{}={}\n'.format(language, result[metrics]))
        logger.info('{}={}'.format(language, result[metrics]))
        if args.task_type == "s_cls":
          total += result['num']
          total_correct += result['correct']
      if args.task_type == "s_cls":
        writer.write('total={}\n'.format(total_correct / total))

  if args.do_predict_dev:
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
    # config = config_class.from_pretrained(args.init_checkpoint) # 211101: FIXME
    model = model_class.from_pretrained(args.init_checkpoint, config=config)
    show_model_scale(model, logger)
    model.to(args.device)
    output_predict_file = os.path.join(args.output_dir, 'dev_results')
    total = total_correct = 0.0
    with open(output_predict_file, 'w') as writer:
      writer.write('======= Predict using the model from {}:\n'.format(args.init_checkpoint))
      for language in args.predict_languages.split(','):
        output_file = os.path.join(args.output_dir, 'dev-{}.tsv'.format(language))
        result = evaluate(args, model, tokenizer, split='dev', language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file, label_list=label_list, processor=processor)
        writer.write('{}={}\n'.format(language, result[metrics]))
        if args.task_type == "s_cls":
          total += result['num']
          total_correct += result['correct']
      if args.task_type == "s_cls":
        writer.write('total={}\n'.format(total_correct / total))

  return result


if __name__ == "__main__":
  main()
