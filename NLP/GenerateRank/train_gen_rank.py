# coding=utf-8
# 2021.03.10 - Changed for Generate & Rank multitask framework
#      Huawei Technologies Co., Ltd.
# Copyright 2022 Huawei Technologies Co., Ltd.
# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
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

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import random
import sys
import time

import numpy as np
import torch
import csv

from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm, trange
from GenerateRankModel import MyMBartForSequenceClassificationAndGeneration
from transformers.models.mbart.modeling_mbart import shift_tokens_right
from transformers import (MBartTokenizer, CONFIG_NAME, MBartConfig,
                         WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from utils import is_equal, clean_text
from exp_tree import corrupt_expression

NUM_STR = [str(i) for i in range(10)] + ['#']
csv.field_size_limit(sys.maxsize)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class LineDataset(Dataset):
    def __init__(self, lines):
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        prob, label = self.lines[item].strip().split('\t')
        return {"prob": prob, "label": label}

def add_rule_negatives(lines, add_num=10, all_expressions=None):
    all_lines = []
    prev_prob = ""
    for line in lines:
        prob, gen, correct, cls_label = line.strip().split('\t')
        if prob != prev_prob: # new prob begins here! generate rule negatives
            prev_prob = prob
            for _ in range(add_num):
                try:
                    rule_neg = corrupt_expression(correct)
                    this_label = '1' if is_equal(correct, rule_neg) else '0'
                    all_lines.append("\t".join([prob, rule_neg, correct, this_label]) + '\n')
                except:
                    print("error??")
                    print(correct)
        all_lines.append(line)
    return all_lines

class TextDataset(Dataset):
    def __init__(self, tokenizer, lines, max_source_length, max_target_length):
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.lines = lines
        self.tokenizer = tokenizer
        self.features = {}
        self.prepare_data()

    def prepare_data(self):
        lines = self.lines
        for i, line in enumerate(tqdm(lines, desc="Prepare train data")):

            items = line.strip().split('\t')

            prob, gen, correct, cls_label = items[0], items[1], items[2], items[3]
            #problem \t generated expression \t groundtruth \t rank_label

            if cls_label == "-1":  #prepare data for generate and revise
                assert gen == "<mask>"
                encoder_input = prob
                decoder_input = correct
                batch_encoding = self.tokenizer.prepare_seq2seq_batch(
                    src_texts=encoder_input,
                    tgt_texts=decoder_input,
                    max_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    src_lang=SRC_LANG,
                    tgt_lang=TGT_LANG,
                    padding="max_length",
                    return_tensors="pt"
                )
                batch_encoding['decoder_input_ids'] = shift_tokens_right(
                    batch_encoding['labels'], pad_token_id=self.tokenizer.pad_token_id)
                for k, v in batch_encoding.items():
                    batch_encoding[k] = v.squeeze(0)
                batch_encoding['cls_label'] = torch.tensor([-100])
            else:  #prepare data for rank function
                encoder_input = prob
                decoder_input = gen
                batch_encoding = self.tokenizer.prepare_seq2seq_batch(
                    src_texts=encoder_input,
                    tgt_texts=decoder_input,
                    max_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    src_lang=SRC_LANG,
                    tgt_lang=TGT_LANG,
                    padding="max_length",
                    return_tensors="pt"
                )
                batch_encoding['decoder_input_ids'] = shift_tokens_right(
                    batch_encoding['labels'], pad_token_id=self.tokenizer.pad_token_id)
                for k, v in batch_encoding.items():
                    batch_encoding[k] = v.squeeze(0)
                batch_encoding['labels'] = torch.full_like(batch_encoding['labels'], -100)
                batch_encoding['cls_label'] = torch.tensor([int(cls_label)])

            self.features[i] = batch_encoding


    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def collect_data_to_trainfile(args):
    samples = []
    for i in range(args.world_size):
        output_samples_file = os.path.join(args.output_dir, f"gen_samples_{i}.train")
        with open(output_samples_file) as f:
            lines = f.readlines()
            for line in lines:
                samples.append(line)
    print("writing in collect")
    with open(args.collect_file, "w") as f:
        for line in samples:
            f.write(line)
    print("write done")

def train(args, tokenizer, device):
    """ Train the model """

    with open(args.train_file) as f:
        raw_train_lines = f.readlines()
    raw_train_dataset = LineDataset(raw_train_lines)
    raw_train_sampler = torch.utils.data.distributed.DistributedSampler(
        raw_train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
    raw_train_dataloader = torch.utils.data.DataLoader(raw_train_dataset, batch_size=16,
                                                   sampler=raw_train_sampler, drop_last=False)

    with open(args.valid_file) as f:
        valid_lines = f.readlines()
    with open(args.test_file) as f:
        test_lines = f.readlines()

    print(f"load model from {args.model_path}")
    config = MBartConfig.from_pretrained(args.model_path)
    config.num_labels = 2
    config.id2label = {"0": "LABEL_0", "1": "LABEL_1"}
    config.label2id = {"LABEL_0": 0, "LABEL_1": 1}
    model = MyMBartForSequenceClassificationAndGeneration.from_pretrained(args.model_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    print("model load done")

    config = model.config
    model.to(device)

    # model size
    size = 0
    for n, p in model.named_parameters():
        size += p.nelement()
    logger.info('Total parameters: {}'.format(size))

    # Training
    # generate samples for revise
    output_samples_file = os.path.join(args.output_dir, f"gen_samples_{args.local_rank}.train")
    args.collect_file = os.path.join(args.output_dir, args.collect_file)
    print("start generate samples")
    gen_start_time = time.time()
    generate_sample(model, tokenizer, raw_train_dataloader, device, output_samples_file, args)
    print(f"finish generate samples in {time.time() - gen_start_time}")

    torch.distributed.barrier()

    if args.local_rank == 0:
        print("collecting train data")
        collect_data_to_trainfile(args)
    torch.distributed.barrier()
    with open(args.collect_file) as f:
        collect_lines = f.readlines()
    all_expressions = None
    if args.rule_negatives:
        collect_lines = add_rule_negatives(collect_lines, add_num=args.num_negatives, all_expressions=all_expressions)
    train_dataset = TextDataset(tokenizer, collect_lines, args.max_source_length, args.max_target_length)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size,
                                                   sampler=train_sampler, drop_last=False)


    output_logging_file = os.path.join(args.output_dir, "log.txt")
    logger.info(f"len dataloader: {len(train_dataloader)}\n")
    logger.info(f"local rank: {args.local_rank}\n")
    logger.info(f"rank: {args.rank}\n")

    t_total = int(len(train_dataloader) * args.num_train_epochs)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    logger.info("args.pos_weight_decay is None!")
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                           output_device=args.local_rank)


    # Train!
    global_step = 0
    tr_loss = tr_cls_loss = tr_mlm_loss = 0.0

    epoch = 0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    best_valid_acc = best_valid_epoch = 0
    valid_rank_acc = valid_acc = test_rank_acc = test_acc = test_acc_all = valid_acc_all = -1
    for _ in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        train_start_time = time.time()
        for step, inputs in enumerate(epoch_iterator):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            model.train()
            outputs = model(**inputs)

            mlm_loss = outputs["loss"]
            cls_loss = outputs["cls_loss"]

            mlm_loss = mlm_loss.mean() if mlm_loss is not None else torch.tensor(0)  # mean() to average on multi-gpu parallel training
            cls_loss = cls_loss.mean() if cls_loss is not None else torch.tensor(0)  # mean() to average on multi-gpu parallel training
            loss = mlm_loss + cls_loss

            loss.backward()

            tr_loss += loss.item()
            tr_cls_loss += cls_loss.item()
            tr_mlm_loss += mlm_loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0 and args.rank==0:
                result = {}
                result['train_loss'] = tr_loss / args.logging_steps
                result['mlm_loss'] = tr_mlm_loss / args.logging_steps
                result['cls_loss'] = tr_cls_loss / args.logging_steps
                result['global_step'] = global_step
                result['epoch'] = epoch
                tr_loss = tr_mlm_loss = tr_cls_loss = 0.0
                with open(output_logging_file, "a") as writer:
                    logger.info("***** Eval results *****")
                    for key in sorted(result.keys()):
                        logger.info("  %s = %s", key, str(result[key]))
                        writer.write("%s = %s\n" % (key, str(result[key])))
                    writer.write("\n")

        # epoch end
        train_end_time = time.time()
        if args.rank == 0 and epoch % args.test_per_epoch == 0:

            result = {}
            result['global_step'] = global_step
            result['epoch'] = epoch
            test_start_time = time.time()
            test_output_dir = os.path.join(args.output_dir, '{}-{}'.format("test_output", global_step))
            if not os.path.exists(test_output_dir):
                os.makedirs(test_output_dir)
            valid_output_file = os.path.join(test_output_dir, "output_gen.valid")

            valid_acc, valid_acc_all, valid_total = gen_test(model, tokenizer, valid_lines, test_file=valid_output_file)
            valid_acc = valid_acc / valid_total
            valid_acc_all = valid_acc_all / valid_total

            with open(valid_output_file) as f:
                valid_rank_lines = f.readlines()
            valid_rank_acc = genrank_test(args, model, tokenizer, valid_rank_lines, tokenizer.pad_token_id)

            test_end_time = time.time()
            result['valid_acc'] = valid_acc
            result['valid_acc_all'] = valid_acc_all
            result['valid_rank_acc'] = valid_rank_acc
            if (valid_rank_acc > best_valid_acc):
                best_valid_acc = valid_acc if valid_rank_acc == -1 else valid_rank_acc
                best_valid_epoch = epoch
                test_output_dir = os.path.join(args.output_dir, '{}-{}'.format("test_output", global_step))
                if not os.path.exists(test_output_dir):
                    os.makedirs(test_output_dir)
                test_output_file = os.path.join(test_output_dir, "output_gen.test")

                test_acc, test_acc_all, test_total = gen_test(model, tokenizer, test_lines, test_file=test_output_file)
                test_acc = test_acc / test_total
                test_acc_all = test_acc_all / test_total

                with open(test_output_file) as f:
                    test_rank_lines = f.readlines()
                test_rank_acc = genrank_test(args, model, tokenizer, test_rank_lines, tokenizer.pad_token_id)

                logging.info("** ** * Saving fine-tuned model ** ** * ")
                # Only save the model it-self
                output_dir = os.path.join(args.output_dir, "saved_model")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                model_to_save = model.module if hasattr(model, 'module') else model

                model_name = WEIGHTS_NAME

                output_model_file = os.path.join(output_dir, model_name)
                output_config_file = os.path.join(output_dir, CONFIG_NAME)

                torch.save(model_to_save.state_dict(), output_model_file)
                config.to_json_file(output_config_file)
                tokenizer.save_pretrained(output_dir)

            result['test_acc_at_best_valid'] = test_acc
            result['test_acc_all_at_best_valid'] = test_acc_all
            result['test_rank_acc_at_best_valid'] = test_rank_acc
            result['best_valid_acc'] = best_valid_acc
            result['best_valid_epoch'] = best_valid_epoch
            result['train_epoch_time'] = train_end_time - train_start_time
            result['test_time'] = test_end_time - test_start_time

            with open(output_logging_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n")

        torch.distributed.barrier()
        # generate samples for revise
        if args.regenerate == 1:
            output_samples_file = os.path.join(args.output_dir, f"gen_samples_{args.local_rank}.train")
            generate_sample(model, tokenizer, raw_train_dataloader, device, output_samples_file, args)

            if args.local_rank == 0:
                print("collecting train data")
                collect_data_to_trainfile(args)
            torch.distributed.barrier()
            with open(args.collect_file) as f:
                collect_lines = f.readlines()
            if args.rule_negatives:
                collect_lines = add_rule_negatives(collect_lines, add_num=args.num_negatives, all_expressions=all_expressions)
            train_dataset = TextDataset(tokenizer, collect_lines, args.max_source_length, args.max_target_length)
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, num_replicas=args.world_size, rank=args.rank)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size,
                                                           sampler=train_sampler, drop_last=False)


    return global_step, tr_loss / global_step

def generate_sample(model, tokenizer, dataloader, device, output_samples_file, args):
    '''
    generate negative samples using the model for revise training
    '''
    samples = []
    if args.use_generate_sample:
        model = model.module if hasattr(model, "module") else model
        model.eval()
        beam_size = args.num_negatives
        for data in tqdm(dataloader):
            prob, label = data['prob'], data['label']
            gen_prob = prob
            batch = tokenizer.prepare_seq2seq_batch(gen_prob, src_lang="zh_CN", return_tensors="pt")
            for k, v in batch.items():
                batch[k] = v.to(device)

            text = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                                  num_beams=beam_size, early_stopping=True, max_length=args.max_target_length,
                                  num_return_sequences=beam_size) # batch * 10, len
            text = tokenizer.batch_decode(text, skip_special_tokens=True)
            text = [clean_text(t) for t in text]

            label = [clean_text(t) for t in label]

            idx = 0
            for p, e in zip(prob, label):
                samples.append((p, "<mask>", e, 0))
                samples.append((p, e, e, 1))
                beam = text[idx*beam_size: (idx+1)*beam_size]
                for b in beam:
                    if is_equal(e, b):
                        samples.append((p, b, b, 1))
                    else:
                        samples.append((p, b, e, 0))
                idx += 1
    else:
        for data in tqdm(dataloader):
            prob, label = data['prob'], data['label']
            label = [clean_text(t) for t in label]
            for p, e in zip(prob, label):
                samples.append((p, "<mask>", e, 0))
                samples.append((p, e, e, 1))

    with open(output_samples_file, "w") as f:
        for sample in samples:
            if sample[1] != "<mask>":
                f.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\t{sample[3]}\n")  # for ranker
            else:
                f.write(f"{sample[0]}\t{sample[1]}\t{sample[2]}\t-1\n")  # for generation

    return samples



def genrank_test(args, model, tokenizer, lines, pad_token_id):
    print("rank testing..........")
    model = model.module if hasattr(model, "module") else model
    model.eval()
    #group generated expressions as batches for each problem
    prob_prev = ""
    all_batch = []
    batch = []
    acc = 0
    for i, line in enumerate(lines):
        line_splited = line.strip().split('\t')
        if len(line_splited) == 3:
            prob, exp, label = line_splited
        else:
            prob, exp, label, groundtruth = line_splited
        if prob != prob_prev:  #new problem begins. add batch to all_batch and reset batch.
            prob_prev = prob
            if i != 0:
                all_batch.append(batch)
            batch = []

        #add cur batch_encoding to batch
        batch_encoding = tokenizer.prepare_seq2seq_batch(
            src_texts=prob,
            tgt_texts=exp,
            max_length=args.max_source_length,
            max_target_length=args.max_target_length,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            padding="max_length",
            return_tensors="pt"
        )

        batch_encoding["decoder_input_ids"] = shift_tokens_right(batch_encoding["labels"], pad_token_id)
        for k, v in batch_encoding.items():
            batch_encoding[k] = v.squeeze(0)
        batch_encoding["cls_label"] = torch.tensor([1])
        batch_encoding.pop("labels")
        batch_encoding["answer_label"] = label
        batch.append(batch_encoding)
    all_batch.append(batch)

    #now all_batch is a list of batches, we go through it and compute test accuracy
    for i, batch_encoding in enumerate(tqdm(all_batch)):
        keys = batch_encoding[0].keys()
        batch = {}
        for k in keys:
            if k == "answer_label":
                labels = [b[k] for b in batch_encoding]
            else:
                batch[k] = torch.stack([b[k] for b in batch_encoding]).cuda()

        output = model(**batch)
        output = output["cls_logits"]
        output = torch.nn.functional.softmax(output)[:,1]
        index = torch.argmax(output, dim=0).item()  #choose the top 1 as the predicted answer
        if labels[index] == "1":  #labels record whether this expression is positive or not
            acc += 1
    return acc * 100.0 / len(all_batch)

def gen_test(model, tokenizer, lines, num_beam=10, num_return_sequences=10, test_file=None):
    print("gen testing ........")
    model = model.module if hasattr(model, "module") else model
    model.eval()
    acc = acc_all = 0
    total = len(lines)
    rank_samples = []
    for i, line in enumerate(tqdm(lines)):
        problem, label = line.strip().split("\t")
        hit_acc = hit_1 = False
        batch = tokenizer.prepare_seq2seq_batch(problem, src_lang=SRC_LANG, return_tensors="pt")
        for k,v in batch.items():
            batch[k] = v.cuda()
        text = model.generate(**batch, decoder_start_token_id=tokenizer.lang_code_to_id["en_XX"],
                              num_beams=num_beam, early_stopping=True, max_length=100, num_return_sequences=num_return_sequences)

        text = tokenizer.batch_decode(text, skip_special_tokens=True)
        text = [clean_text(t) for t in text]

        for idx, candidate in enumerate(text):
            if is_equal(label, candidate):
                hit_acc = True
                if idx == 0:
                    hit_1 = True
            rank_label = '1' if is_equal(label, candidate) else '0'
            rank_samples.append([problem, candidate, rank_label, label])

        if hit_acc:
            acc_all += 1
        if hit_1:
            acc += 1
    with open(test_file, 'w') as f:
        for samp in rank_samples:
            f.write("\t".join(samp) + "\n")

    return acc, acc_all, total

def main(args):
    global SRC_LANG, TGT_LANG
    SRC_LANG = args.src_lang
    TGT_LANG = args.tgt_lang
    assert (torch.cuda.is_available())
    device_count = torch.cuda.device_count()

    # Manually set the device ids.
    # if device_count > 0:
    # args.local_rank = args.rank % device_count
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    print('device_id (local_rank): %s' % args.local_rank)
    print('device_count: %s, rank: %s, world_size: %s' % (device_count, args.rank, args.world_size))
    torch.distributed.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    print("init done")
    if args.rank == 0 and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    torch.distributed.barrier()

    if not os.path.exists(args.output_dir) and args.rank == 0:
        os.makedirs(args.output_dir)

    args.n_gpu = torch.cuda.device_count()
    logger.info("args: {}".format(args))

    # Set seed
    set_seed(args)

    # Load pretrained tokenizer
    tokenizer = MBartTokenizer.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
    new_tokens = [f"#{i}" for i in range(30)]
    tokenizer.add_tokens(new_tokens)

    train(args, tokenizer, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--valid_file", default=None, type=str, required=True,
                        help="The input validing data file (a text file).")
    parser.add_argument("--test_file", default=None, type=str, required=True,
                        help="The input testing data file (a text file).")
    parser.add_argument("--collect_file", default="collect_file.txt", type=str)

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_path", default=None, type=str, required=True,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=32)
    parser.add_argument("--regenerate", type=int, default=1)
    parser.add_argument("--rule_negatives", type=int, default=0)
    parser.add_argument("--num_negatives", type=int, default=10)
    parser.add_argument("--use_generate_sample", type=int, default=1)

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2.0, type=float,
                        help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--test_per_epoch', type=int, default=5,
                        help="Test every X epochs.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="For distributed training: local_rank")

    parser.add_argument("--data_url", type=str, default="", help="s3 url")
    parser.add_argument("--train_url", type=str, default="", help="s3 url")
    parser.add_argument("--src_lang", default='zh_CN', type=str)
    parser.add_argument("--tgt_lang", default='en_XX', type=str)

    args = parser.parse_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    main(args)

