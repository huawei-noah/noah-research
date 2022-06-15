#!/usr/bin/env python3
# -*- coding: utf-8

# Copyright 2021 Huawei Technologies Co., Ltd.
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

import os
import argparse
import time
import numpy as np
import random
import math
import logging
import copy
from tqdm import tqdm
import os.path

import torch
from torch.autograd import Variable
import torch.nn as nn 
from torch.utils.data import DataLoader, Dataset, SequentialSampler, TensorDataset

from PinyinCharDataProcesser import PinyinCharDataProcesser
from NEZHA.modeling_nezha import BertConfig, BertForTokenClassification
from NEZHA.optimization2 import AdamW, get_linear_schedule_with_warmup

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

def PrintModelParaNum(model):
    totalParaNum = 0
    for name, para in model.named_parameters():
        logging.info(f"{name} : {para.shape} : {para.numel()}")
        totalParaNum += para.numel()
    logging.info(f'total parameter number :{totalParaNum}')
    pass

def DumpModel2File(model2dump, optimizer2dump, scheduler2dump, config2dump, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model2dump.state_dict(), os.path.join(save_path,'model.pt'))
    config2dump.to_json_file(os.path.join(save_path,'config.json'))
    torch.save(optimizer2dump.state_dict(), os.path.join(save_path, 'optimizer.pt'))
    torch.save(scheduler2dump.state_dict(), os.path.join(save_path, 'scheduler.pt'))
    pass

def DumpModel2Ckpt(epoch, model2dump, optimizer2dump, scheduler2dump, save_path):
    logging.info(f'dumping at epoch {epoch}.....')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model2dump.config.to_json_file(os.path.join(save_path, 'model_config.json'))
    torch.save(model2dump.state_dict(), os.path.join(save_path, 'model_sd_e'+str(epoch)+'.pt'))
    torch.save(optimizer2dump.state_dict(), os.path.join(save_path, 'optimizer_sd_e'+str(epoch)+'.pt'))
    torch.save(scheduler2dump.state_dict(), os.path.join(save_path, 'scheduler_sd_e'+str(epoch)+'.pt'))
    logging.info(f'finishing dumping at epoch {epoch}.....')
    pass

def main():
    parser = argparse.ArgumentParser(description='PERT model for pinyin2char in PyTorch')
    parser.add_argument('--vocab', type=str, default='data',
                        help='location of the vocabulary')
    parser.add_argument('--pyLex', type=str, default='data',
                        help='location of the pinyin list')
    parser.add_argument('--chardata', type=str, default='',
                        help='location of the char corpus')
    parser.add_argument('--pinyindata', type=str, default='',
                        help='location of the pinyin corpus')
    parser.add_argument('--num_loading_workers', type=int, default=4,
                        help='the number of workers to loading the dataset(pinyin and chardata)')
    parser.add_argument('--bert_config', type=str, default='',
                        help='this is the config file for bert model')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='train batch size')
    parser.add_argument('--seq_length', type=int, default=16,
                        help='the sequence length for training')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--warmup_proportion", default=0.1,type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, 
                        help="Epsilon for Adam optimizer.") 
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--save', type=str,  default='./models/',
                        help='path to save the final model')
    parser.add_argument('--save_per_n_epoches', type=int,  default=1,
                        help='after how many epoches that we should dump the trained model')
    parser.add_argument('--continue_train_index', type=int,  default=0,
                        help='we continue to train the model based on the previous training results')
    parser.add_argument('--prev_model_path', type=str,  default='./models/model.pt',
                        help='the model parameters trained previously')
    parser.add_argument('--prev_optimizer_path', type=str,  default='./models/optimizer.pt',
                        help='the optimizer parameters trained previously')
    parser.add_argument('--prev_scheduler_path', type=str,  default='./models/scheduler.pt',
                        help='the scheduler parameters trained previously')
    args, _ = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Prepare data
    bert_config = BertConfig.from_json_file(args.bert_config)
    theDataProcesser = PinyinCharDataProcesser(
        vocabFile = args.vocab, 
        pinyinFile = args.pyLex,
        seq_length = bert_config.seq_length,
        pyDataFile = args.pinyindata, 
        charDataFile = args.chardata
    )
    theDataset = theDataProcesser.LoadData()
    theSampler = SequentialSampler(theDataset)
    train_dataloader = DataLoader(
        dataset = theDataset,
        batch_size = args.train_batch_size,
        shuffle = False,
        sampler = theSampler,
        num_workers = args.num_loading_workers,
        drop_last = True
        )
    total_train_examples = theDataProcesser.num_samples
    # Prepare model
    model = BertForTokenClassification(
        config = bert_config, 
        num_labels = bert_config.num_labels
        )
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    if args.continue_train_index > 0:
        model.load_state_dict(torch.load(args.prev_model_path, map_location=device))
    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = math.ceil(
        (total_train_examples * args.num_epochs) / (args.train_batch_size * args.gradient_accumulation_steps))
    optimizer = AdamW(
        params = optimizer_grouped_parameters, 
        lr = args.learning_rate, 
        eps = args.adam_epsilon
        )
    warmup_steps = math.ceil(args.warmup_proportion * num_train_optimization_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer = optimizer, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = num_train_optimization_steps
        )
    if args.continue_train_index > 0:
        optimizer.load_state_dict(torch.load(args.prev_optimizer_path, map_location=device))
        scheduler.load_state_dict(torch.load(args.prev_scheduler_path, map_location=device))

    # Training
    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info(f"  Batch size = {args.train_batch_size}")
    logging.info(f"  Max sequence length = {bert_config.seq_length}")
    logging.info(f"  Num steps = {num_train_optimization_steps}")
    start_time = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model.train()
    epoch_size = math.ceil(total_train_examples / args.train_batch_size)
    for epoch in tqdm(range(args.continue_train_index, args.num_epochs)):
        costs = 0.0
        iters = 0
        for step, batch in enumerate(train_dataloader):
            input_ids, labels = batch
            loss = model(
                input_ids = Variable(input_ids).contiguous().to(device), 
                labels = Variable(labels).contiguous().to(device)
                )
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            costs += torch.Tensor.item(loss.data) * bert_config.seq_length
            iters += bert_config.seq_length
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % (math.ceil(epoch_size / 10)) == 0:
                logging.info("{} perplexity: {:8.2f} speed: {} wps".format(step * 1.0 / epoch_size, np.exp(costs / iters),iters * args.train_batch_size / (time.time() - start_time)))
        # save the model every n epoch
        if epoch % args.save_per_n_epoches == 0:
            DumpModel2Ckpt(epoch, model, optimizer, scheduler, args.save)
    logging.info("***** Finishing training *****")
    # Dumpping
    logging.info("******** Dumping ********")
    DumpModel2File(model, optimizer, scheduler, bert_config, args.save)
    logging.info("******** Finishing dumpping ********")
    # PrintModelParaNum(model) # print the model parameters
    pass

if __name__ == "__main__":   
    print ('hello')
    main()
    print('olleh')


