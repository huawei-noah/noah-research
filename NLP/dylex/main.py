# coding=utf-8
# Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

from pathlib import Path
import shutil
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertConfig
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

from finetune_argparse import get_argparse
from common import init_logger, logger, seed_everything, print_config
from trie import SpaceTrieTree
from ner_dict import CmoProcessor, file2list
from bert_dict_for_ner import BertDictForNerSoftmax
from utils_ner import CNerTokenizer
from ner_dict import convert_examples_to_features
from ner_metrics import SeqEntityScore, DictionaryScore, IntentScore, SentenceScore

def train(args, train_dataset, model, tokenizer,ner_label_list, trietree, itos_dict):
    """ Train model """
    n_gpu = torch.cuda.device_count()
    n_gpu = n_gpu if n_gpu > 0 else 1
    args.train_batch_size = args.train_bs_per_gpu * n_gpu
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    bert_param = [x for x in list(model.named_parameters()) if "bert" in x[0]]
    other_param = [x for x in list(model.named_parameters()) if "bert" not in x[0]]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            "lr": args.lr
        },
        {
            'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            "lr": args.lr
        },

        {
            'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
            "lr": args.other_lr
        },
        {
            'params': [p for n, p in other_param if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
            "lr": args.other_lr
        }
    ] # 相比simple把参数分成了bert和other
    args.warmup_steps = int(t_total * args.warmup_proportion)
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss = 0.0
    best_epoch = -1
    global_best_f = 0.0  # 记录dev集上最好F1
    global_test_f = 0.0  # 记录test集上此时的F
    global_test_dict_f = 0.0  # 记录test集上的dict消歧F
    for cur_epoch in range(int(args.num_train_epochs)):
        for _, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0], "attention_mask": batch[2], "token_type_ids": batch[3],
                "labels": batch[4], "input_len": batch[7],
                "name_ids": batch[1], "dict_labels": batch[5], "dict_mask": batch[6], "loss_mask": batch[-1]
            }
            outputs = model(**inputs)
            loss = outputs[0]
            if n_gpu > 1: loss = loss.mean()

            loss.backward()
            tr_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            if (global_step + 1) % args.log_per_n == 0:
                logger.info(f"epoch {cur_epoch}, step {global_step}, loss {tr_loss / global_step:5f}")

            if (global_step + 1) % args.eval_per_n == 0:
                ner_f, _ = evaluate(args, model, tokenizer,ner_label_list ,trietree, itos_dict, data_type="dev")
                test_f, test_dict_f = evaluate(args, model, tokenizer, ner_label_list, trietree, itos_dict, data_type="test")
                logger.info(f"Best test CWS F1 {global_test_f:.5f} @{best_epoch}")
                if ner_f > global_best_f:
                    torch.save(model, args.model_save_dir / "model.pb")
                    global_best_f = ner_f  # 更新最优结果
                    best_epoch = cur_epoch
                    global_test_f, global_test_dict_f = test_f, test_dict_f

    logger.info(f"final wo get best result:")
    logger.info(f"test ner F1 {global_test_f:.5f}, disambiguation F1 {global_test_dict_f:.5f} @{best_epoch}")


def evaluate(args, model, tokenizer,ner_label_list ,trietree, dict_list, data_type="dev"):
    metric = SeqEntityScore(args.id2nerLabel)
    dict_metric = DictionaryScore()
    n_gpu = torch.cuda.device_count()
    n_gpu = n_gpu if n_gpu > 0 else 1
    eval_dataset = load_and_cache_examples(args, tokenizer,ner_label_list, trietree, data_type=data_type)
    eval_bs_per_gpu = args.eval_bs_per_gpu * n_gpu
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=eval_bs_per_gpu)

    # Eval!
    logger.info("***** Running evaluation %s *****", data_type)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_bs_per_gpu)
    logger.info("  total_steps= %d", len(eval_dataloader) * eval_bs_per_gpu)

    eval_loss = 0.0
    for _, batch in enumerate(eval_dataloader):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0], "attention_mask": batch[2], "token_type_ids": batch[3],
            "labels": batch[4], "input_len": batch[7],
            "name_ids": batch[1], "dict_labels": batch[5], "dict_mask": batch[6],
            "loss_mask": batch[-1]
        }
        # print(len(batch[0]), len(batch[1]), len(batch[3]),len(batch[4]), len(batch[5]), len(batch[6]),len(batch[7]), len(batch[-1]))
        outputs = model(**inputs)
        loss, logits_ner, logits_dict = outputs
        if n_gpu > 1: loss = loss.mean()
        eval_loss += loss.item()

        pred_label_ids = logits_ner.argmax(dim=-1)
        true_label_ids = inputs["labels"]
        ner_mask = inputs["attention_mask"] if args.use_subword else inputs["loss_mask"]
        metric.update(pred_label_ids, true_label_ids, ner_mask)

        true_dict_ids = inputs["dict_labels"]
        pred_dict_ids = logits_dict.argmax(dim=-1).view(true_dict_ids.shape)
        dict_metric.update(pred_dict_ids, true_dict_ids, inputs["dict_mask"])

    eval_p, eval_r, eval_f = metric.compute()
    logger.info(f"{data_type} result: P {eval_p:.5f}, R {eval_r:.5f}, F1 {eval_f:.5f}")

    dict_p, dict_r, dict_f = dict_metric.compute()
    logger.info(f"disambiguous result: P {dict_p:.5f}, R {dict_r:.5f}, F1 {dict_f:.5f}")

    return eval_f, dict_f


def predict():
    pass

def load_and_cache_examples(args, tokenizer, ner_label_list, trietree, data_type="train"):
    processor = CmoProcessor()
    # 如果缓存了特征文件，就直接加载，否则重新生成
    cached_features_file = args.output_dir.joinpath(f"cached_{data_type}_{args.train_max_seq_length}_{args.task_name}")
    if cached_features_file.exists():
        logger.info(f"Loading features from cached file {cached_features_file}")
        features = torch.load(cached_features_file)
    else:
        logger.info(f"Creating features from dataset file at{args.data_dir}")

        if data_type == "train":
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == "dev":
            examples = processor.get_dev_examples(args.data_dir)
        elif data_type == "test":
            examples = processor.get_test_examples(args.data_dir)

        features = convert_examples_to_features(
            examples=examples,
            ner_label_list=ner_label_list,
            max_seq_length=args.train_max_seq_length,
            tokenizer=tokenizer,
            trietree=trietree,
            dict_label_list=args.itos_dict,
            max_dict_num=args.max_dict_num
        )
        logger.info(f"Saving features into cached file {cached_features_file}")
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_dict_ids = torch.tensor([f.dict_ids for f in features], dtype=torch.long)
    all_dict_supervises = torch.tensor([f.dict_supervise for f in features], dtype=torch.long)
    all_dict_masks = torch.tensor([f.dict_mask for f in features], dtype=torch.long)
    all_input_len = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_loss_mask = torch.tensor([f.loss_mask for f in features], dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_dict_ids, all_input_mask, all_segment_ids, all_label_ids,
        all_dict_supervises, all_dict_masks, all_input_len, all_loss_mask
    )
    return dataset

def main():
    args = get_argparse().parse_args()

    if args.output_dir.exists():
        if args.overwrite_output:
            shutil.rmtree(args.output_dir)
            args.output_dir.mkdir()
    else:
        args.output_dir.mkdir()

    if not args.model_save_dir.exists():
        args.model_save_dir.mkdir()

    init_logger(log_file=(args.output_dir / "log.txt"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device


    # 准备数据
    processor = CmoProcessor()
    ner_label_list = processor.get_labels(args.data_dir)
    args.id2nerLabel = ner_label_list
    args.nerLabel2id = {label: i for i, label in enumerate(ner_label_list)}
    num_ner_label = len(ner_label_list)  # 序列标签的数目：包括B- I- O等等

    # 载入词典的标签label
    itos_dict = file2list(args.dict_label_path)
    if "O" in itos_dict: itos_dict.remove("O")
    if "[CLS]" in itos_dict: itos_dict.remove("[CLS]")
    itos_dict = ["O", "[CLS]"] + itos_dict
    num_dict_label = len(itos_dict)
    args.itos_dict = itos_dict

    trietree = SpaceTrieTree(args)
    if args.dict_path.exists():
        trietree.load_dict_from_tsv(args.dict_path)
    if (args.data_dir / "vocab.txt").exists():
        trietree.load_dict_from_tsv(args.data_dir / "vocab.txt")

    if args.seed > 0:
        seed_everything(args.seed)

    bert_config = BertConfig.from_pretrained(args.model_name_or_path)
    tokenizer = CNerTokenizer.from_pretrained(args.model_name_or_path)
    model = BertDictForNerSoftmax.from_pretrained(
        args.model_name_or_path,
        config=bert_config,
        args=args,
        num_ner_label=num_ner_label,
        num_dict_label=num_dict_label
    )

    model.to(args.device)
    logger.info(f"Training/Evaluation parameters:")
    print_config(vars(args))
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, ner_label_list, trietree, data_type="train")
        train(args, train_dataset, model, tokenizer,ner_label_list ,trietree, itos_dict)
    # 多了ner_label_list
    logger.info("finish!")


if __name__ == "__main__":
    main()
