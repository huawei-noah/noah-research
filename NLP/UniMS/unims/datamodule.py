# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
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

from os.path import join
import logging
import json
from rouge import Rouge

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataset import MSMODataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MMSumDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer, image_preprocess):
        super(MMSumDataModule, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.image_preprocess = image_preprocess
        self.ROUGE = Rouge()

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            with open(join(self.args["data_path"], "preprocess", f"train_data.json"), "r") as json_file:
                self.train_data = json.load(json_file)
            with open(join(self.args["data_path"], "preprocess", f"valid_data.json"), "r") as json_file:
                self.valid_data = json.load(json_file)

        if stage in (None, 'fit', 'test', 'predict'):
            with open(join(self.args["data_path"], "preprocess", f"test_data.json"), "r") as json_file:
                self.test_data = json.load(json_file)

            self.image_references = {}
            for line in open(join(self.args["data_path"], 'image_annotation.txt'), 'r'):
                line = line.strip()
                if line == 'None':
                    continue
                line = line.split(' ')
                self.image_references[f"test_data/article/{line[0].split('_')[0]}.txt"] = line

    def _make_article_seq_by_len(self, sents, pseudo_label):
        lm_tokenized_article = {
            "input_ids": torch.zeros(1, self.args["sent_token_len"], dtype=torch.int32),
            "attention_mask": torch.zeros(1, self.args["sent_token_len"], dtype=torch.int32),
        }

        sent_len = 0
        cls_position = []
        article = []
        for i in range(len(sents)):
            tokenized_sent = self.tokenizer(
                sents[i],
                add_special_tokens=self.args["add_special_tokens"],
                return_tensors=self.args["return_tensors"],
            )

            # add <s> and <\s>
            if sent_len + tokenized_sent["input_ids"].size()[1] <= self.args["sent_token_len"]:
                cls_position.append(sent_len)
                article.append(sents[i])
                sent_len += tokenized_sent["input_ids"].size()[1]

                lm_tokenized_article["input_ids"][:, cls_position[-1]: sent_len] = tokenized_sent["input_ids"]
                lm_tokenized_article["attention_mask"][:, cls_position[-1]: sent_len] = tokenized_sent[
                    "attention_mask"]
            else:
                break

        pseudo_label = [label for label in pseudo_label if label < len(article)]
        return article, lm_tokenized_article, pseudo_label, cls_position

    def _make_sentence_label(self, article, references):
        refer_sent_label = []
        refer_target = torch.zeros(self.args['max_sent_num']).bool()
        refer_labels = torch.zeros(self.args['max_sent_num'])

        body_now = ''
        sent_orders = list(range(len(article)))
        rouge_scores = []
        for _ in range(len(article)):
            preds = [body_now + article[order] for order in sent_orders]
            refs = [references for _ in range(len(sent_orders))]
            scores = self.ROUGE.get_scores(preds, refs)
            order_orders = sorted(range(len(sent_orders)), key=lambda k: scores[k]['rouge-l']['f'],
                                  reverse=True)

            order = order_orders[0]
            index = sent_orders.pop(order)
            if rouge_scores != [] and scores[order]['rouge-l']['f'] <= rouge_scores[-1]:
                break
            rouge_scores.append(scores[order]['rouge-l']['f'])

            refer_sent_label.append(index)
            refer_target[index] = True
            refer_labels[index] = 1

            body_now += article[index]

        return refer_sent_label, refer_target, refer_labels

    def _preprocess_data(self, data):
        articles = []
        refers = []
        if self.args['use_image']:
            cv_raw_images = []
            raw_images = []
            padded_images = []
        else:
            cv_raw_images = None
            raw_images = None
            padded_images = None
        lm_tokenized_texts = []
        pseudo_labels = []
        cls_positions = []
        image_labels = []
        padded_captions = []
        filepath_set = []
        image_references = []
        sentence_refers = []

        for i in range(len(data)):
            article = [s.lower() if self.args["sent_lower_case"] else s for s in data[i]['article']]
            refer = [r.lower() if self.args["sent_lower_case"] else r for r in data[i]['reference']['refers']]
            if self.args['use_image']:
                C, H, W = data[i]['image'][0].size()
                padded_image = torch.zeros(self.args['max_image_num'], C, H, W)
                padded_image[:len(data[i]['image'])] = data[i]['image']

            article, lm_tokenized_article, pseudo_label, cls_position = self._make_article_seq_by_len(
                article, data[i]['reference']['sentence_label'])

            pseudo_label, refer_target, refer_labels = self._make_sentence_label(article, " ".join(refer))
            reference = {
                "indexes": data[i]['reference']['indexes'],
                "target": refer_target,
                "labels": refer_labels,
            }

            # Remove overLength labels
            for sn in range(len(article), self.args['max_sent_num']):
                reference['target'][sn] = False
                reference['labels'][sn] = 0
            if reference['labels'].sum() == 0:
                continue

            # Add Processed Data
            if self.args["sent_order"] == "original":
                articles.append(article)
            else:
                articles.append([article[index] for index in pseudo_label])
            refers.append(refer)
            lm_tokenized_texts.append(lm_tokenized_article)
            pseudo_labels.append(pseudo_label)
            cls_positions.append(cls_position)
            padded_captions.append(data[i]['caption'])
            filepath_set.append(data[i]['filename'])
            sentence_refers.append(reference)

            if self.args['use_image']:
                image_label = torch.zeros(self.args['max_image_num'])
                if self.args['image_pseudo_label'] == 'distill':
                    distill_label = data[i]['image_reference']['labels']
                    for image_label_i in range(len(distill_label)):
                        image_label[image_label_i] = distill_label[image_label_i]
                elif self.args['image_pseudo_label'] == 'order':
                    for image_label_i in range(3):
                        image_label[image_label_i] = 1
                elif self.args['image_pseudo_label'] == 'rouge':
                    image_label = torch.zeros(self.args['max_image_num'])
                    caption = data[i]['caption']
                    # caption_refer = [refer for _ in range(len(caption))]
                    caption_refer = [" ".join(refer) for _ in range(len(caption))]
                    scores = self.ROUGE.get_scores(caption, caption_refer)
                    order_orders = sorted(range(len(caption)), key=lambda k: scores[k]['rouge-l']['f'],
                                          reverse=True)
                    for image_label_i in range(min(3, len(caption))):
                        image_label[order_orders[image_label_i]] = 1
                cv_raw_images.append(data[i]['cv_raw_image'])
                raw_images.append(data[i]['raw_image'])
                padded_images.append(padded_image)
                image_references.append(data[i]['image_reference'])
                image_labels.append(image_label)

        return {
            'filenames': filepath_set,
            'articles': articles,
            'tokenized_articles': lm_tokenized_texts,
            'cls_positions': cls_positions,
            'refers': refers,
            'sentence_labels': pseudo_labels,
            'sentence_refers': sentence_refers,
            'cv_raw_images': cv_raw_images,
            'raw_images': raw_images,
            'images': padded_images,
            'image_labels': image_labels,
            'image_refers': image_references,
            'captions': padded_captions,
        }

    def _collate_fn(self, data):
        data = self._preprocess_data(data)

        # Stack Tokenized Articles
        tokenized_articles = data.pop('tokenized_articles')
        data['tokenized_articles'] = {
            "input_ids": torch.stack(
                [tokenized_article["input_ids"].squeeze(dim=0) for tokenized_article in tokenized_articles]),
            "attention_mask": torch.stack(
                [tokenized_article["attention_mask"].squeeze(dim=0) for tokenized_article in
                 tokenized_articles]),
        }

        # Tokenize References
        tokenized_refers = self.tokenizer(
            # data['refers'],
            [" ".join(refer) for refer in data['refers']],
            add_special_tokens=self.args["add_special_tokens"],
            truncation=self.args["truncation"],
            max_length=self.args["ref_token_len"],
            padding=self.args["padding"],
            return_tensors=self.args["return_tensors"],
        )
        # Convert padding to -100
        if self.args["padding"] == "max_length" and self.args["ignore_pad_token_for_loss"]:
            for batch_i in range(tokenized_refers["input_ids"].size()[0]):
                for token_i in range(tokenized_refers["input_ids"].size()[1]):
                    if tokenized_refers["input_ids"][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                        tokenized_refers["input_ids"][batch_i][token_i] = -100
        data['tokenized_refers'] = tokenized_refers

        # Stack Sentence Refers
        sentence_refers = data.pop('sentence_refers')
        data['sentence_refers'] = {
            "indexes": torch.stack([sentence_refer["indexes"] for sentence_refer in sentence_refers]),
            "target": torch.stack([sentence_refer["target"] for sentence_refer in sentence_refers]),
            "labels": torch.stack([sentence_refer["labels"] for sentence_refer in sentence_refers]),
        }

        if self.args['use_image']:
            # Stack Images
            data['images'] = torch.stack(data['images'])
            # Stack Image Labels
            data['image_labels'] = torch.stack(data['image_labels'])
            # Stack Image Refers
            image_refers = data.pop('image_refers')
            data['image_refers'] = {
                "indexes": torch.stack([image_refer["indexes"] for image_refer in image_refers]),
                "target": torch.stack([image_refer["target"] for image_refer in image_refers]),
            }

        if "test" in data["filenames"][0]:
            data["split"] = "test"
        elif "valid" in data["filenames"][0]:
            data["split"] = "valid"
        else:
            data["split"] = "train"

        return data

    def make_dataloader(
            self,
            data,
            batch_size,
            image_references,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
    ):
        return DataLoader(
            dataset=MSMODataset(
                data=data,
                dataset_path=self.args["data_path"],
                image_preprocess=self.image_preprocess,
                max_sent_num=self.args["max_sent_num"],
                max_image_num=self.args["max_image_num"],
                image_references=image_references,
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            num_workers=self.args["num_workers"],
            collate_fn=self._collate_fn,
        )

    def train_dataloader(self):
        return self.make_dataloader(
            data=self.train_data,
            batch_size=self.args["batch_size"],
            image_references=None,
            shuffle=True,
            drop_last=True,
            pin_memory=False,
        )

    # def val_dataloader(self):
    #     return self.make_dataloader(
    #         data=self.valid_data,
    #         batch_size=self.args["val_batch_size"],
    #         # image_references=self.image_references,
    #         image_references=None,
    #         shuffle=False,
    #         drop_last=False,
    #         pin_memory=False,
    #     )
    def val_dataloader(self):
        dataloaders = [
            self.make_dataloader(
                data=self.valid_data,
                batch_size=self.args["val_batch_size"],
                # image_references=self.image_references,
                image_references=None,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
            ),
            self.make_dataloader(
                data=self.test_data,
                batch_size=self.args["val_batch_size"],
                image_references=self.image_references,
                shuffle=False,
                drop_last=False,
                pin_memory=False,
            )
        ]
        return dataloaders

    def test_dataloader(self):
        return self.make_dataloader(
            data=self.test_data,
            batch_size=self.args["test_batch_size"],
            image_references=self.image_references,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

    def predict_dataloader(self):
        return self.make_dataloader(
            data=self.test_data,
            batch_size=self.args["test_batch_size"],
            image_references=self.image_references,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )
