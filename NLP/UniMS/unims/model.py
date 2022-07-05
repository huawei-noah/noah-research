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

import json
import logging
import os
from os.path import join
import copy

import torch
import torch.nn.functional as F
from torch import optim
import cv2

from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
import pytorch_lightning as pl
from torchmetrics import RetrievalPrecision

import clip
from utils import cal_novel, save_render, save_CAM
from module import UniMS, BartLearnedPositionalEmbedding, ROUGEScore

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MMSum(pl.LightningModule):
    def __init__(self, args):
        super(MMSum, self).__init__()
        self.args = args

        # Summarization Model
        pretrained_model_path = join(args["pretrained_model_path"], args["backbone"])
        logger.info(pretrained_model_path)
        self.generator = UniMS.from_pretrained(
            pretrained_model_path, visual_guide=self.args['use_image'] and self.args['visual_guide'])

        if self.args['use_language_score']:
            # Sentence Extractor Head
            self.sentence_classifier_head = torch.nn.Sequential(
                torch.nn.LayerNorm(self.generator.config.hidden_size),
                torch.nn.Linear(self.generator.config.hidden_size, 1),
            )
            for n, p in self.sentence_classifier_head.named_parameters():
                if 'weight' in n:
                    p.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in n:
                    p.data.zero_()

        if self.args['use_image']:
            # Multimodal Model: image-text match model
            if self.args['visual_backbone'] in ['LinProj', 'CLIP-RN50', 'CLIP-ViT']:
                multimodal_model, self.image_preprocess = clip.load(
                    args["clip_path"], device=self.device, jit=False)
                self.visual = copy.deepcopy(multimodal_model.visual)
                del multimodal_model

            if self.args['visual_backbone'] == 'LinProj':
                del self.visual.ln_pre
                del self.visual.transformer
                del self.visual.ln_post
                del self.visual.proj

            elif self.args['visual_backbone'].endswith('RN50'):
                del self.visual.attnpool
                for param in self.visual.parameters():
                    param.requires_grad = False

                # Image Post Linear Proj
                self.image_post_linproj = torch.nn.Sequential(
                    torch.nn.LayerNorm(2048),
                    torch.nn.Linear(2048, 768),
                )

                # Image [CLS] Embedding
                self.visual_class_embedding = torch.nn.Parameter(
                    torch.zeros(self.generator.config.d_model, device=self.device))
                self.visual_class_embedding.data.normal_(mean=0.0, std=0.02)

            elif self.args['visual_backbone'].endswith('ViT'):
                del self.visual.ln_post
                del self.visual.proj
                for param in self.visual.parameters():
                    param.requires_grad = False

                # Image Post Linear Proj
                self.image_post_linproj = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.generator.config.hidden_size),
                    torch.nn.Linear(self.generator.config.hidden_size, self.generator.config.hidden_size),
                    # torch.nn.Sigmoid(),
                )

            # Image Position Embedding
            self.visual_embed_positions = BartLearnedPositionalEmbedding(
                self.generator.config.max_position_embeddings,
                self.generator.config.d_model,
            )
            self.visual_embed_positions.weight.data.normal_(mean=0.0, std=0.02)

            if self.args['use_image_score']:
                # Image Extractor Head
                self.image_classifier_head = torch.nn.Sequential(
                    torch.nn.LayerNorm(self.generator.config.hidden_size),
                    torch.nn.Linear(self.generator.config.hidden_size, 1),
                )
                for n, p in self.image_classifier_head.named_parameters():
                    if 'weight' in n:
                        p.data.normal_(mean=0.0, std=0.02)
                    elif 'bias' in n:
                        p.data.zero_()
        else:
            self.image_preprocess = None

        # Optimizer & Scheduler
        self.learning_rate = args["learning_rate"]
        self._weight_decay = args["weight_decay"]
        self._t_total = args["train_params"]["max_steps"]
        self._warmup_iters = int(self._t_total * args["warmup_ratio"])

        # Save Metrics for Validation Set
        if self.args["mode"] not in ['test']:
            self.val_abs_rouge = ROUGEScore()
            if self.args["use_language_score"]:
                self.val_sent_precision = [
                    RetrievalPrecision(k=i + 1) for i in range(3)
                ]
                self.val_ext_rouge = ROUGEScore()

        # Save Metrics for Test Set
        self.test_abs_rouge = ROUGEScore()
        if self.args["use_language_score"]:
            self.test_sent_precision = [
                RetrievalPrecision(k=i + 1) for i in range(3)
            ]
            self.test_ext_rouge = ROUGEScore()

        if self.args["use_image_score"]:
            self.test_image_precision = []
            for i in range(3):
                self.test_image_precision.append(RetrievalPrecision(k=i + 1))

        # Save Test Table for Visualization
        if self.args["mode"] in ['test']:
            self.summary_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
            self.gold_ngram_novel = {1: [0, 0, 0], 2: [0, 0, 0], 3: [0, 0, 0]}
            from module import Similarity
            self.similarity = Similarity(args["clip_path"], "./unims/clip/bpe_simple_vocab_16e6.txt.gz")
            # from module import BERTScore, MoverScore
            # self.BERTScore = BERTScore()
            # self.MoverScore = MoverScore()

    # @pysnooper.snoop()
    def ranking_loss(self, score, pseudo_label, margin=0):
        sorted_score = torch.zeros_like(score)
        for i, label in enumerate(pseudo_label):
            sorted_score[i] = score[label]
        ones = torch.ones_like(score)
        loss_func = torch.nn.MarginRankingLoss(0.0)  # , reduction="sum")
        TotalLoss = loss_func(sorted_score, sorted_score, ones)

        # candidate loss
        for i in range(1, len(sorted_score)):
            # pos_score = score[:, :-i]
            # neg_score = score[:, i:]
            pos_score = sorted_score[:-i]
            neg_score = sorted_score[i:]
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            ones = torch.ones_like(pos_score)
            loss_func = torch.nn.MarginRankingLoss(margin * i)  # , reduction="sum")
            loss = loss_func(pos_score, neg_score, ones)
            TotalLoss += loss

        return TotalLoss  # / len(sorted_score)

    def visual_embedding(self, x):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device), x],
            dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        # x = self.visual.ln_pre(x)

        if self.args['visual_backbone'] == 'ViT':
            # Transformers Layer
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            # x = self.visual.ln_post(x[:, 0, :])

            # Post Linear Layer
            # if self.visual.proj is not None:
            #     x = x @ self.visual.proj
            x = self.image_post_linproj(x)

        return x

    def visual_embedding_cnn(self, x):
        def stem(x):
            for conv, bn in [(self.visual.conv1, self.visual.bn1), (self.visual.conv2, self.visual.bn2),
                             (self.visual.conv3, self.visual.bn3)]:
                x = self.visual.relu(bn(conv(x)))
            x = self.visual.avgpool(x)
            return x

        x = x.type(self.visual.conv1.weight.dtype)
        x = stem(x)
        x = self.visual.layer1(x)
        x = self.visual.layer2(x)
        x = self.visual.layer3(x)
        x = self.visual.layer4(x)
        # x = self.visual.attnpool(x)
        B, E, _, _ = x.size()
        x = x.view(B, E, -1).permute(0, 2, 1)
        x = self.image_post_linproj(x)
        x = torch.cat(
            [self.visual_class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1], device=self.device), x],
            dim=1)  # shape = [*, grid ** 2 + 1, width]

        return x

    def extract_guide(self, generator_outputs, articles, pseudo_labels, cls_positions, sentence_refers):
        encoder_hidden_states = generator_outputs.encoder_hidden_states[self.args['language_score_layer']]
        sent_loss = torch.tensor(0.0, device=self.device)
        batch_text_logits = self.sentence_classifier_head(encoder_hidden_states).squeeze(dim=-1)

        ordered_indices = []
        sent_logits = torch.zeros((batch_text_logits.size()[0], self.args['max_sent_num']),
                                  device=self.device)
        for item_i, (unordered_sents, text_logits, pseudo_label, cls_position) in enumerate(
                zip(articles, batch_text_logits, pseudo_labels, cls_positions)):
            # get sentence scores
            item_sent_logits = torch.zeros(len(unordered_sents), device=self.device)
            for i, pos in enumerate(cls_position):
                item_sent_logits[i] = text_logits[pos]
            # logger.info(item_sent_logits)
            probs = F.softmax(item_sent_logits, dim=0)
            sent_logits[item_i, :len(unordered_sents)] = item_sent_logits

            if self.args['use_ranking_loss']:
                # calculate sequence ranking loss
                sent_loss += self.ranking_loss(probs, pseudo_label, margin=self.args["margin"])

            sorted_probs, indices = probs.sort(descending=True)
            ordered_indices.append(indices)

        sent_logits = F.softmax(sent_logits, dim=-1)
        if not self.args['use_ranking_loss']:
            # calculate binary cross entropy loss
            loss_func = torch.nn.BCELoss()
            sent_loss += loss_func(sent_logits, sentence_refers['labels'])

        return sent_loss, ordered_indices, sent_logits

    def image_guide(self, generator_outputs, image_labels):
        encoder_hidden_states = generator_outputs.encoder_hidden_states[self.args['image_score_layer']]
        batch_image_logits = self.image_classifier_head(encoder_hidden_states).squeeze(dim=-1)

        image_logits = torch.zeros((batch_image_logits.size()[0], self.args['max_image_num']),
                                   device=self.device)
        for i in range(self.args['max_image_num']):
            image_logits[:, i] = batch_image_logits[:, 512 + i * 50]

        # KD from CLIP image score with temperature
        log_image_logits = F.log_softmax(image_logits / self.args['image_tau'], dim=1)
        # image_logits = F.softmax(image_logits / self.args['image_tau'], dim=-1)
        image_labels = F.softmax(image_labels / self.args['image_tau'], dim=-1)
        loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        image_loss = loss_func(log_image_logits, image_labels)
        # loss_func = torch.nn.CrossEntropyLoss()
        # image_loss = loss_func(image_logits, image_labels)

        return image_loss, image_logits

    def training_step(self, batch, batch_idx):
        images = batch['images']
        lm_tokenized_texts = batch['tokenized_articles']
        targets = batch['tokenized_refers']
        articles = batch['articles']
        pseudo_labels = batch['sentence_labels']
        cls_positions = batch['cls_positions']
        image_labels = batch['image_labels']
        sentence_refers = batch['sentence_refers']

        if self.args['use_image']:
            # [BSZ, max_image_num, channel, input_resolution, input_resolution]
            B, I, C, R, _ = images.size()
            # [BSZ * max_image_num, channel, input_resolution, input_resolution]
            images = images.view(-1, C, R, R)

            # visual features from CLIP visual encoder
            # with pretrained image class embedding: [BSZ, max_image_num * (patch_len + 1), visual_width]
            if self.args['visual_backbone'] == 'ResNet50':
                visual_features = self.visual_embedding_cnn(images).view(B, -1,
                                                                         self.generator.config.hidden_size)
            else:
                visual_features = self.visual_embedding(images).view(B, -1, self.generator.config.hidden_size)
            # visual position embedding: [max_image_num * (patch_len + 1), visual_width]
            visual_embed_pos = self.visual_embed_positions(visual_features.size()[:-1])

            # [BSZ, max_image_num * (patch_len + 1), visual_width]
            visual_features = visual_features + visual_embed_pos
            visual_attention_mask = torch.ones(visual_features.size()[:-1], device=self.device)
            lm_tokenized_texts["attention_mask"] = torch.cat(
                [lm_tokenized_texts["attention_mask"], visual_attention_mask], dim=1)
        else:
            visual_features = None

        lm_tokenized_texts["labels"] = targets["input_ids"]
        generator_outputs = self.generator(
            **lm_tokenized_texts,
            output_hidden_states=True,
            visual_features=visual_features,
        )
        loss = generator_outputs.loss
        self.log("train_gen_loss_step", loss)

        if self.args['use_language_score']:
            sent_loss, ordered_indices, sent_logits = self.extract_guide(generator_outputs, articles,
                                                                         pseudo_labels,
                                                                         cls_positions, sentence_refers)
            self.log("train_sent_loss_step", sent_loss)
            loss = loss + self.args['language_balance'] * sent_loss

            # for precision in self.train_sent_precision:
            #     precision.update(sent_logits, sentence_refers['target'], sentence_refers['indexes'])

        if self.args['use_image'] and self.args['use_image_score']:
            image_loss, image_logits = self.image_guide(generator_outputs, image_labels)
            self.log("train_image_loss_step", image_loss)
            loss = loss + self.args['image_balance'] * image_loss

        self.log("train_loss_step", loss)
        if self.args['scheduler']:
            self.log("train_lr_step", self.lr_schedulers().get_lr()[0])

        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        # if self.args["use_language_score"]:
        #     for i, precision in enumerate(self.train_sent_precision):
        #         sent_precision = precision.compute()
        #         self.log(f"train_sent_precision@{i + 1}_epoch", sent_precision)
        #         precision.reset()
        return

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        self.eval()
        images = batch['images']
        lm_tokenized_texts = batch['tokenized_articles']
        # targets = batch['tokenized_refers']
        articles = batch['articles']
        pseudo_labels = batch['sentence_labels']
        cls_positions = batch['cls_positions']
        image_labels = batch['image_labels']
        # refers = batch['refers']
        refers = [" ".join(refer) for refer in batch['refers']]
        image_refers = batch['image_refers']
        sentence_refers = batch['sentence_refers']

        if self.args['use_image']:
            # [BSZ, max_image_num, channel, input_resolution, input_resolution]
            B, I, C, R, _ = images.size()
            # [BSZ * max_image_num, channel, input_resolution, input_resolution]
            images = images.view(-1, C, R, R)

            # visual features from CLIP visual encoder
            # with pretrained image class embedding: [BSZ, max_image_num * (patch_len + 1), visual_width]
            if self.args['visual_backbone'] == 'ResNet50':
                visual_features = self.visual_embedding_cnn(images).view(B, -1,
                                                                         self.generator.config.hidden_size)
            else:
                visual_features = self.visual_embedding(images).view(B, -1, self.generator.config.hidden_size)

            # visual position embedding: [max_image_num * (patch_len + 1), visual_width]
            visual_embed_pos = self.visual_embed_positions(visual_features.size()[:-1])

            # [BSZ, max_image_num * (patch_len + 1), visual_width]
            visual_features = visual_features + visual_embed_pos
            visual_attention_mask = torch.ones(visual_features.size()[:-1], device=self.device)
            lm_tokenized_texts["attention_mask"] = torch.cat(
                [lm_tokenized_texts["attention_mask"], visual_attention_mask], dim=1)
        else:
            visual_features = None

        generator_outputs = self.generator.generate(
            input_ids=lm_tokenized_texts["input_ids"],
            attention_mask=lm_tokenized_texts["attention_mask"],
            # num_beams=4,
            # length_penalty=0.8,
            # length_penalty=2.0,
            max_length=self.args["ref_token_len"],
            min_length=20,
            # early_stopping=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            visual_features=visual_features,
        )

        if self.args['use_language_score']:
            _, ordered_indices, sent_logits = self.extract_guide(generator_outputs, articles, pseudo_labels,
                                                                 cls_positions, sentence_refers)
            ext_outputs = []
            for article, indices, pseudo_label in zip(articles, ordered_indices, pseudo_labels):
                ext_outputs.append(' '.join([article[index] for index in indices[:self.args['extract_num']]]))
            if batch["split"] in ["valid"]:
                self.val_ext_rouge.update(ext_outputs, refers)
                for precision in self.val_sent_precision:
                    precision.update(sent_logits, sentence_refers['target'], sentence_refers['indexes'])
            elif batch["split"] in ["test"]:
                self.test_ext_rouge.update(ext_outputs, refers)
                for precision in self.test_sent_precision:
                    precision.update(sent_logits, sentence_refers['target'], sentence_refers['indexes'])

        if self.args['use_image'] and self.args['use_image_score'] and batch["split"] in ["test"]:
            _, image_logits = self.image_guide(generator_outputs, image_labels)
            for precision in self.test_image_precision:
                precision.update(image_logits, image_refers['target'], image_refers['indexes'])

        abs_outputs = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in generator_outputs.sequences
        ]
        if batch["split"] in ["valid"]:
            self.val_abs_rouge.update(abs_outputs, refers)
        elif batch["split"] in ["test"]:
            self.test_abs_rouge.update(abs_outputs, refers)

        return [' '.join(article) for article in articles], abs_outputs, refers

    def validation_epoch_end(self, validation_step_outputs):
        # dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # save_path = join(self.args["decode_path"], f"results_{dt}_{self.global_rank}.txt")
        # save_result_for_show(
        #     validation_step_outputs,
        #     save_path,
        #     args=self.args,
        #     # use_score=(self.args["use_language_score"]
        #     #            or self.args["use_image_score"]),
        #     use_score=False
        # )

        val_abs_rouge_results = self.val_abs_rouge.compute()
        if self.global_rank == 0:
            logger.info(f"val_abs_rouge_results: {val_abs_rouge_results}")
        self.log("val_abs_R1", val_abs_rouge_results["rouge-1"]["f"])
        self.log("val_abs_R2", val_abs_rouge_results["rouge-2"]["f"])
        self.log("val_abs_RL", val_abs_rouge_results["rouge-L"]["f"])
        self.val_abs_rouge.reset()

        test_abs_rouge_results = self.test_abs_rouge.compute()
        if self.global_rank == 0:
            logger.info(f"test_abs_rouge_results: {test_abs_rouge_results}")
        self.log("test_abs_R1", test_abs_rouge_results["rouge-1"]["f"])
        self.log("test_abs_R2", test_abs_rouge_results["rouge-2"]["f"])
        self.log("test_abs_RL", test_abs_rouge_results["rouge-L"]["f"])
        self.test_abs_rouge.reset()

        if self.args["use_language_score"]:
            val_ext_rouge_results = self.val_ext_rouge.compute()
            if self.global_rank == 0:
                logger.info(f"val_ext_rouge_results: {val_ext_rouge_results}")
            self.log("val_ext_R1", val_ext_rouge_results["rouge-1"]["f"])
            self.log("val_ext_R2", val_ext_rouge_results["rouge-2"]["f"])
            self.log("val_ext_RL", val_ext_rouge_results["rouge-L"]["f"])
            self.val_ext_rouge.reset()

            for i, precision in enumerate(self.val_sent_precision):
                sent_precision = precision.compute()
                if self.global_rank == 0:
                    logger.info(f"val_sent_precision@{i + 1}_epoch: {sent_precision}")
                self.log(f"val_sent_precision@{i + 1}_epoch", sent_precision)
                precision.reset()

            test_ext_rouge_results = self.test_ext_rouge.compute()
            if self.global_rank == 0:
                logger.info(f"test_ext_rouge_results: {test_ext_rouge_results}")
            self.log("test_ext_R1", test_ext_rouge_results["rouge-1"]["f"])
            self.log("test_ext_R2", test_ext_rouge_results["rouge-2"]["f"])
            self.log("test_ext_RL", test_ext_rouge_results["rouge-L"]["f"])
            self.test_ext_rouge.reset()

            for i, precision in enumerate(self.test_sent_precision):
                sent_precision = precision.compute()
                if self.global_rank == 0:
                    logger.info(f"test_sent_precision@{i + 1}_epoch: {sent_precision}")
                self.log(f"test_sent_precision@{i + 1}_epoch", sent_precision)
                precision.reset()

        if self.args['use_image'] and self.args['use_image_score']:
            for i, precision in enumerate(self.test_image_precision):
                image_precision = precision.compute()
                if self.global_rank == 0:
                    logger.info(f"test_image_precision@{i + 1}_epoch: {image_precision}")
                self.log(f"test_image_precision@{i + 1}_epoch", image_precision)
                precision.reset()

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        self.eval()
        tokenized_sents, targets, _ = batch
        loss = self.step(tokenized_sents, targets)
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        self.eval()

        filenames = batch['filenames']
        # if not 'article/6a47d32a3377088255c3f609a861a2109dc33c81.txt' in filenames:
        #     return
        # logger.info(filenames)

        raw_images = batch['raw_images']
        images = batch['images']
        lm_tokenized_texts = batch['tokenized_articles']
        # targets = batch['tokenized_refers']
        articles = batch['articles']
        pseudo_labels = batch['sentence_labels']
        cls_positions = batch['cls_positions']
        image_labels = batch['image_labels']
        # refers = batch['refers']
        refers = [" ".join(refer) for refer in batch['refers']]
        image_refers = batch['image_refers']
        sentence_refers = batch['sentence_refers']
        # result_data = [filenames[0], articles[0], refers[0]]

        if self.args['use_image']:
            B, I, C, R, _ = images.size()
            images = images.view(
                -1, C, R, R)  # [BSZ, max_image_num, channel, input_resolution, input_resolution]

            # visual features from CLIP visual encoder
            # with pretrained image class embedding: [BSZ, max_image_num * (patch_len + 1), visual_width]
            if self.args['visual_backbone'].endswith('RN50'):
                visual_features = self.visual_embedding_cnn(images).view(
                    B, -1, self.generator.config.hidden_size)
            else:
                visual_features = self.visual_embedding(images).view(
                    B, -1, self.generator.config.hidden_size)
            # visual position embedding:
            visual_embed_pos = self.visual_embed_positions(
                visual_features.size()[:-1])  # [max_image_num * (patch_len + 1), visual_width]

            visual_features = visual_features + visual_embed_pos  # [BSZ, max_image_num * (patch_len + 1), visual_width]
            visual_attention_mask = torch.ones(visual_features.size()[:-1], device=self.device)
            lm_tokenized_texts["attention_mask"] = torch.cat(
                [lm_tokenized_texts["attention_mask"], visual_attention_mask], dim=1)
        else:
            visual_features = None

        generator_outputs = self.generator.generate(
            input_ids=lm_tokenized_texts["input_ids"],
            attention_mask=lm_tokenized_texts["attention_mask"],
            num_beams=5,
            length_penalty=2.2,
            max_length=self.args["ref_token_len"],
            min_length=20,
            # no_repeat_ngram_size=2,
            no_repeat_ngram_size=5,
            early_stopping=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            visual_features=visual_features,
        )

        abs_outputs = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for g in generator_outputs.sequences
        ]
        abs_rouge_results = self.test_abs_rouge.update(abs_outputs, refers)
        # abs_bert_score_results = self.BERTScore.update(abs_outputs, refers)
        # abs_mover_score_results = self.MoverScore.update(abs_outputs, refers)

        if self.args['use_language_score']:
            _, ordered_indices, sent_logits = self.extract_guide(generator_outputs, articles, pseudo_labels,
                                                                 cls_positions, sentence_refers)
            ext_outputs = []
            for article, indices, pseudo_label in zip(articles, ordered_indices, pseudo_labels):
                ext_outputs.append([article[index] for index in indices[:3]])
            ext_rouge_results = self.test_ext_rouge.update([' '.join(e) for e in ext_outputs], refers)

        for c, g, s in zip(abs_outputs, refers, articles):
            cal_novel(c, g, ' '.join(s), self.summary_ngram_novel, self.gold_ngram_novel)

        if self.args['use_image'] and self.args['use_image_score']:
            _, image_logits = self.image_guide(generator_outputs, image_labels)
            for precision in self.test_image_precision:
                precision.update(image_logits, image_refers['target'], image_refers['indexes'])
            extract_image = []
            extract_image_cv2 = []
            for item_i, image_logit in enumerate(image_logits):
                # logger.info(image_logit)
                _, indices = image_logit.sort(descending=True)
                index = indices[0]
                if index >= len(raw_images[item_i]):
                    index = len(raw_images[item_i]) - 1
                extract_image.append(raw_images[item_i][index])
                extract_image_cv2.append(batch['cv_raw_images'][item_i][index])
            image_sims = self.similarity.update(extract_image, abs_outputs)

        if self.args['use_language_score'] and self.args['use_image'] and self.args['use_image_score']:
            self.save_results(
                batch, generator_outputs, abs_outputs, abs_rouge_results, ext_outputs, ext_rouge_results,
                image_sims, extract_image_cv2, lm_tokenized_texts)
        elif self.args['use_image'] and self.args['use_image_score']:
            self.save_results_for_wo_ext(batch, abs_outputs, abs_rouge_results, image_sims, extract_image_cv2)
        else:
            self.save_results_for_bart(batch, abs_outputs, abs_rouge_results)

    def save_results(self, batch, generator_outputs, abs_outputs, abs_rouge_results, ext_outputs,
                     ext_rouge_results,
                     image_sims, extract_image_cv2, lm_tokenized_texts):

        encoder_hidden_states = generator_outputs.encoder_hidden_states[self.args['language_score_layer']]
        for hidden_state, cv_raw_images, filename, abs_output, abs_rouge_result, ext_output, ext_rouge_result, image_sim, article, refer, ext_image, pseudo_label, input_ids, attention_mask in zip(
                encoder_hidden_states, batch['cv_raw_images'], batch['filenames'], abs_outputs,
                abs_rouge_results,
                ext_outputs, ext_rouge_results, image_sims, batch['articles'], batch['refers'],
                extract_image_cv2,
                batch['sentence_labels'], lm_tokenized_texts["input_ids"],
                lm_tokenized_texts["attention_mask"]
        ):
            # if not filename == 'article/6a47d32a3377088255c3f609a861a2109dc33c81.txt':
            #     continue
            # Control Good Case
            # if abs_rouge_result['rouge-1']['f'] * 100 < 60:
            #     continue
            # elif ext_rouge_result['rouge-1']['f'] * 100 < 60:
            #     continue
            # elif image_sim < 25:
            #     continue
            # l_num = 0
            # for l in pseudo_label:
            #     if l in [0, 1, 2]:
            #         l_num += 1
            # if l_num > 1:
            #     continue

            # Make Filename & Folds
            filename = filename.split('/')[-1].split('.')[-2]
            foldname = f"{format(abs_rouge_result['rouge-1']['f'] * 100, '.2f')}_{format(ext_rouge_result['rouge-1']['f'] * 100, '.2f')}_{format(image_sim, '.2f')}_{filename}"
            os.mkdir(join(self.args['decode_path'], foldname))

            # Save Text Case
            # input_ids
            seq_len = int(attention_mask[:512].sum().item())
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:seq_len].detach().cpu().numpy().tolist())
            tokens = [token.strip() for token in tokens]

            scores_to_render = hidden_state[:seq_len]
            scores_to_render = self.image_classifier_head(scores_to_render).squeeze(
                dim=1).detach().cpu().numpy().tolist()
            scores_to_render = [
                (s - min(scores_to_render)) / (max(scores_to_render) - min(scores_to_render)) * 2 - 1
                for s in scores_to_render
            ]
            assert len(scores_to_render) == len(tokens)

            save_render(tokens, scores_to_render, join(self.args['decode_path'], foldname, 'render.html'))

            # Save Results
            with open(join(self.args['decode_path'], foldname, 'outputs.txt'), 'w') as file:
                for line in article:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                for line in refer:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                for l in pseudo_label:
                    file.write(f'{article[l]}\n')
                file.write('\n\n\n')
                for line in ext_output:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                file.write(f'{abs_output}\n')

            cv2.imwrite(join(self.args['decode_path'], foldname, 'ext_image.jpg'), ext_image)

            with open(join(self.args['decode_path'], foldname, 'results.json'), 'w') as file:
                json.dump({
                    'abs_rouge_result': abs_rouge_result,
                    'ext_rouge_result': ext_rouge_result,
                    'image_sim': format(image_sim, '.2f'),
                }, file)

            # Save Raw & CAM Images
            os.mkdir(join(self.args['decode_path'], foldname, 'images'))
            os.mkdir(join(self.args['decode_path'], foldname, 'resized_images'))
            os.mkdir(join(self.args['decode_path'], foldname, 'cam_results'))
            for image_i, cv_raw_image in enumerate(cv_raw_images):
                cv2.imwrite(join(self.args['decode_path'], foldname, 'images', f'{filename}_{image_i}.jpg'),
                            cv_raw_image)
                cv2.imwrite(
                    join(self.args['decode_path'], foldname, 'resized_images', f'{filename}_{image_i}.jpg'),
                    cv2.resize(cv_raw_image, (256, 256)))
                cam = hidden_state[512 + 50 * image_i + 1: 512 + 50 * image_i + 50]
                cam = self.sentence_classifier_head(cam).squeeze(dim=1).view(7, 7)
                new_cam = torch.zeros_like(cam)
                new_cam[:, 0] = cam[:, 0]
                new_cam[:, 1] = cam[:, 4]
                new_cam[:, 2] = cam[:, 5]
                new_cam[:, 3] = cam[:, 3]
                new_cam[:, 4] = cam[:, 1]
                new_cam[:, 5] = cam[:, 2]
                new_cam[:, 6] = cam[:, 6]
                cam = new_cam.unsqueeze(dim=2).detach().cpu().numpy()

                save_CAM(
                    cam, cv_raw_image,
                    join(self.args['decode_path'], foldname, 'cam_results', f'{filename}_{image_i}.jpg')
                )

    def save_results_for_wo_ext(self, batch, abs_outputs, abs_rouge_results, image_sims, extract_image_cv2):
        for cv_raw_images, filename, abs_output, abs_rouge_result, image_sim, article, refer, ext_image in zip(
                batch['cv_raw_images'], batch['filenames'], abs_outputs, abs_rouge_results,
                image_sims, batch['articles'], batch['refers'], extract_image_cv2):

            # Make Filename & Folds
            filename = filename.split('/')[-1].split('.')[-2]
            foldname = f"{format(abs_rouge_result['rouge-1']['f'] * 100, '.2f')}_{format(image_sim, '.2f')}_{filename}"
            os.mkdir(join(self.args['decode_path'], foldname))

            # Save Results
            with open(join(self.args['decode_path'], foldname, 'outputs.txt'), 'w') as file:
                for line in article:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                for line in refer:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                file.write(f'{abs_output}\n')

            cv2.imwrite(join(self.args['decode_path'], foldname, 'ext_image.jpg'), ext_image)

            with open(join(self.args['decode_path'], foldname, 'results.json'), 'w') as file:
                json.dump({
                    'abs_rouge_result': abs_rouge_result,
                    'image_sim': format(image_sim, '.2f'),
                }, file)

            # Save Raw & CAM Images
            os.mkdir(join(self.args['decode_path'], foldname, 'images'))
            os.mkdir(join(self.args['decode_path'], foldname, 'resized_images'))
            for image_i, cv_raw_image in enumerate(cv_raw_images):
                cv2.imwrite(join(self.args['decode_path'], foldname, 'images', f'{filename}_{image_i}.jpg'),
                            cv_raw_image)
                cv2.imwrite(
                    join(self.args['decode_path'], foldname, 'resized_images', f'{filename}_{image_i}.jpg'),
                    cv2.resize(cv_raw_image, (256, 256)))

    def save_results_for_bart(self, batch, abs_outputs, abs_rouge_results):
        for filename, abs_output, abs_rouge_result, article, refer in zip(batch['filenames'], abs_outputs,
                                                                          abs_rouge_results,
                                                                          batch['articles'],
                                                                          batch['refers']):

            # Make Filename & Folds
            filename = filename.split('/')[-1].split('.')[-2]
            foldname = f"{format(abs_rouge_result['rouge-1']['f'] * 100, '.2f')}_{filename}"
            os.mkdir(join(self.args['decode_path'], foldname))

            # Save Results
            with open(join(self.args['decode_path'], foldname, 'outputs.txt'), 'w') as file:
                for line in article:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                for line in refer:
                    file.write(f'{line}\n')
                file.write('\n\n\n')
                file.write(f'{abs_output}\n')

            with open(join(self.args['decode_path'], foldname, 'results.json'), 'w') as file:
                json.dump({'abs_rouge_result': abs_rouge_result, }, file)

    def on_predict_epoch_end(self, predict_step_outputs):
        test_abs_rouge_results = self.test_abs_rouge.compute()
        if self.global_rank == 0:
            logger.info(f"abs_rouge_results: {test_abs_rouge_results}")
        self.test_abs_rouge.reset()

        # BERTScore = self.BERTScore.compute()
        # if self.global_rank == 0:
        #     logger.info(f"BERTScore: {BERTScore}")
        # self.BERTScore.reset()
        #
        # MoverScore = self.MoverScore.compute()
        # if self.global_rank == 0:
        #     logger.info(f"MoverScore: {MoverScore}")
        # self.MoverScore.reset()

        if self.args["use_language_score"]:
            test_ext_rouge_results = self.test_ext_rouge.compute()
            if self.global_rank == 0:
                logger.info(f"ext_rouge_results: {test_ext_rouge_results}")
            self.test_ext_rouge.reset()

        if self.args['use_image'] and self.args['use_image_score']:
            for i, precision in enumerate(self.test_image_precision):
                image_precision = precision.compute()
                if self.global_rank == 0:
                    logger.info(f"predict_image_precision@{i + 1}_epoch: {image_precision}")
                precision.reset()
            if self.global_rank == 0:
                simi_score = self.similarity.compute()
                logger.info(f"image-text similarity: {simi_score}")

        print(self.summary_ngram_novel, self.gold_ngram_novel)
        for n in self.summary_ngram_novel.keys():
            # summary_ngram_novel[n] = summary_ngram_novel[n][2]/len(src_lines)
            # gold_ngram_novel[n] = gold_ngram_novel[n][2]/len(src_lines)
            self.summary_ngram_novel[n] = self.summary_ngram_novel[n][0] / self.summary_ngram_novel[n][1]
            self.gold_ngram_novel[n] = self.gold_ngram_novel[n][0] / self.gold_ngram_novel[n][1]
        print(self.summary_ngram_novel, self.gold_ngram_novel)

    def configure_optimizers(self):
        # Grouped Parameters for Weight Decay
        if self.args["weight_decay"] > 0:
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                        self._weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay":
                        0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = self.parameters()

        # Set Optimizer
        if self.args["optimizer"] == "Adam":
            optimizer = optim.Adam(optimizer_grouped_parameters, lr=self.learning_rate)
        elif self.args["optimizer"] == "AdamW":
            optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        else:
            raise ValueError

        # Set Scheduler
        lr_scheduler = None
        if self.args["scheduler"] == "LinearWarmup":
            lr_scheduler = get_linear_schedule_with_warmup(optimizer, self._warmup_iters, self._t_total)
        elif self.args["scheduler"] == "PolynomialWarmup":
            lr_scheduler = get_polynomial_decay_schedule_with_warmup(optimizer, self._warmup_iters,
                                                                     self._t_total)
        elif self.args["scheduler"] == "CosineWarmup":
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, self._warmup_iters, self._t_total)
        elif self.args["scheduler"] == "ConstantWarmup":
            lr_scheduler = get_constant_schedule_with_warmup(optimizer, self._warmup_iters)

        if lr_scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                }
            }
        else:
            return optimizer
