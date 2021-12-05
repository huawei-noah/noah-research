# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from argparse import Namespace
from dataclasses import is_dataclass

from pytorch_lightning.core.lightning import LightningModule
import torch
from torch.nn import Linear, Dropout, Module
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer

from configs.config import ExpConfig
from lib.utils import compute_ood_metrics, get_dataclass_params, get_weighted_sampler
from lib.score_functions import AbstractMahalanobisScore, LogitsScoreFunction


class DistilBertWrapper(Module):
    def __init__(self, distil_bert_model):
        super(DistilBertWrapper, self).__init__()
        self.encoder = distil_bert_model

    @property
    def config(self):
        return self.encoder.config

    def forward(self, input_ids, attention_mask, inputs_embeds):
        output = self.encoder(input_ids=input_ids,
                              attention_mask=attention_mask,
                              inputs_embeds=inputs_embeds)[0]
        pooled_output = output[:, 0]
        return output, pooled_output


class TransformerClassifier(Module):
    def __init__(self, transformer, hidden_dropout_prob, num_labels):
        super().__init__()
        self.num_labels = num_labels

        self.transformer = transformer
        self.dropout = Dropout(hidden_dropout_prob)
        self.classifier = Linear(self.transformer.config.hidden_size, self.num_labels)
        self.feats = None

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def forward(self, seq=None, att_idxs=None, inputs_embeds=None):
        bert_feats = self.transformer(input_ids=seq, attention_mask=att_idxs, inputs_embeds=inputs_embeds)
        self.feats = bert_feats[1]
        pooled_output = self.dropout(self.feats)
        return self.classifier(pooled_output)


class TransformerModule(LightningModule):
    def __init__(self,
                 hparams: ExpConfig,
                 transformer,
                 score_fn,
                 collate_fn,
                 tokenizer: BertTokenizer,
                 train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.score_fn = score_fn
        if is_dataclass(hparams):
            self.config: ExpConfig = hparams
            self.hparams = get_dataclass_params(hparams)
        else:
            self.config = Namespace(**hparams)
            self.hparams = hparams
        self.collate_fn = collate_fn
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self.saved_ood_scores = None

    def forward(self, *args, **kwargs):
        return self.transformer.forward(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        seq,  att_idxs, labels, is_ood = batch['seq'], batch['attention_mask'], batch['labels'], batch['is_ood']
        logits = self.transformer(seq, att_idxs)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(-1) == labels).float().mean()
        logs = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }
        return {
            'loss': loss,
            'log': logs
        }

    def training_epoch_end(self, outputs):
        if isinstance(self.score_fn, AbstractMahalanobisScore):
            train_labels, train_feats = self._get_train_outputs()
            self.score_fn.update(train_feats=train_feats, train_labels=train_labels)
        avg_accuracy = torch.stack([x['train_accuracy'] for x in outputs]).mean().item()
        return {
            'progress_bar': {
                'train/avg_accuracy': avg_accuracy
            }
        }

    def on_epoch_end(self) -> None:
        self.trainer.test()
        self.trainer.testing = False

    def _get_train_outputs(self):
        train_outputs = []
        self.eval()
        with torch.no_grad():
            data_loader = DataLoader(self._train_dataset, batch_size=self.config.batch_size,
                                     collate_fn=self.collate_fn, shuffle=False, num_workers=0)
            for idx, batch in enumerate(data_loader):
                seq, att_idxs, labels, is_ood = batch['seq'], batch['attention_mask'], batch['labels'], batch['is_ood']
                logits = self.transformer(seq.to(self.device), att_idxs.to(self.device))
                feats = self.transformer.feats
                res = {
                    'labels': labels,
                    'logits': logits,
                    'feats': feats
                }
                train_outputs.append(res)
        self.train(True)
        train_labels = torch.cat([x['labels'] for x in train_outputs]).cpu().numpy().astype(int)
        train_feats = torch.cat([x['feats'] for x in train_outputs])
        return train_labels, train_feats

    def validation_step(self, batch, batch_idx):
        seq, att_idxs, labels, is_ood = batch['seq'], batch['attention_mask'], batch['labels'], batch['is_ood']
        logits = self.transformer(seq, att_idxs)
        res = {
            'labels': labels,
            'is_ood': is_ood
        }

        if isinstance(self.score_fn, LogitsScoreFunction):
            ood_scores = self.score_fn(logits)
        elif isinstance(self.score_fn, AbstractMahalanobisScore):
            ood_scores = self.score_fn(self.transformer.feats)
        else:
            raise ValueError(f'Unknown score function class: {self.score_fn.__class__}')
        res['ood_scores'] = ood_scores
        res['logits'] = logits
        return res

    def test_step(self, batch, batch_idx):
        res = self.validation_step(batch, batch_idx)
        self.saved_ood_scores = res["ood_scores"]
        return res

    def _eval(self, outputs, prefix):
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        is_ood = torch.cat([x['is_ood'] for x in outputs]).cpu().numpy().astype(int)
        ood_scores = torch.cat([x['ood_scores'] for x in outputs]).cpu().numpy()

        in_logits = logits[is_ood == 0]
        in_labels = labels[is_ood == 0]
        loss = 0.0
        accuracy = 0.0
        if in_labels.numel():
            loss = F.cross_entropy(in_logits, in_labels)
            accuracy = (in_logits.argmax(dim=-1) == in_labels).float().mean().item()

        ood_metrics = compute_ood_metrics(ood_scores, is_ood, prefix=f'{prefix}/')
        return loss, accuracy, ood_metrics, ood_scores

    def validation_epoch_end(self, outputs):
        loss, accuracy, ood_metrics, ood_scores = self._eval(outputs, "val")
        return {
            f'val_loss': loss,
            'log': {
                **ood_metrics,
                f'val/accuracy': accuracy,
                f'val/loss': loss,
                f'val/mean_ood_score': ood_scores.mean()
            }
        }

    def test_epoch_end(self, outputs):
        loss, accuracy, ood_metrics, ood_scores = self._eval(outputs, "test")
        return {
            f'test_loss': loss,
            'log': {
                **ood_metrics,
                f'test/accuracy': accuracy,
                f'test/loss': loss,
                f'test/mean_ood_score': ood_scores.mean()
            }
        }

    def configure_optimizers(self):
        return Adam(self.transformer.parameters(), self.config.lr)

    def train_dataloader(self) -> DataLoader:
        data_loader_params = {
            'collate_fn': self.collate_fn,
            'batch_size': self.config.batch_size,
            'num_workers': self.config.n_workers
        }
        if self.config.balance_classes:
            sampler = get_weighted_sampler(self._train_dataset)
            data_loader_params['sampler'] = sampler
        else:
            data_loader_params['shuffle'] = True
        return DataLoader(
            self._train_dataset, **data_loader_params
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val_dataset, collate_fn=self.collate_fn,
                          batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.n_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_dataset, collate_fn=self.collate_fn,
                          batch_size=self.config.batch_size,
                          shuffle=False, num_workers=self.config.n_workers)
