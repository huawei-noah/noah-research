# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from typing import List, Dict, Any

from pytorch_lightning import LightningModule
import torch
import torch.nn.functional as F
from pytorch_lightning.core.decorators import data_loader
from torch.utils.data import DataLoader

from lib.score_functions import LogitsScoreFunction, AbstractMahalanobisScore
from lib.utils import make_hparams, compute_ood_metrics, get_weighted_sampler, compute_l2_penalty


class ClassificationModule(LightningModule):
    def __init__(self, config, classifier, collate_fn, score_fn,
                 train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.config = config
        self.classifier = classifier
        self.score_fn = score_fn
        assert self.config.temperature > 0, "Temperature should be positive"
        self.hparams = make_hparams(config)
        self.collate_fn = collate_fn
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset

    def forward(self, inputs):
        return self.classifier(inputs)

    def training_step(self, batch, batch_idx):
        seq, labels, is_ood = batch
        logits = self.classifier(seq)
        loss = self.cross_entropy_loss(logits=logits, labels=labels)
        loss += compute_l2_penalty(self.classifier, self.config.classifier.l2_coeff)
        accuracy = (logits.argmax(-1) == labels).float().mean()
        logs = {
            'train_loss': loss,
            'train_accuracy': accuracy
        }
        return {
            'loss': loss,
            'log': logs
        }

    def _get_train_outputs(self):
        train_outputs = []
        self.eval()
        with torch.no_grad():
            data_loader = DataLoader(self._train_dataset,
                                     batch_size=self.config.training.train_batch_size,
                                     collate_fn=self.collate_fn, shuffle=False,
                                     num_workers=0)
            for idx, batch in enumerate(data_loader):
                seq, labels, is_ood = batch
                logits = self.classifier(seq.to(self.device))
                feats = self.classifier.feats
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

    def validation_step(self, batch, batch_idx):
        return self._compute_step(batch)

    def validation_epoch_end(self, outputs):
        return self._eval(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._eval(outputs, 'test')

    def test_step(self, batch, batch_idx):
        return self._compute_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.config.training.optimizer.lr,
            betas=(self.config.training.optimizer.get('beta1', 0.9),
                   self.config.training.optimizer.get('beta2', 0.999))
        )
        return optimizer

    def cross_entropy_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

    @data_loader
    def train_dataloader(self):
        data_loader_params = {
            'collate_fn': self.collate_fn,
            'batch_size': self.config.training.train_batch_size,
            'num_workers': self.config.training.num_workers
        }
        if self.config.training.balance_classes:
            sampler = get_weighted_sampler(self._train_dataset)
            data_loader_params['sampler'] = sampler
        else:
            data_loader_params['shuffle'] = self.config.training.shuffle
        return DataLoader(
            self._train_dataset, **data_loader_params
        )

    @data_loader
    def val_dataloader(self):
        return DataLoader(
            self._val_dataset, batch_size=self.config.training.val_batch_size,
            collate_fn=self.collate_fn, shuffle=False,
            num_workers=self.config.training.num_workers
        )

    @data_loader
    def test_dataloader(self):
        return DataLoader(
            self._test_dataset, batch_size=self.config.training.val_batch_size,
            collate_fn=self.collate_fn, shuffle=False,
            num_workers=self.config.training.num_workers
        )

    def _compute_step(self, batch):
        seq, labels, is_ood = batch
        logits = self.classifier(seq)
        if isinstance(self.score_fn, LogitsScoreFunction):
            ood_scores = self.score_fn(logits)
        elif isinstance(self.score_fn, AbstractMahalanobisScore):
            ood_scores = self.score_fn(self.classifier.feats)
        else:
            raise ValueError(f'Unknown score function class: {self.score_fn.__class__}')
        return {
            'logits': logits,
            'labels': labels,
            'is_ood': is_ood,
            'ood_scores': ood_scores
        }

    def _eval(self, outputs: List[Dict[str, Any]], step_name: str):
        assert step_name in ('val', 'test')
        logits = torch.cat([x['logits'] for x in outputs])
        labels = torch.cat([x['labels'] for x in outputs])
        is_ood = torch.cat([x['is_ood'] for x in outputs])

        in_logits = logits[is_ood == 0]
        in_labels = labels[is_ood == 0]
        loss = self.cross_entropy_loss(logits=in_logits, labels=in_labels)
        accuracy = (in_logits.argmax(dim=-1) == in_labels).float().mean()

        ood_scores = torch.cat([x['ood_scores'] for x in outputs]).cpu().numpy()
        is_ood = is_ood.cpu().numpy().astype(int)

        ood_metrics = compute_ood_metrics(ood_scores, is_ood, prefix=f'{step_name}/')
        return {
            f'{step_name}_loss': loss,
            'log': {
                **ood_metrics,
                f'{step_name}/accuracy': accuracy,
                f'{step_name}/loss': loss
            }
        }
