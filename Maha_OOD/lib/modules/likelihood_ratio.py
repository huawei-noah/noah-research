# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.from typing import List, Dict, Any

import numpy as np
from pytorch_lightning import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.core.decorators import data_loader
from torch.distributions import Categorical
from torch.utils.data import DataLoader, WeightedRandomSampler

from lib.data_utils import Vocab
from lib.modules.embedder import GloveEmbedder
from lib.modules.language_model import LSTMLM
from lib.utils import make_hparams, compute_ood_metrics, compute_l2_penalty


class LikelihoodratioModule(LightningModule):
    def __init__(self, config, vocab: Vocab, collate_fn,
                 train_dataset=None, val_dataset=None, test_dataset=None):
        super().__init__()
        self.config = config
        self.hparams = make_hparams(config)
        self.vocab = vocab

        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self.collate_fn = collate_fn

        lm_embedder = GloveEmbedder(config=config.embedder, vocab=vocab)
        self.lm = LSTMLM(self.config.lm, lm_embedder, self.vocab)

        background_lm_embedder = GloveEmbedder(config=config.embedder, vocab=vocab)
        self.background_lm = LSTMLM(self.config.background_lm, background_lm_embedder, self.vocab)

        self.p_noise = config.p_noise
        self.noise_type = config.get('noise_type', 'uniform')
        assert self.noise_type in ('uniform', 'unigram', 'uniroot')
        if self.noise_type == 'unigram':
            word_freq = torch.tensor(self.vocab.word_counts, dtype=torch.float)
        elif self.noise_type == 'uniroot':
            word_freq = torch.tensor(self.vocab.word_counts, dtype=torch.float).sqrt()
        elif self.noise_type == 'uniform':
            word_freq = torch.tensor([1] * len(self.vocab), dtype=torch.float)
            word_freq[self.vocab.special_token_idxs] = 0
        else:
            raise ValueError(f'Noise type {self.noise_type} is not supported!\n'
                             f'Choose one from: uniform, unigram, uniroot')
        self.word_distr = Categorical(probs=word_freq / word_freq.sum())

        self.lm_l2_coeff = self.config.lm.get('l2_coeff', 0)
        self.background_lm_l2_coeff = self.config.background_lm.get('l2_coeff', 0)
        self.loss_xe = nn.NLLLoss(ignore_index=self.vocab.pad_idx, reduction='none')

    def forward(self, inputs):
        pass

    def training_step(self, batch, batch_idx, optimizer_idx):
        seq, labels, is_ood = batch
        logs = {}
        if optimizer_idx == 0:
            # Main LM model training
            inp = seq[:, :-1]
            trg = seq[:, 1:]
            logits = self.lm(inp)

            loss = self.cross_entropy_loss(logits, trg)
            if self.lm_l2_coeff > 0:
                l2_penalty = compute_l2_penalty(module=self.lm, l2_coeff=self.lm_l2_coeff)
                loss += l2_penalty
            logs['train/lm_loss'] = loss
        else:
            # Background LM model training
            seq = self.add_noise(seq)
            inp = seq[:, :-1]
            trg = seq[:, 1:]
            logits = self.background_lm(inp)
            loss = self.cross_entropy_loss(logits, trg)
            if self.background_lm_l2_coeff > 0:
                l2_penalty = compute_l2_penalty(module=self.background_lm,
                                                l2_coeff=self.background_lm_l2_coeff)
                loss += l2_penalty
            logs['train/back_lm_loss'] = loss
        return {
            'loss': loss,
            'log': logs
        }

    def add_noise(self, seq):
        mask = (torch.rand(seq.size(), device=seq.device) < self.p_noise)
        spc_mask = ~(seq == self.vocab.pad_idx) & ~(seq == self.vocab.eos_idx) & ~(seq == self.vocab.bos_idx)
        mask = mask & spc_mask
        new_values = self.word_distr.sample(seq.size()).to(seq.device)
        seq[mask] = new_values[mask]
        return seq

    def validation_step(self, batch, batch_idx):
        seq, labels, is_ood = batch
        ood_scores = self._compute_ood_score(seq)
        res = {
            'is_ood': is_ood,
            'labels': labels,
            'ood_scores': ood_scores
        }

        if torch.all(is_ood == 1):
            res['loss'] = float('inf')
            return res

        seq = seq[is_ood != 1]
        inp, trg = seq[:, :-1], seq[:, 1:]
        logits = self.lm(inp)
        loss = self.cross_entropy_loss(logits, trg)
        res['loss'] = loss
        return res

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs if x['loss'] != float('inf')]).mean()
        res = self._eval(outputs, step_name='val')
        res['log']['val_loss'] = val_loss
        return res

    def _compute_ood_score(self, seq):
        inp, trg = seq[:, :-1], seq[:, 1:]
        background_lm_logp_word = self._get_word_log_probs(self.background_lm, inp, trg)
        lm_logp_word = self._get_word_log_probs(self.lm, inp, trg)
        logp_diff = lm_logp_word - background_lm_logp_word
        logp_diff[trg == self.vocab.pad_idx] = 0
        ood_scores = logp_diff.sum(-1)
        return ood_scores

    @staticmethod
    def _get_word_log_probs(model, inp, trg):
        logits = model(inp)
        logp = F.log_softmax(logits, dim=-1)
        logp_word = torch.gather(logp, dim=-1, index=trg.unsqueeze(2)).squeeze(2)
        return logp_word

    def test_step(self, batch, batch_idx):
        seq, labels, is_ood = batch
        ood_scores = self._compute_ood_score(seq)
        res = {
            'is_ood': is_ood,
            'labels': labels,
            'ood_scores': ood_scores
        }
        val_res = self.validation_step(batch, batch_idx)
        res['loss'] = val_res['loss']
        if 'pred' in val_res:
            res['pred'] = val_res['pred']
        return res

    def test_epoch_end(self, outputs):
        return self._eval(outputs, 'test')

    def configure_optimizers(self):
        self.lm_optimizer = torch.optim.Adam(
            self.lm.parameters(),
            lr=self.config.training.optimizer.lr,
            betas=(self.config.training.optimizer.get('beta1', 0.9),
                   self.config.training.optimizer.get('beta2', 0.999))
        )

        self.background_lm_optimizer = torch.optim.Adam(
            self.background_lm.parameters(),
            lr=self.config.training.back_optimizer.lr,
            betas=(self.config.training.back_optimizer.get('beta1', 0.9),
                   self.config.training.back_optimizer.get('beta2', 0.999))
        )

        if self.config.training.scheduler.use:
            lm_sched = {
                'scheduler': torch.optim.lr_scheduler.ExponentialLR(
                                    optimizer=self.lm_optimizer,
                                    gamma=self.config.training.scheduler.gamma
                             ),
                'interval': 'step'
            }
            return [self.lm_optimizer, self.background_lm_optimizer], [lm_sched]

        return self.lm_optimizer, self.background_lm_optimizer

    def cross_entropy_loss(self, logits, trg_seq):
        batch_size, seq_len = trg_seq.size()
        logp = F.log_softmax(logits, dim=-1)
        logp_words = logp.view(-1, len(self.vocab))
        trg_lengths = (trg_seq == self.vocab.eos_idx).nonzero()[:, 1]
        target = trg_seq.contiguous().view(-1)
        loss_xe = self.loss_xe(logp_words, target)
        loss_xe = loss_xe.view(batch_size, seq_len).sum(dim=-1) / trg_lengths
        loss_xe = loss_xe.mean()
        # loss_xe = loss_xe.view(batch_size, seq_len).sum(dim=1).mean()
        return loss_xe

    @data_loader
    def train_dataloader(self):
        data_loader_params = {
            'collate_fn': self.collate_fn,
            'batch_size': self.config.training.train_batch_size,
            'num_workers': self.config.training.num_workers
        }
        if self.config.training.get('balance_classes', False):
            target = self._train_dataset.vectorized_labels
            class_sample_count = np.unique(target, return_counts=True)[1]
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target])
            samples_weight = torch.from_numpy(samples_weight).double()
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
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

    def _eval(self, outputs: List[Dict[str, Any]], step_name: str):
        is_ood = torch.cat([x['is_ood'] for x in outputs]).cpu().numpy().astype(int)
        ood_scores = -torch.cat([x['ood_scores'] for x in outputs]).cpu().numpy()
        avg_loss = torch.stack([x['loss'] for x in outputs if x['loss'] != float('inf')]).mean().item()

        ood_metrics = compute_ood_metrics(ood_scores, is_ood, prefix=f'{step_name}/')
        return {
            f'{step_name}_loss': avg_loss,
            'log': ood_metrics
        }
