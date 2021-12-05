# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import torch.nn as nn
from omegaconf import DictConfig

from lib.data_utils import Vocab


class LSTMLM(nn.Module):
    def __init__(self,
                 config: DictConfig,
                 embedder,
                 vocab: Vocab
                 ):
        super(LSTMLM, self).__init__()
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.embedder = embedder
        self.emb2hid = nn.Linear(self.embedder.embedding_dim, self.config.hidden_size)
        self.input_dropout = nn.Dropout(p=self.config.input_dropout)
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=False,
            dropout=self.config.lstm_dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.hidden2vocab = nn.Linear(in_features=self.config.hidden_size,
                                      out_features=self.vocab_size)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = self.embedder(seq)  # (b s e)
        x = self.emb2hid(x)
        x = self.input_dropout(x)
        output, _ = self.lstm(x)
        rnn_features = self.dropout(output)
        logits = self.hidden2vocab(rnn_features)
        return logits
