# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.data_utils import Vocab
from lib.modules.embedder import Embedder, GloveEmbedder


class BOWClassifier(nn.Module):
    def __init__(self, vocab, config):
        super(BOWClassifier, self).__init__()
        self.vocab = vocab
        self.config = config.classifier
        self.linear = nn.Linear(len(self.vocab), self.config.num_classes)

    def forward(self, x):
        return self.linear(x)


class CBOWClassifier(nn.Module):
    def __init__(self, vocab, config):
        super(CBOWClassifier, self).__init__()
        self.vocab = vocab
        self.config = config.classifier
        self.embeddings = GloveEmbedder(self.vocab, config.embedder)
        self.linear = nn.Linear(self.embeddings.embedding_dim, self.config.num_classes)
        self.penultimate_size = self.embeddings.embedding_dim

    def forward(self, seq):
        embed = self.embeddings(seq)  # BxSxE
        self.feats = embed.mean(1)  # BxE
        logits = self.linear(self.feats)
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self,
                 config: DictConfig,
                 vocab: Vocab
                 ):
        super(LSTMClassifier, self).__init__()
        self.config = config.classifier
        self.vocab = vocab
        self.embedder = GloveEmbedder(self.vocab, config.embedder)
        self.emb2hid = nn.Linear(self.embedder.embedding_dim, self.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            bidirectional=self.config.bidirectional,
            dropout=self.config.lstm_dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(p=self.config.dropout)
        self.input_dropout = nn.Dropout(p=self.config.lstm_dropout)
        self.rnn_features_size = (2 if self.config.bidirectional else 1) * self.config.hidden_size
        self.output_layer = nn.Linear(in_features=self.rnn_features_size, out_features=self.config.num_classes)
        self.penultimate_size = self.rnn_features_size

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # TODO: maybe use packed sequence
        x = self.embedder(seq)  # (b s e)
        x = self.input_dropout(x)
        x = self.emb2hid(x)
        output, _ = self.lstm(x)
        self.feats = output[:, -1, :].squeeze(1)
        rnn_features = self.dropout(self.feats)
        logits = self.output_layer(rnn_features)
        return logits


class OptionalLayer(nn.Module):
    def __init__(self, layer, use_layer=True):
        super(OptionalLayer, self).__init__()
        self.layer = layer
        self.use_layer = use_layer

    def forward(self, x):
        if self.use_layer:
            return self.layer(x)
        return x


class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, p_drop, activation,
                 use_batch_norm=True):
        """
        BLock with linear layer, optinonal batch normalization layer,
        activation function and dropout.

        Args:
            in_features: input dimension size
            out_features: out dimension size
            p_drop: dropout probability
            activation: activation function
            use_batch_norm: whether to use batch normalization
        """
        super(FCBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
            OptionalLayer(nn.BatchNorm1d(out_features), use_batch_norm),
            activation,
            nn.Dropout(p_drop),
        )

    def forward(self, x):
        return self.layers(x)


class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for text classification.

    Args:
        config: omegaconf configuration object
        vocab: vocabulary object
    """
    def __init__(self, config: DictConfig, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self.config = config.classifier
        self.embeddings = GloveEmbedder(self.vocab, config.embedder)
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=self.config.num_channels,
                kernel_size=(ks, self.embeddings.embedding_dim),
            )
            for ks in self.config.ks_list
        ])
        self.dropout = nn.Dropout(self.config.dropout)

        num_cnn_features = len(self.config.ks_list) * self.config.num_channels

        self.interim_layers = nn.ModuleList()
        prev_num_features = num_cnn_features
        for hid_size in self.config.hid_list:
            self.interim_layers.append(
                FCBlock(in_features=prev_num_features,
                        out_features=hid_size,
                        p_drop=self.config.interim_dropout,
                        activation=nn.LeakyReLU(),
                        use_batch_norm=self.config.use_batch_norm)
            )
            prev_num_features = hid_size

        output_in = self.config.hid_list[-1] if self.config.hid_list else num_cnn_features
        self.penultimate_size = output_in
        self.output = nn.Linear(output_in, self.config.num_classes)

    def forward(self, seq: torch.Tensor):
        embed = self.embeddings(seq)  # BxSxE
        embed = embed.unsqueeze(1)  # BxiCxSxE

        # Apply convolutions and max pooling over time
        feature_maps = [F.relu(conv(embed)).squeeze(3) for conv in self.convs]  # [N x oC x S]*Nk
        pooled_feature_maps = [F.max_pool1d(feature_map, feature_map.size(2)).squeeze(2) for feature_map in feature_maps]

        # Concatenate feature maps
        self.feats = torch.cat(pooled_feature_maps, 1)
        features = self.dropout(self.feats)
        for layer in self.interim_layers:
            features = layer(features)
            # change penultimate features
            self.feats = features
        logits = self.output(features)
        return logits
