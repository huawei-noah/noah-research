# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid

epsilon = 1e-8


class LangEncoder(nn.Module):
    def __init__(self, device, finetune=False, scratch=False):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel, AutoConfig

        self.device = device
        self.modelname = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.model = AutoModel.from_pretrained(self.modelname).to(self.device)
        self.lang_size = 768

    def forward(self, langs):
        try:
            langs = langs.tolist()
        except:
            pass

        with torch.no_grad():
            encoded_input = self.tokenizer(langs, return_tensors="pt", padding=True)
            input_ids = encoded_input["input_ids"].to(self.device)
            attention_mask = encoded_input["attention_mask"].to(self.device)
            lang_embedding = self.model(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state
            lang_embedding = lang_embedding.mean(1)
        return lang_embedding


class LanguageReward(nn.Module):
    def __init__(self, ltype, im_dim, hidden_dim, lang_dim, simfunc=None):
        super().__init__()
        self.ltype = ltype
        self.sim = simfunc
        self.sigm = Sigmoid()
        self.pred = nn.Sequential(
            nn.Linear(im_dim * 2 + lang_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, e0, eg, le):
        info = {}
        return self.pred(torch.cat([e0, eg, le], -1)).squeeze(), info
