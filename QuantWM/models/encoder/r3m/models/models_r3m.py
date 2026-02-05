# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from numpy.core.numeric import full
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.linear import Identity
import torchvision
from torchvision import transforms
from .. import utils
from pathlib import Path
from torchvision.utils import save_image
import torchvision.transforms as T

epsilon = 1e-8


def do_nothing(x):
    return x


class R3M(nn.Module):
    def __init__(
        self,
        device,
        lr,
        hidden_dim,
        size=34,
        l2weight=1.0,
        l1weight=1.0,
        langweight=1.0,
        tcnweight=0.0,
        l2dist=True,
        bs=16,
    ):
        super().__init__()

        self.device = device
        self.use_tb = False
        self.l2weight = l2weight
        self.l1weight = l1weight
        self.tcnweight = tcnweight  ## Weight on TCN loss (states closer in same clip closer in embedding)
        self.l2dist = l2dist  ## Use -l2 or cosine sim
        self.langweight = langweight  ## Weight on language reward
        self.size = size  ## Size ResNet or ViT
        self.num_negatives = 3
        
        self.latent_ndim = 1
        self.emb_dim = 512
        self.name = "r3m"

        ## Distances and Metrics
        self.cs = torch.nn.CosineSimilarity(1)
        self.bce = nn.BCELoss(reduce=False)
        self.sigm = Sigmoid()

        params = []
        ######################################################################## Sub Modules
        ## Visual Encoder
        if size == 18:
            self.outdim = 512
            self.convnet = torchvision.models.resnet18(pretrained=False)
        elif size == 34:
            self.outdim = 512
            self.convnet = torchvision.models.resnet34(pretrained=False)
        elif size == 50:
            self.outdim = 2048
            self.convnet = torchvision.models.resnet50(pretrained=False)
        elif size == 0:
            from transformers import AutoConfig

            self.outdim = 768
            self.convnet = AutoModel.from_config(
                config=AutoConfig.from_pretrained("google/vit-base-patch32-224-in21k")
            ).to(self.device)

        if self.size == 0:
            self.normlayer = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        else:
            self.normlayer = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        self.convnet.fc = Identity()
        self.convnet.train()
        params += list(self.convnet.parameters())

        ## Language Reward
        if self.langweight > 0.0:
            ## Pretrained DistilBERT Sentence Encoder
            from r3m.models.models_language import LangEncoder, LanguageReward

            self.lang_enc = LangEncoder(self.device, 0, 0)
            self.lang_rew = LanguageReward(
                None, self.outdim, hidden_dim, self.lang_enc.lang_size, simfunc=self.sim
            )
            params += list(self.lang_rew.parameters())
        ########################################################################

        ## Optimizer
        self.encoder_opt = torch.optim.Adam(params, lr=lr)

    def get_reward(self, e0, es, sentences):
        ## Only callable is langweight was set to be 1
        le = self.lang_enc(sentences)
        return self.lang_rew(e0, es, le)

    ## Forward Call (im --> representation)
    def forward(self, obs, num_ims=1, obs_shape=[3, 224, 224]):
        if obs_shape != [3, 224, 224]:
            preprocess = nn.Sequential(
                transforms.Resize(256),
                transforms.CenterCrop(224),
                self.normlayer,
            )
        else:
            preprocess = nn.Sequential(
                self.normlayer,
            )

        ## Input must be [0, 1], [3,244,244]
        dims = len(obs.shape)
        orig_shape = obs.shape
        if dims == 3:
            obs = obs.unsqueeze(0)
        elif dims > 4:
            obs = obs.reshape(-1, *orig_shape[-3:])
        obs_p = preprocess(obs)
        h = self.convnet(obs_p)
        if dims == 3:
            h = h.squeeze(0)
        elif dims > 4:
            h = h.reshape(*orig_shape[:-3], -1)
        h = h.unsqueeze(1) 
        return h

    def sim(self, tensor1, tensor2):
        if self.l2dist:
            d = -torch.linalg.norm(tensor1 - tensor2, dim=-1)
        else:
            d = self.cs(tensor1, tensor2)
        return d
