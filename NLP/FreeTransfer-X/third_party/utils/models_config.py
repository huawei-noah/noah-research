# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

from transformers.configuration_utils import PretrainedConfig

class BiLSTMConfig(PretrainedConfig):
  model_type = "bilstm"
  def __init__(self, 
      vocab_size=119547,
      num_labels=3,
      embed_dim=768,
      hidden_size=768,
      num_layers=2,
      pad_token_id=0,
      from_pretrained_embed=False,
      pretrained_embeddings=None,
      dropout=0.5,
      initializer_range=0.02,
      **kwargs
      ):
    super().__init__(pad_token_id=pad_token_id, **kwargs)
    '''
    self.num_labels = num_labels
    '''
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # self.pad_token_id = pad_token_id
    self.from_pretrained_embed = from_pretrained_embed
    self.pretrained_embeddings = pretrained_embeddings
    self.dropout = dropout
    self.initializer_range=initializer_range


class CNNConfig(PretrainedConfig):
  model_type = "cnn"
  def __init__(self, 
      vocab_size=119547,
      num_labels=3,
      embed_dim=768,
      num_channels=256,
      kernel_size=[3, 4, 5],
      max_sen_len=128, # NOTE: should be consistent with preprocessing
      pad_token_id=0,
      from_pretrained_embed=False,
      pretrained_embeddings=None,
      dropout=0.5,
      initializer_range=0.02,
      **kwargs
      ):
    super().__init__(pad_token_id=pad_token_id, **kwargs)
    '''
    self.num_labels = num_labels
    # self.pad_token_id = pad_token_id
    '''
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.num_channels = num_channels
    self.kernel_size = kernel_size
    self.max_sen_len = max_sen_len
    self.dropout = dropout
    self.from_pretrained_embed = from_pretrained_embed
    self.pretrained_embeddings = pretrained_embeddings
    self.initializer_range = initializer_range


class MLPConfig(PretrainedConfig):
  model_type = "mlp"
  def __init__(self, 
      vocab_size=119547,
      num_labels=3,
      embed_dim=768,
      hidden_size=768,
      num_hidden_layers=3,
      pad_token_id=0,
      from_pretrained_embed=False,
      pretrained_embeddings=None,
      dropout=0.5,
      initializer_range=0.02,
      **kwargs
      ):
    super().__init__(pad_token_id=pad_token_id, **kwargs)
    '''
    self.num_labels = num_labels
    '''
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.dropout = dropout
    # self.pad_token_id = pad_token_id
    self.from_pretrained_embed = from_pretrained_embed
    self.pretrained_embeddings = pretrained_embeddings
    self.initializer_range=initializer_range


