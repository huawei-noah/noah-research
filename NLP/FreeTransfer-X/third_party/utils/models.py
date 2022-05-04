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

import sys
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
# from DataUtils.Common import seed_num
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence # 211215

from transformers.modeling_utils import PreTrainedModel

from utils.models_config import BiLSTMConfig, CNNConfig, MLPConfig
from utils.models_utils import CRF # 211215

from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaClassificationHead
from transformers.modeling_outputs import (
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)

logger = logging.getLogger(__name__)


class NaiveModel(PreTrainedModel):
  """
  An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
  models.
  """
  """
  # config_class = BertConfig
  # load_tf_weights = load_tf_weights_in_bert
  # base_model_prefix = "bert"
  # _keys_to_ignore_on_load_missing = [r"position_ids"]
  """

  '''
  def _init_weights(self, module):
    pass
  '''

  def _init_weights(self, module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.bias is not None:
        module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
      if module.padding_idx is not None:
        module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)


class BiLSTMForSequenceClassification(NaiveModel):
  '''`BiLSTM-Max` arch
  '''
  config_class = BiLSTMConfig
  base_model_prefix = "bilstm"
  def __init__(self, config):
    super().__init__(config)
    self.sent_cls = config.sent_cls # 211101
    self.config = config
    self.num_labels = config.num_labels
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers
    self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
    if config.from_pretrained_embed: # pretrained embedding
      # self.embed.weight.data.copy_(config.pretrained_embeddings)
      self.embed.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)
    self.bilstm = nn.LSTM(
        input_size=config.embed_dim, 
        hidden_size=self.hidden_size // 2, # `// 2`: reduce bidirection `2H` to `H`
        num_layers=config.num_layers, 
        dropout=config.dropout, 
        bidirectional=True, 
        bias=True, # False,
        batch_first=True, # in & out: [T, B, H] -> [B, T, H]
        )

    # self.hidden2label1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
    # self.hidden2label2 = nn.Linear(self.hidden_size // 2, self.num_labels)
    if self.sent_cls:
      self.hidden2label1 = nn.Linear(self.hidden_size, self.hidden_size // 2) # 210830
    else:
      self.hidden2label1 = nn.Linear(self.hidden_size * 2, self.hidden_size // 2) # 210830
    self.hidden2label2 = nn.Linear(self.hidden_size // 2, self.num_labels) # 210830

    self.dropout = nn.Dropout(config.dropout)

    self.batchnorm = nn.BatchNorm1d(self.hidden_size // 2)

    self.init_weights()


  def forward(self,
      input_ids=None,
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # batch_size = len(input_ids)

    embed = self.embed(input_ids) # [B, T, H]
    embed = self.dropout(embed) # 210820: added
    bilstm_out, _ = self.bilstm(embed) # [B, T, 2H]
    bilstm_out = torch.transpose(bilstm_out, 1, 2) # [B, 2H, T], for max-pool
    bilstm_out = F.tanh(bilstm_out)
    # bilstm_out = F.relu(bilstm_out) # TODO: 210823
    # bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2) # BiLSTM-max: kernal-size = T, [B, 2H, 1] -> [B, 2H]
    bilstm_out = F.avg_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)

    if self.sent_cls:
      bilstm_out_concat = bilstm_out
    else:
      embed_1 = self.embed(input_ids_1) # [B, T, H]
      embed_1 = self.dropout(embed_1) # 210820: added
      bilstm_out_1, _ = self.bilstm(embed_1) # [B, T, 2H]
      bilstm_out_1 = torch.transpose(bilstm_out_1, 1, 2) # [B, 2H, T], for max-pool
      bilstm_out_1 = F.tanh(bilstm_out_1)
      # bilstm_out_1 = F.relu(bilstm_out_1) # TODO: 210823
      # bilstm_out_1 = F.max_pool1d(bilstm_out_1, bilstm_out_1.size(2)).squeeze(2) # BiLSTM-max: kernal-size = T, [B, 2H, 1] -> [B, 2H]
      bilstm_out_1 = F.avg_pool1d(bilstm_out_1, bilstm_out_1.size(2)).squeeze(2)
      bilstm_out_concat = torch.cat([bilstm_out, bilstm_out_1], dim=1) # 210830: [B, 4H]

    y = self.hidden2label1(bilstm_out_concat)
    y = self.batchnorm(y) # 210830
    y = F.relu(y) # 210830
    y = self.dropout(y)
    y = self.hidden2label2(y)
    logits = y

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      loss = loss.mean()
      # print("[BiLSTM][forward] logits: {}".format(list(logits.view(-1, self.num_labels))))
      # print("[BiLSTM][forward] labels: {}".format(list(labels.view(-1))))

    output = (logits,)
    return ((loss,) + output) if loss is not None else output


class CNNForSequenceClassification(NaiveModel):
  '''arch from `https://arxiv.org/abs/1408.5882`
  impl from `https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch/blob/master/Model_TextCNN/model.py`

  Config example:
    class Config(object):
      embed_dim = 300
      num_channels = 100
      kernel_size = [3,4,5]
      num_labels = 4
      max_sen_len = 30
      dropout = 0.8
  '''
  config_class = CNNConfig
  base_model_prefix = "cnn"
  def __init__(self, config):
    super().__init__(config)
    self.sent_cls = config.sent_cls
    self.config = config
    
    # Embedding Layer
    self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
    self.token_type_embeddings = nn.Embedding(2, config.embed_dim)
    if config.from_pretrained_embed:
      self.embeddings.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)
    
    # This stackoverflow thread clarifies how conv1d works
    # https://stackoverflow.com/questions/46503816/keras-conv1d-layer-parameters-filters-and-kernel-size/46504997
    self.convs = nn.ModuleList()
    for i in range(len(config.kernel_size)):
      self.convs.append(nn.Sequential(
        nn.Conv1d(in_channels=config.embed_dim, out_channels=config.num_channels, kernel_size=config.kernel_size[i]),
        nn.ReLU(),
        nn.MaxPool1d(config.max_sen_len - config.kernel_size[i]+1)
        )) # NOTE: embed-dim is channel-dim instead of convolution direction. `Conv1d` only convolutes along 1-dim which is timestep-dim in here.
    '''
    self.conv1 = nn.Sequential(
      nn.Conv1d(in_channels=config.embed_dim, out_channels=config.num_channels, kernel_size=config.kernel_size[0]), # TODO: 
      nn.ReLU(),
      nn.MaxPool1d(config.max_sen_len - config.kernel_size[0]+1)
    )
    self.conv2 = nn.Sequential(
      nn.Conv1d(in_channels=config.embed_dim, out_channels=config.num_channels, kernel_size=config.kernel_size[1]),
      nn.ReLU(),
      nn.MaxPool1d(config.max_sen_len - config.kernel_size[1]+1)
    )
    self.conv3 = nn.Sequential(
      nn.Conv1d(in_channels=config.embed_dim, out_channels=config.num_channels, kernel_size=config.kernel_size[2]),
      nn.ReLU(),
      nn.MaxPool1d(config.max_sen_len - config.kernel_size[2]+1)
    )
    '''
    
    self.dropout = nn.Dropout(config.dropout)
    
    # Fully-Connected Layer
    if self.sent_cls:
      self.fc = nn.Linear(config.num_channels*len(config.kernel_size), config.num_labels)
    else:
      self.fc = nn.Linear(config.num_channels*len(config.kernel_size)*2, config.num_labels) # 210917
    
    # Softmax non-linearity
    self.softmax = nn.Softmax()

    
  def forward(self,
      input_ids=None, # [B, T]
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
  ):
    embedded_sent = self.embeddings(input_ids).permute(0,2,1) # [B, E, T], e.g. embedded_sent.shape = (batch_size=64,embed_dim=300,max_sen_len=20)
    if not self.sent_cls:
      embedded_sent_1 = self.embeddings(input_ids_1).permute(0,2,1) # [B, E, T], e.g. embedded_sent.shape = (batch_size=64,embed_dim=300,max_sen_len=20)
    # print("[CNN][forward] token_type_ids: {}".format(token_type_ids))
    # token_type_embed = self.token_type_embeddings(token_type_ids).permute(0,2,1)
    # embedded_sent = embedded_sent + token_type_embed
    
    conv_outs, conv_outs_1 = [], []
    for i in range(len(self.config.kernel_size)):
      # print("[CNN][forward] embedded_sent.device: {}".format(embedded_sent.device))
      # print("[CNN][forward] conv-{}.device: {}".format(i, self.convs[i].device))
      out = self.convs[i](embedded_sent).squeeze(2)
      conv_outs.append(out)
      if not self.sent_cls:
        out_1 = self.convs[i](embedded_sent_1).squeeze(2)
        conv_outs_1.append(out_1)
    
    all_out = torch.cat(conv_outs, 1)
    final_feature_map = self.dropout(all_out)
    if self.sent_cls:
      logits = self.fc(final_feature_map) # [B, C]
    else:
      all_out_1 = torch.cat(conv_outs_1, 1)
      final_feature_map_1 = self.dropout(all_out_1)
      logits = self.fc(torch.cat([final_feature_map, final_feature_map_1], -1)) # [B, C]
    # self.softmax(logits)

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

    output = (logits,)
    return ((loss,) + output) if loss is not None else output


class MLPForSequenceClassification(NaiveModel):
  '''
  config:
    vocab_size
    embed_dim
    hidden_size
    num_hidden_layers 
    dropout
    num_labels
    from_pretrained_embed
    pretrained_embeddings
  '''
  config_class = MLPConfig
  base_model_prefix = "mlp"
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
    if config.from_pretrained_embed:
      self.embeddings.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)

    self.dropout = nn.Dropout(config.dropout)

    self.layer1 = nn.Sequential(
      nn.Linear(in_features=config.embed_dim, out_features=config.hidden_size, bias=True),
      nn.ReLU(),
      nn.Dropout(config.dropout),
    )
    self.layers = nn.Sequential()
    for i in range(config.num_hidden_layers-1):
      self.layers.add_module(str(i*3), nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True))
      self.layers.add_module(str(i*3+1), nn.ReLU())
      self.layers.add_module(str(i*3+2), nn.Dropout(config.dropout))

    # if config.cuda:
    # for i in range(len(self.layers)):
    #   for j in range(len(self.layers[i])):
    #     self.layers[i][j] = self.layers[i][j].cuda()

    # self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True)
    self.fc = nn.Linear(in_features=config.hidden_size*2, out_features=config.num_labels, bias=True) # 210917


  def forward(self,
      input_ids=None, # [B, T]
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
  ):
    emb, emb_1 = self.embeddings(input_ids), self.embeddings(input_ids_1) # [B, T, E]
    emb, emb_1 = self.dropout(emb), self.dropout(emb_1)
    sent_emb, sent_emb_1 = emb.mean(dim=1), emb_1.mean(dim=1) # vector mean

    features, features_1 = self.layer1(sent_emb), self.layer1(sent_emb_1)
    features, features_1 = self.layers(features), self.layers(features_1)

    logits = self.fc(torch.cat([features, features_1], -1))
    # add activation?

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

    output = (logits,)
    return ((loss,) + output) if loss is not None else output


class BiLSTMForTokenClassification(NaiveModel): # 211215: for tag
  '''`BiLSTM` /`BiLSTM-CRF` arch
  impl:
    sequence tagging: `Bidirectional LSTM-CRF Models for Sequence Tagging`
    [USE] `https://github.com/jidasheng/bi-lstm-crf`
  '''
  config_class = BiLSTMConfig
  base_model_prefix = "bilstm"
  def __init__(self, config):
    super().__init__(config)
    # self.sent_cls = config.sent_cls # 211101
    self.config = config
    self.num_labels = config.num_labels
    self.hidden_size = config.hidden_size
    self.num_layers = config.num_layers
    self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.pad_token_id)
    if config.from_pretrained_embed: # pretrained embedding
      # self.embed.weight.data.copy_(config.pretrained_embeddings)
      self.embed.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)
    self.bilstm = nn.LSTM(
        input_size=config.embed_dim, 
        hidden_size=self.hidden_size // 2, # `// 2`: reduce bidirection `2H` to `H`
        num_layers=config.num_layers, 
        dropout=config.dropout, 
        bidirectional=True, 
        bias=True, # False,
        batch_first=True, # in & out: [T, B, H] -> [B, T, H]
        )

    self.linear1 = nn.Linear(self.hidden_size, self.hidden_size // 2)
    self.linear2 = nn.Linear(self.hidden_size // 2, self.num_labels) # 210830

    self.dropout = nn.Dropout(config.dropout)

    self.batchnorm = nn.BatchNorm1d(self.hidden_size // 2)

    self.init_weights()

    self.crf = CRF(self.hidden_size, self.num_labels) # 211215


  def forward(self,
      input_ids=None,
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
      loss_type="ce", # 211216: `crf`, `ce`
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # batch_size = len(input_ids)

    embed = self.embed(input_ids) # [B, T, H]
    embed = self.dropout(embed) # 210820: added
    bilstm_out, _ = self.bilstm(embed) # [B, T, H]
    bilstm_out = F.tanh(bilstm_out)

    y = self.linear1(bilstm_out) # [B, T, H/2]
    y = torch.transpose(y, 1, 2) # [B, H/2, T], for batchnorm
    y = self.batchnorm(y) # 210830
    y = torch.transpose(y, 1, 2) # [B, T, H/2]
    y = F.relu(y) # 210830
    y = self.dropout(y)
    if loss_type == "ce":
      y = self.linear2(y) # [B, T, C]

    logits = y

    loss = None
    if labels is not None:
      if loss_type == "ce": # 211216: naive
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
          active_loss = attention_mask.view(-1) == 1
          active_labels = torch.where(
              active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
              )
          loss = loss_fct(logits.view(-1, self.num_labels), active_labels)
        else:
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif loss_type == "crf": # TODO 211216: support crf
        # 1) __build_features
        # sorted_seq_length, perm_idx = seq_length.sort(descending=True) # TODO: not necessary?
        # 2) crf
        loss = self.crf.loss(y, labels, masks=attention_mask) # TODO: check

      else:
        raise NotImplementedError(f" invalid loss_type = {loss_type}")
      # print("[BiLSTM][forward] logits: {}".format(list(logits.view(-1, self.num_labels))))
      # print("[BiLSTM][forward] labels: {}".format(list(labels.view(-1))))

    output = (logits,)
    return ((loss,) + output) if loss is not None else output

class BiRnnCrf(nn.Module):
  def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, num_rnn_layers=1, rnn="lstm"):
    super(BiRnnCrf, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.vocab_size = vocab_size
    self.tagset_size = tagset_size

    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    RNN = nn.LSTM if rnn == "lstm" else nn.GRU
    self.rnn = RNN(embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
             bidirectional=True, batch_first=True)
    self.crf = CRF(hidden_dim, self.tagset_size)

  def __build_features(self, sentences):
    masks = sentences.gt(0)
    embeds = self.embedding(sentences.long())

    seq_length = masks.sum(1)
    sorted_seq_length, perm_idx = seq_length.sort(descending=True)
    embeds = embeds[perm_idx, :]

    pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
    packed_output, _ = self.rnn(pack_sequence)
    lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
    _, unperm_idx = perm_idx.sort()
    lstm_out = lstm_out[unperm_idx, :]

    return lstm_out, masks

  def loss(self, xs, tags):
    features, masks = self.__build_features(xs)
    loss = self.crf.loss(features, tags, masks=masks)
    return loss

  def forward(self, xs):
    # Get the emission scores from the BiLSTM
    features, masks = self.__build_features(xs)
    scores, tag_seq = self.crf(features, masks)
    return scores, tag_seq


class CNNForTokenClassification(NaiveModel): # 211228: for tag
  '''arch from `https://aclanthology.org/D17-1283/`
  `dilated CNN` arch

  Config example:
    class Config(object):
      embed_dim = 300
      num_channels = 100
      kernel_size = [3,4,5]
      num_labels = 4
      max_sen_len = 30
      dropout = 0.8
  '''
  config_class = CNNConfig
  base_model_prefix = "cnn"
  def __init__(self, config):
    super().__init__(config)
    # self.sent_cls = config.sent_cls
    self.config = config
    num_block = 3 # FIXME: hard-code
    self.layers = [
      {"dilation": 1},
      {"dilation": 1},
      {"dilation": 2}] # FIXME: hard-code
    # NOTE: 3 * 3 = 9 conv layers
    
    # Embedding Layer
    self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
    if config.from_pretrained_embed:
      self.embeddings.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)
    
    net = nn.Sequential()
    norms_1 = nn.ModuleList([LayerNorm(config.max_sen_len) for _ in range(len(self.layers))]) # FIXME: LN along `max_sen_len`?
    norms_2 = nn.ModuleList([LayerNorm(config.max_sen_len) for _ in range(num_block)])
    for i in range(len(self.layers)):
      dilation = self.layers[i]["dilation"]
      kernel_size = config.kernel_size[0] # NOTE: different kernel size? FIXME: different `kernel_size` -> RuntimeError: LN's input changed 128->127, which doesn't match LN's a b size
      single_block = nn.Conv1d(in_channels=config.num_channels,
                   out_channels=config.num_channels,
                   kernel_size=kernel_size,
                   dilation=dilation,
                   padding=kernel_size // 2 + dilation - 1)
      net.add_module("layer%d"%i, single_block)
      net.add_module("relu", nn.ReLU())
      net.add_module("layernorm", norms_1[i])

    self.idcnn = nn.Sequential()

    for i in range(num_block):
      self.idcnn.add_module("block%i" % i, net)
      self.idcnn.add_module("relu", nn.ReLU())
      self.idcnn.add_module("layernorm", norms_2[i])
    
    self.dropout = nn.Dropout(config.dropout)
    
    # Fully-Connected Layer
    self.fc_in = nn.Linear(config.embed_dim, config.num_channels)
    self.fc = nn.Linear(config.num_channels, config.num_channels * 2)
    self.fc_cls = nn.Linear(config.num_channels * 2, config.num_labels)
    
    # Softmax non-linearity
    # self.softmax = nn.Softmax()
    
  def forward(self,
      input_ids=None, # [B, T]
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
      loss_type="ce", # 211216: `crf`, `ce`
  ):
    embedded_sent = self.embeddings(input_ids)
    embedded_sent = self.fc_in(embedded_sent).permute(0, 2, 1) # [B, E, T], e.g. embedded_sent.shape = (batch_size=64,embed_dim=300,max_sen_len=20)
    
    outs = self.idcnn(embedded_sent).permute(0, 2, 1) # [B, T, E]
    outs = self.fc(outs)
    
    logits = self.fc_cls(outs) # [B, T, C]
    logits = self.dropout(logits)

    loss = None
    if labels is not None:
      if loss_type == "ce": # 211216: naive
        loss_fct = CrossEntropyLoss()
        if attention_mask is not None:
          active_loss = attention_mask.view(-1) == 1
          active_labels = torch.where(
              active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
              )
          loss = loss_fct(logits.view(-1, self.config.num_labels), active_labels)
        else:
          loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
      elif loss_type == "crf": # TODO 211216: support crf
        # 1) __build_features
        # sorted_seq_length, perm_idx = seq_length.sort(descending=True) # TODO: not necessary?
        # 2) crf
        pass

      else:
        raise NotImplementedError(f" invalid loss_type = {loss_type}")

    output = (logits,)
    return ((loss,) + output) if loss is not None else output


class LayerNorm(nn.Module):
  def __init__(self, features, eps=1e-6):
    super(LayerNorm, self).__init__()
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True)
    # print(f"x.size() = {x.size()}")
    # print(f"mean.size() = {mean.size()}")
    # print(f"std.size() = {std.size()}")
    # print(f"a.size() = {self.a_2.size()}")
    # print(f"b.size() = {self.b_2.size()}")
    return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class MLPForTokenClassification(NaiveModel): # TODO: for tag
  '''
  config:
    vocab_size
    embed_dim
    hidden_size
    num_hidden_layers 
    dropout
    num_labels
    from_pretrained_embed
    pretrained_embeddings
  '''
  config_class = MLPConfig
  base_model_prefix = "mlp"
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
    if config.from_pretrained_embed:
      self.embeddings.from_pretrained(embeddings=config.pretrained_embeddings, freeze=True)

    self.dropout = nn.Dropout(config.dropout)

    self.layer1 = nn.Sequential(
      nn.Linear(in_features=config.embed_dim, out_features=config.hidden_size, bias=True),
      nn.ReLU(),
      nn.Dropout(config.dropout),
    )
    self.layers = nn.Sequential()
    for i in range(config.num_hidden_layers-1):
      self.layers.add_module(str(i*3), nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size, bias=True))
      self.layers.add_module(str(i*3+1), nn.ReLU())
      self.layers.add_module(str(i*3+2), nn.Dropout(config.dropout))

    # if config.cuda:
    # for i in range(len(self.layers)):
    #   for j in range(len(self.layers[i])):
    #     self.layers[i][j] = self.layers[i][j].cuda()

    # self.fc = nn.Linear(in_features=config.hidden_size, out_features=config.num_labels, bias=True)
    self.fc = nn.Linear(in_features=config.hidden_size*2, out_features=config.num_labels, bias=True) # 210917


  def forward(self,
      input_ids=None, # [B, T]
      input_ids_1=None,
      inputs_embeds=None,
      labels=None,
      return_dict=None,
      attention_mask=None,
      token_type_ids=None,
  ):
    emb, emb_1 = self.embeddings(input_ids), self.embeddings(input_ids_1) # [B, T, E]
    emb, emb_1 = self.dropout(emb), self.dropout(emb_1)
    sent_emb, sent_emb_1 = emb.mean(dim=1), emb_1.mean(dim=1) # vector mean

    features, features_1 = self.layer1(sent_emb), self.layer1(sent_emb_1)
    features, features_1 = self.layers(features), self.layers(features_1)

    logits = self.fc(torch.cat([features, features_1], -1))
    # add activation?

    loss = None
    if labels is not None:
      loss_fct = CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

    output = (logits,)
    return ((loss,) + output) if loss is not None else output


class TinyBertForSequenceClassification(BertPreTrainedModel): # 211115: test distillation
  def __init__(self, config, 
      fit_size=1024, # FIXME: adopted when reload model. `run_distill.py` evaluation should provide this arg
      ):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config

    self.bert = BertModel(config)
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    self.init_weights()

    self.fit_dense = nn.Linear(config.hidden_size, fit_size) # 211115: tinybert

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    is_student=False, # 211115: tinybert
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
      Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
      config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
      If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    # print(f" ### DEBUG ### return_dict = {return_dict}")
    return_dict = False

    outputs = self.bert(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=True, # 211115: tinybert
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )

    print(f" ### DEBUG ### {type(outputs)}")
    print(f" ### DEBUG ### {len(outputs)}")
    #print(f" ### DEBUG ### {outputs}")

    # pooled_output = outputs[1]
    # 211115: tinybert
    sequence_output, pooled_output = outputs[0], outputs[1] # att_output = outputs[4]
    tmp = []
    if is_student:
      for s_id, sequence_layer in enumerate(sequence_output):
        tmp.append(self.fit_dense(sequence_layer))
      # sequence_output = tmp
      outputs = (tmp,) + outputs[1:]

    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
          self.config.problem_type = "single_label_classification"
        else:
          self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)


    if not return_dict:
      print(f" ### DEBUG ### {loss}")
      # output = (logits,) + outputs[2:]
      output = (logits,) + outputs # 211115: tinybert
      return ((loss,) + output) if loss is not None else output

    '''
    return SequenceClassifierOutput(
      loss=loss,
      logits=logits,
      hidden_states=outputs.hidden_states,
      attentions=outputs.attentions,
    )
    '''


class TinyRobertaForSequenceClassification(RobertaPreTrainedModel): # 211115: test distillation
  _keys_to_ignore_on_load_missing = [r"position_ids"]

  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    self.config = config

    self.roberta = RobertaModel(config, add_pooling_layer=False)
    self.classifier = RobertaClassificationHead(config)

    self.init_weights()

  def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
  ):
    r"""
    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
      Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
      config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
      If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    """
    # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    return_dict = False

    outputs = self.roberta(
      input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      position_ids=position_ids,
      head_mask=head_mask,
      inputs_embeds=inputs_embeds,
      output_attentions=True, # 211115: test distillation
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
    )
    sequence_output = outputs[0]
    logits = self.classifier(sequence_output)

    print(f" ### xlmr DEBUG ### {type(outputs)}")
    print(f" ### xlmr DEBUG ### {len(outputs)}")
    #print(f" ### xlmr DEBUG ### {outputs}")

    loss = None
    if labels is not None:
      if self.config.problem_type is None:
        if self.num_labels == 1:
          self.config.problem_type = "regression"
        elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
          self.config.problem_type = "single_label_classification"
        else:
          self.config.problem_type = "multi_label_classification"

      if self.config.problem_type == "regression":
        loss_fct = MSELoss()
        if self.num_labels == 1:
          loss = loss_fct(logits.squeeze(), labels.squeeze())
        else:
          loss = loss_fct(logits, labels)
      elif self.config.problem_type == "single_label_classification":
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      elif self.config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

    if not return_dict:
      print(f" ### xlmr DEBUG ### {loss}")
      # output = (logits,) + outputs[2:]
      output = (logits,) + outputs # 211115: test distillation
      return ((loss,) + output) if loss is not None else output
    '''
      return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
      )
    '''


