# coding=utf-8
# Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from attention import Attention
from transformers import BertPreTrainedModel
from transformers import BertModel

class BertDictForNerSoftmax(BertPreTrainedModel):
    def __init__(self, config, args, num_ner_label=None, num_dict_label=None):
        super(BertDictForNerSoftmax, self).__init__(config)
        self.LAMBDA = args.LAMBDA
        self.config = config
        self.max_seq_length = args.train_max_seq_length
        self.max_dict_num = args.max_dict_num
        self.num_labels = num_ner_label
        self.use_subword = args.use_subword
        self.bert = BertModel(config)
        self.dict_emb = nn.Embedding(num_dict_label, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.text_atts = Attention(
            config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob,
            config.hidden_dropout_prob
        )

        self.dict_cls_atts = Attention(
            config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob,
            config.hidden_dropout_prob
        )

        self.dict_atts = Attention(
            config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob,
            config.hidden_dropout_prob
        )

        self.output_atts = Attention(
            config.hidden_size, config.num_attention_heads, config.attention_probs_dropout_prob,
            config.hidden_dropout_prob
        )

        self.slot_classifier = nn.Linear(config.hidden_size, num_ner_label)
        self.dict_classifier = nn.Linear(config.hidden_size, 2)
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.slot_loss_fct = nn.CrossEntropyLoss()
        self.dict_loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(
        self, 
        input_ids, attention_mask=None, token_type_ids=None, 
        labels=None, input_len=None, 
        name_ids=None, dict_labels=None, dict_mask=None, loss_mask=None):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        bz = sequence_output.size(0)

        # batch * dict_num * seq_len * hidden_size -> (batch * dict_num) * seq_len * hidden_size
        dict_embs = self.dict_emb(name_ids)
        dict_embs = dict_embs.contiguous().view(-1, self.max_seq_length, self.config.hidden_size)

        # batch * seq_len * hidden_size -> (batch * dict_num) * seq_len * hidden_size
        seq_query = sequence_output.unsqueeze(1).repeat(1, self.max_dict_num, 1, 1).view(
            -1, self.max_seq_length, self.config.hidden_size)

        seq_dict_mask = attention_mask.unsqueeze(1).repeat(1, self.max_dict_num, 1).view(-1, self.max_seq_length)

        # (batch * dict_num) * seq_len * hidden_size
        dict_features = seq_query + dict_embs
        dict_features = self.text_atts(dict_features, dict_features, dict_features, seq_dict_mask)

        # batch * dict_num * hidden_size
        dict_cls_features = dict_features[:, 0, :].squeeze().view(-1, self.max_dict_num, self.config.hidden_size)
        dict_features_cls = self.dict_cls_atts(dict_cls_features, dict_cls_features, dict_cls_features, dict_mask)
        logits_dict = self.dict_classifier(self.dropout(dict_features_cls))

        dict_cls_mask = torch.argmax(logits_dict, dim=-1)
        # batch * (dict_num+1)
        dict_cls_mask = torch.cat(
            [torch.ones(bz, 1, dtype=torch.long, device=self.torch_device), dict_cls_mask], dim=-1
        )

        # (bz * seq_length) * dict_num
        dict_cls_mask = dict_cls_mask.unsqueeze(1).repeat(1, self.max_seq_length, 1).view(-1, self.max_dict_num + 1)

        # dict_features (batch * dict_num) * seq_len * hidden_size
        dict_features_seq = dict_features.view(
            -1, self.max_dict_num, self.max_seq_length, self.config.hidden_size
        ).permute(0, 2, 1, 3).contiguous().view(-1, self.max_dict_num, self.config.hidden_size)

        # (batch * seq_len) * hidden_size
        seq_query = sequence_output.view(-1, self.config.hidden_size)
        # (batch * seq_len) * 1 * hidden_size
        seq_query = seq_query.unsqueeze(dim=1)

        # dict_features batch * seq_len * (dict_num+1)  * hidden_size
        dict_features_seq = torch.cat([seq_query, dict_features_seq], dim=1)
        dict_features_seq = self.dict_atts(dict_features_seq, dict_features_seq, dict_features_seq, dict_cls_mask)
        dict_features_seq = dict_features_seq.view(
            -1, self.max_seq_length, self.max_dict_num + 1, self.config.hidden_size
        )

        # first dict attention
        joint_output = sequence_output + dict_features_seq[:, :, 0, :].squeeze()
        joint_output = self.output_atts(joint_output, joint_output, joint_output, attention_mask)
        logits_slot = self.slot_classifier(self.dropout(joint_output))
        # print(logits_slot.shape)
        logits_dict = logits_dict.view(-1, 2)
        total_loss = 0.0 
        if labels is not None:
            if self.use_subword:            
                active_loss = attention_mask.view(-1) == 1
            else:
                active_loss = loss_mask.view(-1) == 1
            active_logits = logits_slot.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            slot_loss = self.slot_loss_fct(active_logits, active_labels)

            active_dict_loss = dict_mask.view(-1) == 1
            active_dict_logits = logits_dict[active_dict_loss]
            active_dict_supervise = dict_labels.view(-1)[active_dict_loss]
            dict_loss = self.dict_loss_fct(active_dict_logits, active_dict_supervise)
            total_loss =  self.LAMBDA*slot_loss + dict_loss
            return total_loss, logits_slot, logits_dict
        else:
            return logits_slot, logits_dict