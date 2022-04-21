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

import os
import pickle
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_bert import BertPreTrainedModel


class CrossAligner(nn.Module):
    def __init__(self, args, num_classes):
        super(CrossAligner, self).__init__()
        self.args = args
        self.linear = nn.Linear(self.args.max_seq_len * num_classes, num_classes)

    def forward(self, x):
        x = torch.reshape(x, (x.shape[0], -1))
        return self.linear(x)


class SequenceClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class TokenClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TokenClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


class XNLUModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super(XNLUModel, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(get_intent_labels(args))
        self.num_slot_labels = len(get_slot_labels(args))
        self.roberta = XLMRobertaModel(config=config)
        self.intent_classifier = SequenceClassifier(config.hidden_size, self.num_intent_labels)
        self.slot_classifier = TokenClassifier(config.hidden_size, self.num_slot_labels)
        self.cross_aligner = CrossAligner(args, self.num_slot_labels)

    def forward(self, input_ids, intent_labels=None, slot_labels=None):
        outputs = self.roberta(input_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # <s>

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        total_loss = 0

        if intent_labels is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_labels.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_labels.view(-1))
            total_loss += intent_loss

        if slot_labels is not None:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels.view(-1))
            total_loss += slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs
        return (total_loss,) + outputs


def get_intent_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "intents.pkl"), 'rb'))


def get_slot_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "slots.pkl"), 'rb'))
