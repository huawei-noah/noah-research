# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Third Party Open Source Notice
# The starting point for this repo was cloned from [JointBERT](https://github.com/monologg/JointBERT).
# Some unmodified code that does not constitute the key methodology introduced in our paper remains in the codebase.

import os
import pickle
import torch.nn as nn
from transformers import XLMRobertaModel
from transformers.modeling_bert import BertPreTrainedModel


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

        outputs = ((intent_logits, slot_logits),) + outputs[2:]
        return (total_loss,) + outputs


def get_intent_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "intents.pkl"), 'rb'))


def get_slot_labels(args):
    return pickle.load(open(os.path.join(args.data_dir, args.task, "slots.pkl"), 'rb'))


class XPairModel(BertPreTrainedModel):
    def __init__(self, config, args):
        super(XPairModel, self).__init__(config)
        self.args = args
        self.num_labels = args.num_pair_labels
        self.roberta = XLMRobertaModel(config=config)
        self.classifier = SequenceClassifier(config.hidden_size, self.num_labels)

    def forward(self, input_ids, labels):
        outputs = self.roberta(input_ids)
        pooled_output = outputs[1]  # <s>

        logits = self.classifier(pooled_output)
        loss = 0

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        outputs = (logits, ) + outputs[2:]
        return (loss,) + outputs
