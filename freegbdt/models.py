# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

from torch import nn
import torch


class ModelClassifier(nn.Module):
    def __init__(self, hidden_dim, n_labels, head):
        super().__init__()

        self.head = head

        if head == "linear":
            self.out_proj = nn.Linear(hidden_dim, n_labels)
        elif head == "multilayer":
            self.dense = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(hidden_dim, n_labels)

    def forward(self, features, return_features=False):
        if self.head == "linear":
            x = self.out_proj(features)
        elif self.head == "multilayer":
            x = self.dropout(features)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)

        if return_features:
            return x, features
        else:
            return x


class ModelForSequenceClassification(nn.Module):
    def __init__(self, base, hidden_dim, n_labels=1, head="linear"):
        super().__init__()
        self.base = base
        self.hidden_dim = hidden_dim
        self.n_labels = n_labels

        self.classifier = ModelClassifier(self.hidden_dim, self.n_labels, head=head).to(
            next(self.parameters()).device
        )
        self.initial_state = self.state_dict()
        self.required_grad = {
            name: p.requires_grad for name, p in self.base.named_parameters()
        }

    def reset(self):
        self.load_state_dict(self.initial_state)
        self.classifier = ModelClassifier(self.hidden_dim, self.n_labels).to(
            next(self.parameters()).device
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_features=True,
        **kwargs
    ):
        base_out = self.base.forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

        base_out = base_out[0][:, 0, :]

        if return_features:
            logits, features = self.classifier(base_out, return_features=True)

            outputs = (logits, features)
        else:
            logits = self.classifier(base_out)
            outputs = (logits,)

        if labels is not None:
            if self.n_labels == 1:
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits.view(-1), labels.view(-1).float())
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view((-1, self.n_labels)), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs
