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
from collections import Counter
#from seqeval.metrics import f1_score as ner_f1_score
from sequence_labeling import f1_score as ner_f1_score
from sequence_labeling import sentence_score
from sklearn.metrics import f1_score

class SentenceScore:

    def __init__(self, id2label, id2intent_label):
        self.id2label = id2label
        self.id2intent_label = id2intent_label
        self.reset()

    def reset(self):
        self.nb_correct = 0  # 预测正确的个数
        self.nb_pred = 0  # 预测总个数
        self.nb_true = 0  # 实际个数

    def compute(self):
        acc = self.nb_correct / self.nb_true if self.nb_true > 0 else 0
        # precision = self.nb_correct / self.nb_pred if self.nb_pred > 0 else 0
        # f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
        print(self.nb_correct, self.nb_true)
        return acc

    def update(self, pred_label_ids, true_label_ids,  pred_intent_label_ids, true_intent_label_ids, mask):
        num_correct, num_batch = sentence_score(true_label_ids, pred_label_ids,
                                                true_intent_label_ids, pred_intent_label_ids,
                                                self.id2label, self.id2intent_label, mask)
        self.nb_correct += num_correct
        self.nb_true += num_batch
        return

class IntentScore:

    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.nb_correct = 0  # 预测正确的个数
        self.nb_pred = 0  # 预测总个数
        self.nb_true = 0  # 实际个数

    def compute(self):
        accuracy = self.nb_correct / self.nb_true if self.nb_true > 0 else 0
        # accuracy = self.nb_correct / self.nb_pred if self.nb_pred > 0 else 0
        # f1 = (2 * recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
        return accuracy

    def update(self, pred_label_ids, true_label_ids):
        true_labels = [item for sublist in true_label_ids for item in sublist]  # list: batch*1->batch
        pred_labels = [item for sublist in pred_label_ids for item in sublist]
        # print(torch.max(pred_labels), torch.max(true_labels),true_labels, pred_labels, self.id2label, len(self.id2label))
        pred_labels = ['B-'+ self.id2label[i] for i in pred_labels]
        true_labels = ['B-'+ self.id2label[i] for i in true_labels]
        cur_f1, nb_correct, nb_pred, nb_true = ner_f1_score(true_labels, pred_labels)
        self.nb_correct += nb_correct
        # self.nb_pred += nb_pred
        self.nb_true += nb_true
        # return cur_f1

class SeqEntityScore:
    
    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.nb_correct = 0 # 预测正确的个数
        self.nb_pred = 0 # 预测总个数
        self.nb_true = 0 # 实际个数

    def compute(self):
        recall = self.nb_correct / self.nb_true if self.nb_true > 0 else 0
        precision = self.nb_correct / self.nb_pred if self.nb_pred > 0 else 0
        f1 = (2*recall*precision) / (recall+precision) if (recall+precision) > 0 else 0
        return precision, recall, f1

    def update(self, pred_label_ids, true_label_ids, mask):
        pred_labels = flat_by_mask(pred_label_ids, mask)
        true_labels = flat_by_mask(true_label_ids, mask)
        pred_labels = [self.id2label[i] for i in pred_labels]
        true_labels = [self.id2label[i] for i in true_labels]
        cur_f1, nb_correct, nb_pred, nb_true = ner_f1_score(true_labels, pred_labels)
        self.nb_correct += nb_correct
        self.nb_pred += nb_pred
        self.nb_true += nb_true
        return cur_f1

class DictionaryScore:
    def __init__(self):
        self.reset()

    def reset(self):
        self.nb_correct = 0 # 预测正确的个数
        self.nb_pred = 0 # 预测总个数
        self.nb_true = 0 # 实际个数

    def compute(self):
        recall = self.nb_correct / self.nb_true if self.nb_true > 0 else 0
        precision = self.nb_correct / self.nb_pred if self.nb_pred > 0 else 0
        f1 = (2*recall*precision) / (recall+precision) if (recall+precision) > 0 else 0
        return precision, recall, f1

    def update(self, pred_label_ids, true_label_ids, mask):
        # for nested list
        if any(isinstance(s, list) for s in pred_label_ids):
            pred_label_ids = [item for sublist in pred_label_ids for item in sublist]
        if any(isinstance(s, list) for s in true_label_ids):
            true_label_ids = [item for sublist in true_label_ids for item in sublist]
        
        pred_label_ids = flat_by_mask(pred_label_ids, mask)
        true_label_ids = flat_by_mask(true_label_ids, mask)

        share = (pred_label_ids * true_label_ids)
        self.nb_correct += share.sum().item()
        self.nb_pred += pred_label_ids.sum().item()
        self.nb_true += true_label_ids.sum().item()
        

def flat_by_mask(t, mask):
    """
        给定一个Tensor，通过mask剪裁
    """
    mask = mask.to(torch.bool)
    return torch.masked_select(t, mask)