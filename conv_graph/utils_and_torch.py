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

# coding=utf-8

import numpy as np
import ast
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, classification_report
from torch.nn import functional as F
from self_play.conv_graph import SelfPlayConvGraph


def get_edges_overlap(bigger_graph: SelfPlayConvGraph, smaller_graph: SelfPlayConvGraph):
    final_state = str([1] * (len(bigger_graph.belief_state_to_idx) + len(bigger_graph.dialog_act_to_idx)))
    edges_small = [e for e in smaller_graph.graph.edges() if final_state not in e]
    edges_big = [e for e in bigger_graph.graph.edges() if final_state not in e]
    print("Shared edges: %d" % len(set(edges_small).intersection(set(edges_big))))
    print("Edge counts for each: %d and %d" % (len(edges_big), len(edges_small)))


def get_convgraph_oracle(bigger_graph: SelfPlayConvGraph, smaller_graph: SelfPlayConvGraph) -> np.array:
    index = -1
    x_list, y_list = smaller_graph.generate_standard_data(unique=False)
    final_state = str([1] * (len(smaller_graph.belief_state_to_idx) + len(smaller_graph.dialog_act_to_idx)))
    padding = str([0] * (len(smaller_graph.belief_state_to_idx) + len(smaller_graph.dialog_act_to_idx)))
    indices_to_delete = set()
    for X, Y in zip(x_list, y_list):
        index += 1
        if index % 500 == 0:
            print("Processed %d of %d" % (index, len(x_list)))
        available_nodes = bigger_graph.graph.nodes()
        for x in X:
            x_str = str([int(x_float) for x_float in x])
            if x_str in [final_state, padding]:
                continue
            if x_str not in available_nodes:
                indices_to_delete.add(index)
                break
            available_nodes = bigger_graph.graph[x_str]
        if index not in indices_to_delete:
            y_str = str([int(y_float) for y_float in Y])
            available_dialog_acts = [str(ast.literal_eval(node)[-len(smaller_graph.dialog_act_to_idx):]) for node in available_nodes]
            if y_str not in available_dialog_acts:
                indices_to_delete.add(index)
    x_list = np.delete(x_list, list(indices_to_delete), axis=0)
    y_list = np.delete(y_list, list(indices_to_delete), axis=0)
    print("Overlap between them using ConvGraph: %d out of %d" % (len(x_list), index + 1))
    print("-----------------------------------------------")
    return x_list, y_list


def get_data_overlap(x_one_batch, y_one_batch, x_two_batch, y_two_batch, bigger_graph, smaller_graph):
    count = 0
    dict_one, dict_two = dict(), dict()
    for x_one, y_one in zip(x_one_batch, y_one_batch):
        key = str(x_one.tostring() + y_one.tostring())
        dict_one[key] = dict_one.get(key, 0) + 1
    for x_two, y_two in zip(x_two_batch, y_two_batch):
        key = str(x_two.tostring() + y_two.tostring())
        dict_two[key] = dict_two.get(key, 0) + 1
    for key in dict_one:
        count += dict_two.get(key, 0)
    print("Unique examples: %d, %d" % (len(dict_one.keys()), len(dict_two.keys())))
    print("Overlap with train: %d" % count)

    final_state = str([1] * (len(bigger_graph.belief_state_to_idx) + len(bigger_graph.dialog_act_to_idx)))
    edges_small = [e for e in smaller_graph.graph.edges() if final_state not in e]
    edges_big = [e for e in bigger_graph.graph.edges() if final_state not in e]
    print("Shared edges: %d" % len(set(edges_small).intersection(set(edges_big))))
    print("Edge counts: %d and %d" % (len(edges_big), len(edges_small)))

# -------------------------------------- PYTORCH CODE FROM HERE --------------------------------------


def load_checkpoint(model, optimizer, filename='checkpoint'):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(filename, checkpoint['epoch']))
    return model, optimizer


class Classifier(nn.Module):

    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size=256, batch_first=True)
        self.hidden_to_logits = nn.Linear(256, output_size)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out, _ = self.lstm(inputs)
        logits = self.hidden_to_logits(self.relu(out[:, -1, :]))
        return logits


class SoftBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    __constants__ = ['weight', 'pos_weight', 'reduction']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
        super(SoftBCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, orig_input, conv_graph):
        min_targets = []
        for inp, orig_inp in zip(input, orig_input):
            valid_targets = conv_graph.get_valid_dialog_acts(orig_inp)

            min_target = None
            min_mean_loss = None
            for valid in valid_targets:
                valid_tens = torch.FloatTensor(valid).cuda()
                temp_mean_loss = F.binary_cross_entropy_with_logits(inp, valid_tens, self.weight, pos_weight=self.pos_weight, reduction=self.reduction).mean()
                if min_target is None:
                    min_target = valid
                    min_mean_loss = temp_mean_loss.item()
                elif temp_mean_loss.item() < min_mean_loss:
                    min_target = valid
                    min_mean_loss = temp_mean_loss.item()
            min_targets.append(min_target)
        min_targets = torch.FloatTensor(min_targets).cuda()
        return F.binary_cross_entropy_with_logits(input, min_targets, self.weight, pos_weight=self.pos_weight, reduction=self.reduction)


def f1(y_pred, y_true):
    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return 2 * (precision * recall) / (precision + recall + epsilon)


def validate_model(classifier, data_generator, convgraph, train_with_soft_loss, device):
    classifier.eval()
    if train_with_soft_loss:
        loss_function = SoftBCEWithLogitsLoss()
    else:
        loss_function = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        valid_f1s = []
        valid_losses = []
        for inputs, labels in data_generator:
            inputs, labels = inputs.to(device), labels.to(device)
            output = classifier(inputs)

            for out, lab in zip(output, labels):
                valid_f1s.append(f1((out > 0.0).int(), lab).cpu())
            if train_with_soft_loss:
                loss = loss_function(output, inputs, convgraph)
            else:
                loss = loss_function(output, labels)
            valid_losses.append(loss.data.mean().cpu())
    classifier.train()
    return np.mean(valid_losses), np.mean(valid_f1s)


def evaluate_model(classifier, data_generator, eval_graph, device, report=False):
    classifier.eval()
    with torch.no_grad():
        gold_labels = []
        nearest_gold_labels = []
        pred_labels = []
        soft_f1s = []
        for inputs, labels in data_generator:
            inputs, labels = inputs.to(device), labels.to(device)
            output = classifier(inputs)

            for inp, out, lab in zip(inputs, output, labels):
                y_pred = (out > 0.0).int()
                pred_labels.append(y_pred.cpu().numpy())
                gold_labels.append(lab.cpu().numpy())
                nearest_gold, best_f1 = eval_graph.get_best_f1_score(inp.data.tolist(), y_pred.data.tolist())
                soft_f1s.append(best_f1)
                nearest_gold_labels.append(nearest_gold)
        print("Hard F-Score (exact match): %.3f" % f1_score(y_true=np.array(gold_labels, dtype='float32'), y_pred=np.array(pred_labels, dtype='float32'), average='samples'))
        print("Soft F-Score (best match): %f" % np.mean(soft_f1s))
        if report:
            print(classification_report(y_true=np.array(nearest_gold_labels, dtype='float32'),
                                        y_pred=np.array(pred_labels, dtype='float32'),
                                        target_names=eval_graph.dialog_act_to_idx.keys(), digits=3))
    classifier.train()


# from multiwoz.conv_graph import MultiWozConvGraph
# print("Creating ConvGraphs for MultiWOZ.")
# train_graph = MultiWozConvGraph(dir_name="../multiwoz/", file_names=['train.json'])
# dev_graph = MultiWozConvGraph(dir_name="../multiwoz/", file_names=['val.json'])
# test_graph = MultiWozConvGraph(dir_name="../multiwoz/", file_names=['test.json'])
#
# print("---------------------------------")
# print("Data overlap for MultiWOZ dev")
# get_data_overlap(train_graph, dev_graph)
# print("Data overlap for MultiWOZ test")
# get_data_overlap(train_graph, test_graph)
# print("---------------------------------")
#
# get_graph_overlap(bigger_graph=train_graph, smaller_graph=dev_graph)
# get_graph_overlap(bigger_graph=train_graph, smaller_graph=test_graph)

# for dataset in ["movie", "restaurant"]:
#     train_graph = SelfPlayConvGraph(dir_name="../self_play/" + dataset, file_names=['/train.json'])
#     dev_graph = SelfPlayConvGraph(dir_name="../self_play/" + dataset, file_names=["/dev.json"])
#     test_graph = SelfPlayConvGraph(dir_name="../self_play/" + dataset, file_names=["/test.json"])
#
#     print("---------------------------------")
#     print("Data overlap for %s dev" % dataset)
#     get_data_overlap(train_graph, dev_graph)
#     print("Data overlap for %s test" % dataset)
#     get_data_overlap(train_graph, test_graph)
#     print("---------------------------------")
#
#     get_graph_overlap(bigger_graph=train_graph, smaller_graph=dev_graph)
#     get_graph_overlap(bigger_graph=train_graph, smaller_graph=test_graph)
