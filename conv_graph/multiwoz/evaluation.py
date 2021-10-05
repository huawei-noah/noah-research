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

import os
import random
import numpy as np
from torch import optim
import torch.nn as nn
import torch.utils.data
from torch.utils.data import TensorDataset, DataLoader
from multiwoz.conv_graph import MultiWozConvGraph
from utils_and_torch import get_convgraph_oracle, evaluate_model, Classifier, get_data_overlap, get_edges_overlap
from utils_and_torch import SoftBCEWithLogitsLoss, validate_model, f1, load_checkpoint

seed = 123456789
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['TF_CUDNN_DETERMINISM'] = str(1)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


history = 4
# default history is 4
train_with_soft_loss = False
# soft loss training is slow, be patient
max_epochs, max_val_f1, patience = 50, 0, 3
train_graph = MultiWozConvGraph(dir_name="./", file_names=['train.json'], seq_length=history)
dev_graph = MultiWozConvGraph(dir_name="./", file_names=['val.json'], seq_length=history)
test_graph = MultiWozConvGraph(dir_name="./", file_names=['test.json'], seq_length=history)
eval_graph = MultiWozConvGraph(dir_name="./", file_names=['train.json', 'val.json', 'test.json'], seq_length=history)
# baseline training
x_train, y_train = train_graph.generate_standard_data(unique=False)
# -----------------------------------------------------------------
# downsampling training
# x_train, y_train = train_graph.generate_standard_data(unique=True)
# -----------------------------------------------------------------
# oracle training
# x_t, y_t = get_convgraph_oracle(train_graph, dev_graph)
# x_train = np.concatenate((x_train, x_t))
# y_train = np.concatenate((y_train, y_t))
# x_t, y_t = get_convgraph_oracle(train_graph, test_graph)
# x_train = np.concatenate((x_train, x_t))
# y_train = np.concatenate((y_train, y_t))
# -----------------------------------------------------------------
# data duplication training
# x_train, y_train = train_graph.generate_standard_data(unique=False)
# x_train = np.concatenate((x_train, x_train))
# y_train = np.concatenate((y_train, y_train))
# -----------------------------------------------------------------
# data augmentation training only
# x_train, y_train = train_graph.generate_augmented_data()
# -----------------------------------------------------------------
# data augmentation + baseline training
# x_t, y_t = train_graph.generate_augmented_data()
# x_train, y_train = train_graph.generate_standard_data(unique=False)
# x_train = np.concatenate((x_train, x_t))
# y_train = np.concatenate((y_train, y_t))
# -----------------------------------------------------------------
print("Total Train Sequences: %d" % len(x_train))
x_dev, y_dev = dev_graph.generate_standard_data(unique=False)
print("Total Dev Sequences: %d" % len(x_dev))
# -----------------------------------------------------------------
x_test, y_test = test_graph.generate_standard_data(unique=True)
print("Total Deduplicated Test Sequences: %d" % len(x_test))
# -----------------------------------------------------------------
full_x_test, full_y_test = test_graph.generate_standard_data(unique=False)
print("Total Full Test Sequences: %d" % len(full_x_test))
# -----------------------------------------------------------------
state_length = len(train_graph.belief_state_to_idx) + len(train_graph.dialog_act_to_idx)
target_length = len(train_graph.dialog_act_to_idx)
print("Input Size: %d, Output Size: %d" % (state_length, target_length))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classifier = Classifier(state_length, target_length)
print("No of model parameters: %d" % sum([param.nelement() for param in classifier.parameters()]))
print("-----------------------------------------------")

if torch.cuda.is_available():
    classifier.cuda()

params = {'batch_size': 32, 'shuffle': True}
optimizer = optim.RMSprop(classifier.parameters())
training_set = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
training_generator = DataLoader(training_set, **params)
validation_set = TensorDataset(torch.tensor(x_dev, dtype=torch.float32), torch.tensor(y_dev, dtype=torch.float32))
validation_generator = DataLoader(validation_set, **params)
no_dupl_test_set = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
no_dupl_test_generator = DataLoader(no_dupl_test_set, **params)
full_test_set = TensorDataset(torch.tensor(full_x_test, dtype=torch.float32), torch.tensor(full_y_test, dtype=torch.float32))
full_test_generator = DataLoader(full_test_set, **params)

if train_with_soft_loss:
    loss_function = SoftBCEWithLogitsLoss()
else:
    loss_function = nn.BCEWithLogitsLoss()

for epoch in range(max_epochs):
    f1s = []
    losses = []
    for inputs, labels in training_generator:
        classifier.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        output = classifier(inputs)
        if train_with_soft_loss:
            loss = loss_function(output, inputs, train_graph)
        else:
            loss = loss_function(output, labels)

        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean().cpu())

        for out, lab in zip(output, labels):
            f1s.append(f1((out > 0.0).float(), lab).cpu())

    valid_loss, valid_f1 = validate_model(classifier, validation_generator, dev_graph, train_with_soft_loss, device)
    # noinspection PyStringFormat
    print('[%d/%d] Train Loss: %.3f, Train F1: %.3f, Val Loss: %.3f, Val F1: %.3f,' % (epoch + 1, 50, np.mean(losses), np.mean(f1s), valid_loss, valid_f1))

    # Early stopping
    if valid_f1 > max_val_f1:
        state = {'epoch': epoch + 1, 'state_dict': classifier.state_dict(), 'optimizer': optimizer.state_dict(), }
        torch.save(state, 'checkpoint')

        max_val_f1 = valid_f1
        max_epochs = 0
    else:
        max_epochs += 1
        if max_epochs >= patience:
            classifier, optimizer = load_checkpoint(classifier, optimizer)
            print("Stopped early and went back to Validation f1: %.3f" % max_val_f1)
            break

print("---------------------- DEVELOPMENT SET REPORT --------------------------")
evaluate_model(classifier, validation_generator, eval_graph, device)

print("--------------------- DEDUPLICATED TEST SET REPORT -------------------------")
evaluate_model(classifier, no_dupl_test_generator, eval_graph, device)

print("--------------------- FULL TEST SET REPORT -------------------------")
evaluate_model(classifier, full_test_generator, eval_graph, device, report=False)

if False:
    print("===================SOME GRAPH STATS===================")
    print("Overlap between train and dev (dev size: %d)" % len(x_dev))
    get_data_overlap(x_train, y_train, x_dev, y_dev, train_graph, dev_graph)
    get_edges_overlap(train_graph, dev_graph)
    print("===================SOME GRAPH STATS===================")
    print("Overlap between train and test (test size: %d)" % len(x_test))
    get_data_overlap(x_train, y_train, x_test, y_test, train_graph, test_graph)
    get_edges_overlap(train_graph, test_graph)
    print("===================SOME GRAPH STATS===================")
