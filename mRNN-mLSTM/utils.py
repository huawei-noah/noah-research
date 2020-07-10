#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for
# more details.

import numpy as np
import torch

def create_dataset(data, look_back=1):
    """creat train dataset for time series prediction"""
    data_yp, data_yc = [], []
    for i in range(len(data) - look_back):
        data_yp.append(data[i, 0])
        data_yc.append(data[i + look_back, 0])
    return np.array(data_yp), np.array(data_yc)

def padding(data, length, input_size):
    term = [0] * input_size
    data_pad = []
    for text in data:
        if len(text) >= length:
            text = np.array(text[0:length])
        else:
            pad_list = np.array([term] * (length - len(text)))
            text = np.vstack([np.array(text), pad_list])
        data_pad.append(text)
    data_pad = np.array(data_pad)
    return data_pad

def accuracy(output, labels):
    """compute the accuracy for review classification"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# transform data to tensor in torch
def to_torch(state):
    state = torch.from_numpy(state).float()
    return state