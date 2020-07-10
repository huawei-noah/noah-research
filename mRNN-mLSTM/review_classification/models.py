#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it
# under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for
# more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers


class RNN(nn.Module):
    """RNN model for review classification"""
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(dropout)
        self.rnn1 = nn.RNN(self.input_size, self.hidden_size,
                           dropout=dropout)
        self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size,
                            dropout=dropout)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, length, hidden_state=None):
        samples = self.dropout(input_y)
        rnn_out1, last_rnn_hidden = self.rnn1(samples)
        rnn_out1 = F.tanh(rnn_out1)
        rnn_out, last_rnn_hidden = self.rnn2(rnn_out1)
        rnn_out = F.tanh(rnn_out)
        rnn_out_sel = torch.cat([rnn_out[length[i], i].unsqueeze(0) for i in
                                 range(length.shape[0])], dim=0)
        output = self.hidden2output(rnn_out_sel)
        return F.log_softmax(output, dim=1)


class LSTM(nn.Module):
    """LSTM model for review classification"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(LSTM, self).__init__()
        self.in_size = in_size
        self.h_size = hidden_size
        self.out_size = out_size
        self.dropout = nn.Dropout(dropout)
        self.lstm_cell_1 = nn.LSTMCell(in_size, hidden_size)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, out_size)

    def forward(self, inputs, length, hx1=None, cx1=None, hx2=None, cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.h_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        for times in range(time_steps):
            if hx1 is None:
                hx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx1 is None:
                cx1 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            inputs_t = self.dropout(inputs[times, :])
            hx1, cx1 = self.lstm_cell_1(inputs_t, (hx1, cx1))
            hx1 = F.tanh(hx1)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.h_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell_2(hx1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.output(outputs_sel)
        return F.log_softmax(logit, dim=1)


class MRNNFixD(nn.Module):
    """mRNN with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, output_size, k,
                 dropout, bias=True):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.bd1 = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.dropout = nn.Dropout(dropout)
        self.mrnn_cell_1 = layers.MRNNFixDCell(input_size, hidden_size,
                                               hidden_size, k)
        self.lstm_cell_2 = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_values):
        weights = torch.ones(self.k, 1, d_values.size(1),
                             dtype=d_values.dtype, device=d_values.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[0, hidden].view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, length, hid=None, hx2=None, cx2=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d_values = 0.5 * F.sigmoid(self.bd1)
        weight_d = self.get_wd(self.d_values)

        for times in range(time_steps):
            temp = self.dropout(inputs[times, :])
            outputs1, hid = self.mrnn_cell_1(temp, weight_d, hid)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell_2(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2

        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)


class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for review classification"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                             requires_grad=True)
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMFixDCell(self.input_size,
                                               self.hidden_size,
                                               self.hidden_size,
                                               self.k)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.hidden2output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_values):
        weights = torch.ones(self.k, 1, d_values.size(1),
                             dtype=d_values.dtype, device=d_values.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[0, hidden].view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, length, hidden=None, h_c=None, hx2=None,
                cx2=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.hidden_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        self.d_values = 0.5 * F.sigmoid(self.b_d)
        weight_d = self.get_wd(self.d_values)
        for times in range(time_steps):
            outputs1, hidden, h_c = self.mlstm_cell(inputs[times, :], hidden,
                                                    h_c, weight_d)
            if hx2 is None:
                hx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            if cx2 is None:
                cx2 = torch.zeros(batch_size, self.hidden_size,
                                  dtype=inputs.dtype, device=inputs.device)
            hx2, cx2 = self.lstm_cell(outputs1, (hx2, cx2))
            hx2 = F.tanh(hx2)
            outputs[times, :] = hx2
        outputs_sel = torch.cat([outputs[length[i], i].unsqueeze(0) for i
                                 in range(length.shape[0])], dim=0)
        logit = self.hidden2output(outputs_sel)
        return F.log_softmax(logit, dim=1)