#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it
# under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import layers


class RNN(nn.Module):
    """RNN model for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.rnn = nn.RNN(self.input_size, self.hidden_size)
        self.hidden2output = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_y, hidden_state=None):
        samples = input_y
        rnn_out, last_rnn_hidden = self.rnn(samples, hidden_state)
        output = self.hidden2output(rnn_out.view(-1, self.hidden_size))
        return output.view(samples.shape[0], samples.shape[1],
                           self.output_size), \
               last_rnn_hidden


class LSTM(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.in_size = input_size
        self.h_size = hidden_size
        self.out_size = output_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.Tensor(time_steps, batch_size, self.out_size)
        if hidden_state is None:
            h_0 = torch.zeros(batch_size, self.h_size)
            c_0 = torch.zeros(batch_size, self.h_size)
            hidden_state = (h_0, c_0)
        else:
            h_0 = hidden_state[0]
            c_0 = hidden_state[1]
        for times in range(time_steps):
            h_0, c_0 = self.lstm_cell(inputs[times, :], (h_0, c_0))
            outputs[times, :] = self.output(h_0)
        return outputs, (h_0, c_0)


class MRNNFixD(nn.Module):
    """mRNN with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNFixD, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.b_d = Parameter(torch.Tensor(torch.zeros(1, input_size)),
                             requires_grad=True)
        self.mrnn_cell = layers.MRNNFixDCell(input_size, hidden_size,
                                             output_size, k)
    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_ws(d_value[0, hidden].
                                                         view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        self.d_matrix = 0.5 * F.sigmoid(self.b_d)
        weights_d = self.get_wd(self.d_matrix)
        for times in range(time_steps):
            outputs, hidden_state = self.mrnn_cell(inputs[times, :], weights_d,
                                                   hidden_state)
        return outputs, hidden_state


class MRNN(nn.Module):
    """mRNN with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNN, self).__init__()
        self.k = k
        self.input_size = input_size
        self.output_size = output_size
        self.mrnn_cell = layers.MRNNCell(input_size, hidden_size,
                                         output_size, k)

    def forward(self, inputs, hidden_state=None):
        time_steps = inputs.size(0)
        batch_size = inputs.size(1)
        outputs = torch.Tensor(time_steps, batch_size, self.output_size)
        for times in range(time_steps):
            outputs[times, :], hidden_state = self.mrnn_cell(inputs[times, :],
                                                             hidden_state)
        return outputs, hidden_state


class MLSTMFixD(nn.Module):
    """mLSTM with fixed d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMFixD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.d_values = Parameter(torch.Tensor(torch.zeros(1, hidden_size)),
                                  requires_grad=True)
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMFixDCell(self.input_size,
                                               self.hidden_size,
                                               self.output_size,
                                               self.k)
        self.sigmoid = nn.Sigmoid()

    def get_w(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def get_wd(self, d_value):
        weights = torch.ones(self.k, 1, d_value.size(1), dtype=d_value.dtype,
                             device=d_value.device)
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = self.get_w(d_value[0, hidden].
                                                        view([1]))
        return weights.squeeze(1)

    def forward(self, inputs, hidden_states=None):
        if hidden_states is None:
            hidden = None
            h_c = None
        else:
            hidden = hidden_states[0]
            h_c = hidden_states[1]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        self.d_values_sigmoid = 0.5 * F.sigmoid(self.d_values)
        weights_d = self.get_wd(self.d_values_sigmoid)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c = self.mlstm_cell(inputs[times, :],
                                                             hidden,
                                                             h_c,
                                                             weights_d)
        return outputs, (hidden, h_c)


class MLSTM(nn.Module):
    """mLSTM with dynamic d for time series prediction"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size
        self.mlstm_cell = layers.MLSTMCell(self.input_size, self.hidden_size,
                                           self.k, self.output_size)

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            hidden = None
            h_c = None
            d_values = None
        else:
            hidden = hidden_state[0]
            h_c = hidden_state[1]
            d_values = hidden_state[2]
        time_steps = inputs.shape[0]
        batch_size = inputs.shape[1]
        outputs = torch.zeros(time_steps, batch_size, self.output_size,
                              dtype=inputs.dtype, device=inputs.device)
        for times in range(time_steps):
            outputs[times, :], hidden, h_c, d_values = \
                self.mlstm_cell(inputs[times, :], hidden, h_c, d_values)
        return outputs, (hidden, h_c, d_values)

