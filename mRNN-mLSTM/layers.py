#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more
# details.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MRNNFixDCell(nn.RNNCellBase):
    """memory augmented RNN with fixed D"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNFixDCell, self).__init__(input_size, hidden_size, bias,
                                           num_chunks=1)
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)
        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(input_size, hidden_size, bias=False)
        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inputs, weight_d, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            m_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            x_s = torch.zeros(self.k - 1, inputs.size(0), self.input_size,
                              dtype=inputs.dtype, device=inputs.device)
            hidden_state = (h_0, m_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        x_s = hidden_state[2]

        x_combine = torch.cat([x_s, inputs.view(-1, inputs.size(0),
                                         inputs.size(1))], 0)
        x_filter = torch.einsum('ijk,ik->ijk', [x_combine, weight_d]).\
            sum(dim=0)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        hid = F.tanh(self.h_h(h_0) + self.x_h(inputs))
        z_out = F.tanh(self.h_z(hid) + self.m_z(mem))
        xs_out = x_combine[1:, :]
        return z_out, (hid, mem, xs_out)


class MRNNCell(nn.RNNCellBase):
    """memory augmented RNN with dynamic D"""
    def __init__(self, input_size, hidden_size, output_size, k, bias=True):
        super(MRNNCell, self).__init__(input_size, hidden_size, bias,
                                       num_chunks=1)  # weight_ih, weight_hh
        self.k = k
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.d_d = nn.Linear(input_size, input_size, bias=True)
        self.h_d = nn.Linear(hidden_size, input_size, bias=False)
        self.m_d = nn.Linear(hidden_size, input_size, bias=False)
        self.x_d = nn.Linear(input_size, input_size, bias=False)

        self.m_m = nn.Linear(hidden_size, hidden_size, bias=True)
        self.f_m = nn.Linear(input_size, hidden_size, bias=False)

        self.h_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.x_h = nn.Linear(input_size, hidden_size, bias=False)

        self.h_z = nn.Linear(hidden_size, output_size, bias=True)
        self.m_z = nn.Linear(hidden_size, output_size, bias=False)

    def get_ws(self, d_values):
        k = self.k
        weights = [1.] * (k + 1)
        for i in range(k):
            weights[k - i - 1] = weights[k - i] * (i - d_values) / (i + 1)
        return torch.cat(weights[0:k])

    def filter_d(self, h_c, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1))
        batch_size = weights.shape[1]
        hidden_size = weights.shape[2]
        for sample in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, sample, hidden] = \
                    self.get_ws(d_values[sample, hidden].view([1]))
        outputs = h_c.mul(weights).sum(dim=0)
        return outputs

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            h_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            m_0 = torch.zeros(inputs.size(0), self.hidden_size,
                              dtype=inputs.dtype, device=inputs.device)
            d_0 = torch.zeros(inputs.size(0), self.input_size,
                              dtype=inputs.dtype, device=inputs.device)
            x_s = torch.zeros(self.k - 1, inputs.size(0), self.input_size,
                             dtype=inputs.dtype, device=inputs.device)
            hidden_state = (h_0, m_0, d_0, x_s)

        h_0 = hidden_state[0]
        m_0 = hidden_state[1]
        d_0 = hidden_state[2]
        x_s = hidden_state[3]

        # dynamic d
        d_values = 0.5 * F.sigmoid(self.d_d(d_0) + self.h_d(h_0) +
                                   self.m_d(m_0) + self.x_d(inputs))

        x_combine = torch.cat([x_s, inputs.view(-1, inputs.size(0),
                                         inputs.size(1))], 0)
        x_filter = self.filter_d(x_combine, d_values)
        mem = F.tanh(self.m_m(m_0) + self.f_m(x_filter))
        hid = F.tanh(self.h_h(h_0) + self.x_h(inputs))
        z_out = self.h_z(hid) + self.m_z(mem)
        xs_out = x_combine[1:, :]
        return z_out, (hid, mem, d_values, xs_out)


class MLSTMFixDCell(nn.Module):
    """memory augmented LSTM with fixed D"""
    def __init__(self, input_size, hidden_size, output_size, k):
        super(MLSTMFixDCell, self).__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size

        self.c_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, sample, hidden, cell_tensor, weights):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size,
                                 dtype=sample.dtype, device=sample.device)
        if cell_tensor is None:
            cell_tensor = torch.zeros(self.k, batch_size, self.hidden_size,
                                      dtype=sample.dtype, device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        first = torch.einsum('ijk,ik->ijk', [-cell_tensor, weights]).sum(dim=0)
        i_gate = self.i_gate(combined)
        o_gate = self.o_gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.c_gate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        h_c = torch.cat([cell_tensor, cell.view([-1, cell.size(0),
                                                 cell.size(1)])], 0)
        h_c_1 = h_c[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, h_c_1

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))


class MLSTMCell(nn.Module):
    """memory augmented LSTM with dynamic D"""
    def __init__(self, input_size, hidden_size, k, output_size):
        super(MLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.output_size = output_size

        self.c_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.i_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.f_gate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.o_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_ws(self, d_values):
        weights = [1.] * (self.k + 1)
        for i in range(0, self.k):
            weights[self.k - i - 1] = weights[self.k - i] * (i - d_values) / \
                                      (i + 1)
        return torch.cat(weights[0:self.k])

    def filter_d(self, cell_tensor, d_values):
        weights = torch.ones(self.k, d_values.size(0), d_values.size(1),
                       dtype=d_values.dtype, device=d_values.device)
        hidden_size = weights.shape[2]
        batch_size = weights.shape[1]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                weights[:, batch, hidden] = \
                    self.get_ws(d_values[batch, hidden].view([1]))
        outputs = cell_tensor.mul(weights).sum(dim=0)
        return outputs

    def forward(self, sample, hidden, cell_tensor, d_0):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size,
                                 dtype=sample.dtype, device=sample.device)
        if cell_tensor is None:
            cell_tensor = torch.zeros(self.k, batch_size, self.hidden_size,
                                      dtype=sample.dtype, device=sample.device)
        if d_0 is None:
            d_0 = torch.zeros(batch_size, self.hidden_size,
                              dtype=sample.dtype, device=sample.device)

        combined = torch.cat((sample, hidden), 1)
        combined_d = torch.cat((sample, hidden, d_0), 1)
        d_values = self.f_gate(combined_d)
        d_values = self.sigmoid(d_values) * 0.5
        first = -self.filter_d(cell_tensor, d_values)
        i_gate = self.i_gate(combined)
        o_gate = self.o_gate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.c_gate(combined)
        c_tilde = self.tanh(c_tilde)

        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        h_c = torch.cat([cell_tensor, cell.view([-1, cell.size(0),
                                                 cell.size(1)])], 0)
        h_c_1 = h_c[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, h_c_1, d_values

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))
