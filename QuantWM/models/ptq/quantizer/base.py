# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn as nn
import os


class BaseQuantizer(nn.Module):

    def __init__(self, bit_type, observer, module_type):
        super(BaseQuantizer, self).__init__()
        self.bit_type = bit_type
        self.observer = observer
        self.module_type = module_type
        # self.group_size = -1
        # self.group_size = 128
        self.group_size = int(os.environ['W_GROUP_SIZE']) if 'W_GROUP_SIZE' in os.environ else -1
        # print(f"[BaseQuantizer] self.group_size: {self.group_size}, type(self.group_size): {type(self.group_size)}")

    def get_reshape_range(self, inputs):
        range_shape = None
        if self.module_type == 'conv_weight':
            range_shape = (-1, 1, 1, 1)
        elif self.module_type == 'linear_weight':
            range_shape = (-1, 1)
        elif self.module_type == 'activation':
            if len(inputs.shape) == 2:
                range_shape = (1, -1)
            elif len(inputs.shape) == 3:
                range_shape = (1, 1, -1)
            elif len(inputs.shape) == 4:
                range_shape = (1, -1, 1, 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return range_shape

    def update_quantization_params(self, *args, **kwargs):
        pass

    def quant(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def dequantize(self, inputs, scale=None, zero_point=None):
        raise NotImplementedError

    def forward(self, inputs):
        if self.bit_type.bits >= 16:
            return inputs
        if self.module_type == 'linear_weight' and self.group_size > 0:
            dim1, dim2 = inputs.shape
            inputs = inputs.reshape(-1, self.group_size)
        outputs = self.quant(inputs)
        outputs = self.dequantize(outputs)

        if self.module_type == 'linear_weight' and self.group_size > 0:
            outputs = outputs.reshape(dim1, dim2)
        return outputs
