# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch
import os


class BaseObserver:

    def __init__(self, module_type, bit_type, calibration_mode):
        self.module_type = module_type
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.max_val = None
        self.min_val = None
        self.eps = torch.finfo(torch.float32).eps
        # self.group_size = -1
        # self.group_size = 128
        self.group_size = int(os.environ['W_GROUP_SIZE']) if 'W_GROUP_SIZE' in os.environ else -1
        # print(f"[BaseObserver] self.group_size: {self.group_size}, type(self.group_size): {type(self.group_size)}")

    def reshape_tensor(self, v):
        if not isinstance(v, torch.Tensor):
            v = torch.tensor(v)
        v = v.detach()
        if self.module_type in ['conv_weight']:
            v = v.reshape(v.shape[0], -1)       #### Channel-wise
        elif self.module_type in ['linear_weight']:
            v = v.reshape(v.shape[0], -1)  # (cout, cin)   #### Channel-wise
            if self.group_size > 0:
                # w   [co, ci] == [co, n_group * group_size] => [co*n_group, group_size]
                v = v.reshape(-1, self.group_size)
        elif self.module_type == 'activation':
            if len(v.shape) == 4:
                v = v.permute(0, 2, 3, 1)
            v = v.reshape(-1, v.shape[-1])
            v = v.transpose(0, 1)
        else:
            raise NotImplementedError
        return v

    def update(self, v):
        # update self.max_val and self.min_val
        raise NotImplementedError

    def get_quantization_params(self, *args, **kwargs):
        raise NotImplementedError
