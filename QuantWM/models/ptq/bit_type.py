# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
import torch.nn as nn


class BitType:

    def __init__(self, bits, signed, name=None):
        self.bits = bits
        self.signed = signed
        if name is not None:
            self.name = name
        else:
            self.update_name()

    @property
    def upper_bound(self):
        if not self.signed:
            return 2**self.bits - 1
        return 2**(self.bits - 1) - 1

    @property
    def lower_bound(self):
        if not self.signed:
            return 0
        return -(2**(self.bits - 1))

    @property
    def range(self):
        return 2**self.bits

    def update_name(self):
        self.name = ''
        if not self.signed:
            self.name += 'uint'
        else:
            self.name += 'int'
        self.name += '{}'.format(self.bits)


BIT_TYPE_LIST = [
    BitType(3, True, 'int3'),
    BitType(3, False, 'uint3'),
    BitType(4, True, 'int4'),
    BitType(4, False, 'uint4'),
    BitType(6, True, 'int6'),
    BitType(6, False, 'uint6'),
    BitType(8, True, 'int8'),
    BitType(8, False, 'uint8'),
    BitType(16, True, 'int16'),
    BitType(16, False, 'uint16')
]
BIT_TYPE_DICT = {bit_type.name: bit_type for bit_type in BIT_TYPE_LIST}

