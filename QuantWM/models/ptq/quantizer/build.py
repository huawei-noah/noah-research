# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
from .log2 import Log2Quantizer
from .uniform import UniformQuantizer

str2quantizer = {'uniform': UniformQuantizer, 'log2': Log2Quantizer}


def build_quantizer(quantizer_str, bit_type, observer, module_type):
    if bit_type is None:
        return None
    quantizer = str2quantizer[quantizer_str]
    return quantizer(bit_type, observer, module_type)
