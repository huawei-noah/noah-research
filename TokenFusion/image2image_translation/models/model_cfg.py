#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch.nn as nn
from .model_pruning import MixVisionTransformerGen, MixVisionTransformerDis
from .modules import LayerNormParallel
from functools import partial


class gen_b0(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 128, 256, 512], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2, 2], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class gen_b1(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2, 2], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class gen_b2(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 4, 4, 6, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class gen_b3(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 4, 4, 18, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class gen_b4(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 8, 8, 27, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class gen_b5(MixVisionTransformerGen):
    def __init__(self, **kwargs):
        super(gen_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 6, 6, 40, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b0(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 128, 256, 512], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2, 2], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b1(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[2, 2, 2, 2, 2], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b2(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 4, 4, 6, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b3(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 4, 4, 18, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b4(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 8, 8, 27, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


class dis_b5(MixVisionTransformerDis):
    def __init__(self, **kwargs):
        super(dis_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 256, 512, 1024], num_heads=[1, 2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4, 4],
            qkv_bias=True, depths=[3, 6, 6, 40, 3], sr_ratios=[16, 8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)
