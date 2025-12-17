# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_name: str
    model_class: str
    model_kwargs: dict
    api_key: str
    base_url: str
    extra_params: dict

    def __init__(self,
                 model_class: str,
                 model_name: str,
                 api_key: str,
                 base_url: str,
                 **kwargs):
        self.model_class = model_class
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.extra_params = kwargs
