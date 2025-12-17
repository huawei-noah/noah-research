# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Metrics:
    speedup: float
    original_time: float
    optimized_time: float
    error: Optional[str] = None
    traceback: Optional[str] = None

    def to_dict(self):
        return asdict(self)
