# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from torch.distributions import Categorical
from lib.score_functions import register_score


class LogitsScoreFunction:
    def __init__(self, temperature=1.):
        assert temperature > 0
        self.temperature = temperature

    def __call__(self, logits):
        raise NotImplementedError


@register_score('max')
class MSPScore(LogitsScoreFunction):
    def __init__(self, temperature=1.):
        super(MSPScore, self).__init__(temperature)

    def __call__(self, logits):
        return 1 - logits.div(self.temperature).softmax(dim=1).max(1)[0]


@register_score('entropy')
class EntropyScore(LogitsScoreFunction):
    def __init__(self, temperature=1.):
        super(EntropyScore, self).__init__(temperature)

    def __call__(self, logits):
        return Categorical(logits=logits.div(self.temperature)).entropy()
