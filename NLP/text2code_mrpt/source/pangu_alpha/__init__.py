# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified version of GPT-2 into the PanGu-Alpha architecture.
# - The network include an additional head on top of the backbone model.
# ============================================================================
from typing import TYPE_CHECKING

from transformers.file_utils import (
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_tokenizers_available,
    is_torch_available
)


_import_structure = {
    "configuration_pangualpha": ["PanguAlphaConfig"],
    "tokenization_pangualpha": ["PanguAlphaTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_pangualpha"] = [
        "PanguAlphaModel",
    ]

if TYPE_CHECKING:
    from .configuration_pangualpha import PanguAlphaConfig

    if is_torch_available():
        from .modeling_pangualpha import (
            PanguAlphaModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
