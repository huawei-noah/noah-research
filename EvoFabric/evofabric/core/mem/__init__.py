# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from ._base_mem import (
    MemBase
)

from ._cognitive_mem import (
    CognitiveMem
)

from ._retrieval_mem import (
    RetrievalMem
)

from ._chat_mem import (
    ChatMem
)

from ._task_mem import (
    TaskMem
)

from ._default_mem_prompt_en import (
    FACT_RETRIEVAL_PROMPT_EN,
    DEFAULT_UPDATE_MEMORY_PROMPT_EN,
    FEAT_DEFINE_PROMPT_EN,
    TASK_SUMMARY_PROMPT_EN
)

from ._default_mem_prompt_zh import (
    FACT_RETRIEVAL_PROMPT_ZH,
    DEFAULT_UPDATE_MEMORY_PROMPT_ZH,
    FEAT_DEFINE_PROMPT_ZH,
    TASK_SUMMARY_PROMPT_ZH
)
