# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


from ._config import (
    config,
    LLMConfig,
    WebParserConfig,
    WebSearchConfig,
    SolutionConfig,
    ExpConfig,
    StructureConfig,
    PromptConfig,
    GraphConfig
)

from .prompts import (
    CRITIC_TWICE_USER_PROMPT,
    CRITIC_USER_PROMPT,
    GUIDED_SUMMARY_PROMPT,
    SELECTOR_ITERATION_USER_PROMPT,
    SELECTOR_USER_PROMPT,
    SOLVER_USER_PROMPT,
    WEB_PARSER_PROMPT_HTML,
    WEB_PARSER_PROMPT_PDF,
    repeat_prompt
)

from ._graph import (
    get_agent,
    list_ele_overwrite,
    build_rethinker_graph,
    run_rethinker_graph
)

from .adapter import (
    generate_stop_condition,
    get_client,
    FastSlowThinkOpenAIChatClient,
)

from .agent import (
    CodingAgentResult,
    CodingAgent
)

from .evaluation import (
    BaseBenchmarkEvaluator,
    HLEEvaluator,
    XBenchEvaluator,
    GaiaEvaluator,
    JUDGE_PROMPT_HLE,
    JUDGE_PROMPT_XBENCH,
    JUDGE_PROMPT_GAIA
)

from .nodes import (
    AsyncNodeWithCacheAndConcurrencyLimit,
    SolutionWithReThinkNode,
    CriticWithRethinkNode,
    ConfidenceGuideSelectNode,
    GuidedSummaryNode,
    DispatchNode,
    get_dispatch_filter,
)

from .tools import (
    execute_python_code,
    web_parse,
    web_search,
    download_and_read_pdf
)