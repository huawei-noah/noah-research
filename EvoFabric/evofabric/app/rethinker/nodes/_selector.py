# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
import re
from typing import Optional

import math
from pydantic import PrivateAttr

from evofabric.core.factory import safe_get_attr
from evofabric.core.graph import current_ctx, stream_writer_env, StreamCtx
from evofabric.core.typing import State, StateDelta
from evofabric.logger import get_logger
from ._base import AsyncNodeWithCacheAndConcurrencyLimit
from ._utils import parse_answer
from .._config import config
from ..prompts import repeat_prompt


def _calculate_ppl(token_log_probs: list):
    """
    Compute perplexity from per-token log probabilities.

    Args:
        token_log_probs (list): A list of token-level log probability records,
            where each element contains a ``"logprob"`` field.

    Returns:
        float: The computed perplexity. Returns ``inf`` if the input is empty.
    """
    if not token_log_probs:
        return float('inf')
    log_prob_list = [lp['logprob'] for lp in token_log_probs]
    ppl = math.exp(-sum(log_prob_list) / len(log_prob_list))
    return ppl


def _get_latin_squre(n: int):
    """
    Generate an n x n Latin square.

    Each row is a cyclic shift of the first row, ensuring that every number
    in ``[0, n-1]`` appears exactly once in each row and column.

    Args:
        n (int): Size of the Latin square. Must be positive.

    Returns:
        list[list[int]]: The generated Latin square.

    Raises:
        ValueError: If ``n`` is not a positive integer.
    """
    if n <= 0:
        raise ValueError("N must be positive")

    square = list()
    for i in range(n):
        row = [(i + j) % n for j in range(n)]
        square.append(row)
    return square


def _get_selected(response: dict, fetch_pattern: str = r'<select>Response (\d+)</select>'):
    sel_idx = None
    match = re.search(fetch_pattern, response["response"].split("FINAL DECISION:")[-1])
    if match:
        sel_idx = int(match.group(1)) - 1
    return sel_idx


def _fetch_top_index(response_history, fetch_pattern: str = r'<select>Response (\d+)</select>'):
    """
    Extract selected indices from a history of model responses.

    For each response, this function looks for a pattern indicating a selected
    choice (e.g., "Response 3" inside a <select> tag). If found, it converts
    the matched number to a zero-based index. If not found, defaults to 0.

    Args:
        response_history (list[dict]): List of response records, each containing
            a "response" string.
        fetch_pattern (str): Regular expression used to extract the selected
            response number. Defaults to r'<select>Response (\d+)</select>'.

    Returns:
        set[int]: A set of zero-based indices extracted from the responses.
    """
    sel_idx_set = set()
    for history in response_history:
        sel_idx_set.add(_get_selected(history))
    return sel_idx_set


def _to_history_prompt(history_last_selection: list):
    prompt_list = list()
    for (idx, response, entropy) in history_last_selection:
        if entropy is None:
            prompt_list += [f"### Round {idx}: {response}"]
        else:
            prompt_list += [f"### Round {idx}: {response} (corresponding entropy: {entropy})"]
    prompt = "\n".join(prompt_list)
    return prompt


class ConfidenceGuideSelectNode(AsyncNodeWithCacheAndConcurrencyLimit):
    input_key: Optional[str] = None
    """Key from which candidate solutions (or intermediate results) are retrieved for selection"""

    output_key: str = "selector"
    """Key under which the selected (final) solution is stored"""

    repeat_prompt: int = 1

    _response_history: list[dict] = PrivateAttr(default_factory=list)

    _select_history: list[tuple] = PrivateAttr(default_factory=list)

    def _get_entropy_from_result(self, result: dict):
        meta: Optional[dict] = result.get("meta", None)
        if not meta:
            return 0
        log_probs: Optional[list] = meta.get('token_logprobs', None)
        if log_probs is None:
            return 0
        return _calculate_ppl(log_probs)

    async def _init_select(self, query: str, response_list: list, latin_square: list, num_responses: int):
        wait_select_responses = "\n".join(
            [f"### Response {idx + 1}:\n{response_list[idx]}" for idx in latin_square[0]]
        )
        init_select_prompt = repeat_prompt(
            str(config.prompts.selector_user_prompt).format(
                query=query,
                PARALLEL_NUM=num_responses,
                responses=wait_select_responses
            ),
            self.repeat_prompt
        )
        with stream_writer_env(StreamCtx(meta={"selector": "init"})):
            response = await self.agent(init_select_prompt)
        self._response_history.append(response.model_dump())

    async def _iterative_select(self, query: str, response_list: list, latin_square: list, num_responses: int):
        for i in range(config.solution.selector_iteration):
            last_result = self._response_history[-1]["client_responses"][-1]
            last_content = last_result['content']
            entropy = self._get_entropy_from_result(last_result)
            wait_select_responses = "\n".join(
                [f"### Response {idx + 1}:\n{response_list[idx]}" for idx in latin_square[(i + 1) % num_responses]])
            self._select_history.append(
                (f"{i + 1}", last_content, entropy)
            )
            prompt = repeat_prompt(
                str(config.prompts.selector_iteration_user_prompt).format(
                    query=query,
                    PARALLEL_NUM=num_responses,
                    last_selection=_to_history_prompt(self._select_history),
                    responses=wait_select_responses
                ),
                self.repeat_prompt
            )
            with stream_writer_env(StreamCtx(meta={"selector": f"iter {i}"})):
                response = await self.agent(prompt)
            self._response_history.append(response.model_dump())

    async def _final_select(self, query: str, select_set: set, response_list: list):
        wait_select_responses = "\n".join(
            [f"### Response {idx + 1}:\n{response_list[idx]}" for idx in select_set]
        )
        last_result = self._response_history[-1]["client_responses"][-1]
        last_content = last_result['content']
        entropy = self._get_entropy_from_result(last_result)
        self._select_history.append(
            (f"{config.solution.selector_iteration}", last_content, entropy)
        )

        prompt = repeat_prompt(
            str(config.prompts.selector_iteration_user_prompt).format(
                query=query,
                PARALLEL_NUM=len(select_set),
                last_selection=_to_history_prompt(self._select_history),
                responses=wait_select_responses
            ),
            self.repeat_prompt
        )
        with stream_writer_env(StreamCtx(meta={"selector": "final"})):
            response = await self.agent(prompt)
        self._response_history.append(response.model_dump())

    async def _run(self, state: State) -> StateDelta:
        logger = get_logger()
        query = safe_get_attr(state, "query")

        answer_list = safe_get_attr(state, self.input_key)
        response_list = [parse_answer(answer["response"]) for answer in answer_list]
        num_responses = len(response_list)
        latin_square = _get_latin_squre(num_responses)

        await self._init_select(query, response_list, latin_square, num_responses)
        await self._iterative_select(query, response_list, latin_square, num_responses)

        select_set = _fetch_top_index(self._response_history)
        len_sel_set = len(select_set)
        logger.info(f"Select sets: {select_set}, num. selected: {len_sel_set}")

        if len_sel_set > 1:
            await self._final_select(query, select_set, response_list)

        return {
            self.output_key: {
                "selection_history": self._response_history,
                "selected_response": _get_selected(self._response_history[-1]),
                "response_list": response_list
            },
        }
