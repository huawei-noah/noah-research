# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import asyncio
import nest_asyncio
from core.inference import multi_inference

nest_asyncio.apply()


class CMOPScorer:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        return asyncio.run(self.call_async(*args, **kwargs))

    async def call_async(self, predictors, prompts, data, agg='mean', max_threads=1, tasks=None):
        """
        Evaluates a set of prompt candidates across multiple tasks in parallel and
        returns the aggregated performance scores for each prompt.
        """
        all_coros = []

        for p_idx, prompt in enumerate(prompts):
            for t_idx, (task, predictor, minibatch) in enumerate(zip(tasks, predictors, data)):
                coro = multi_inference(
                    predictor, prompt, minibatch, task
                )
                all_coros.append(coro)

        results = await asyncio.gather(*all_coros)

        final_prompt_scores = []
        for p_idx, prompt in enumerate(prompts):
            task_scores = []
            for t_idx, (task, result) in enumerate(zip(tasks, results)):
                score, texts, labels, preds, exs = result
                task_scores.append(score)
            final_prompt_scores.append(np.mean(task_scores))

        return final_prompt_scores
