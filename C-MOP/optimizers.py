# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
import random
import re
import requests
import time
import asyncio
import nest_asyncio
from jinja2 import Template
from pathlib import Path
from sklearn.cluster import KMeans
from typing import List, Tuple, Any, Set

from core.gradient_manager import GradientBatcher, GradientMomentumManager
from core.inference import calculate_score, optimizer_model_infer
from core.inference import multi_inference
from core.sampler import ContrastiveClusterSampler
from utils.helpers import GLOBAL_CONFIG
from utils.prompt import PROMPT_IMPROVEMENT_TEMPLATE

nest_asyncio.apply()


class CMOP:
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, embedder=None):
        self.opt = args
        self.out_txt = Path(self.opt["out"]).parent / "info.txt"
        self.out_txt.parent.mkdir(parents=True, exist_ok=True)
        self.max_threads = max_threads

        self.evaluator_fn, self.scorer = evaluator_fn, scorer
        self.embedder = embedder
        self.current_step = 0
        self.prompt_with_gradient_history = {}

        self.sampler = ContrastiveClusterSampler()
        self.momentum_manager = GradientMomentumManager()

    @staticmethod
    def parse_tags(text, start="<START>", end="<END>"):
        return re.findall(f'{re.escape(start)}(.*?){re.escape(end)}', text, re.DOTALL)

    def _get_case_str(self, task, predictor, t, l, p, ex):
        """
        Constructs a standardized string for a single sample by mapping task-specific
        instructions to the prediction and the ground truth label.
        """
        if hasattr(task, "cfinbench_postprocess"):
            q_type = ex.get("question_type")
            mapping = {"multi_choice": "multi_user_prompt", "single_choice": "single_user_prompt",
                       "judgment": "judgement_user_prompt"}
            attr = mapping.get(q_type, "task_instruct")
            instruct = getattr(predictor, attr)
        elif hasattr(task, "stringify_prediction"):
            l = task.stringify_prediction(l)
            p = task.stringify_prediction(p)
            instruct = predictor.task_instruct
        else:
            instruct = predictor.task_instruct

        query = Template(instruct).render(text=t.strip())
        return (f'**[User Query]:** "{query}"\n'
                f'**[Desired Output (Label)]:** {l}\n'
                f'**[Actual LLM Output (Prediction)]:** {p}\n\n')

    def _sample_error_str(self, texts, labels, preds, task, predictor, exs, n=4):
        """ Categorizes model outputs into 'correct' and 'error' sets."""
        errors, corrects = [], []
        for i in range(len(preds)):
            case_str = self._get_case_str(task, predictor, texts[i], labels[i], preds[i], exs[i])
            is_correct = calculate_score(task, [labels[i]], [preds[i]], [exs[i]]) >= 100
            (corrects if is_correct else errors).append(case_str)

        s_errors = random.sample(errors, min(len(errors), n))
        s_corrects = random.sample(corrects, min(len(corrects), n))
        return s_errors, s_corrects, s_corrects + s_errors

    def score_candidates(self, prompts, tasks, predictors, train_exs_multi_task):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(
            prompts, train_exs_multi_task, tasks, predictors,
            scorer=self.scorer,
            rounds=self.opt['eval_rounds'],
            num_prompts_per_round=self.opt['eval_prompts_per_round'],
            samples_per_eval=self.opt['samples_per_eval'],
            max_threads=self.max_threads
        )
        return evals

    async def mutate_add_constraint(self, prompt: str, constraint: str) -> List[str]:
        """
        Executes the prompt improvement step by integrating a specific textual
        gradient into the existing prompt template.
        """
        u_prompt = PROMPT_IMPROVEMENT_TEMPLATE.format(prompt=prompt, new_constraint=constraint)
        res = await self._mulit_inference("", u_prompt, temperature=0.2)
        return self.parse_tags(res)

    async def extract_multi_consensus_gradients(self, prompt, m_labels, m_preds, m_texts, m_exs, predictors, tasks,
                                                f_clusters, g_clusters, step):
        """
        Extracts stable textual gradients by clustering instance and applying
        momentum-based consensus across iterations.
        """
        all_err, all_corr, all_cases = [], [], []
        for task, pred, txt, lbl, prd, ex in zip(tasks, predictors, m_texts, m_labels, m_preds, m_exs):
            e, c, a = self._sample_error_str(txt, lbl, prd, task, pred, ex, n=len(prd))
            all_err.extend(e)
            all_corr.extend(c)
            all_cases.extend(a)

        if not all_err: return []

        emb, lbls, centers = self._cluster_cases(all_cases, f_clusters)
        final_samples = self.sampler.sample(all_cases, lbls, emb, centers, all_err, all_corr, self.out_txt)
        batcher = GradientBatcher(samples_per_gradient=1)
        tasks_list = [self._mulit_inference("", batcher.format_meta_prompt(mini_batch, prompt), 0.3) for mini_batch in
                      batcher.create_mini_batches(final_samples)]
        results = await asyncio.gather(*tasks_list)
        gradients = []
        for gradient_text in results:
            gradients += self.parse_tags(gradient_text)

        g_emb, g_lbls, _ = self._cluster_cases(gradients, g_clusters)
        self.momentum_manager.add_new_gradients(gradients, g_emb, step)
        return self.momentum_manager.get_momentum_gradients(g_clusters, self.out_txt)

    def _cluster_cases(self, texts, n):
        """ Performs semantic clustering on a set of text embedding using K-Means."""
        if not texts: return np.array([]), [], []
        n = min(len(texts), n)
        embeddings = self.embedder.encode(texts)
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        return embeddings, kmeans.fit_predict(embeddings), kmeans.cluster_centers_

    async def _mulit_inference(self, sys, user, temperature):
        """
        Wraps the synchronous optimizer model inference call in an asynchronous executor to
        enable non-blocking concurrent processing during the optimization loop.
        """
        loop = asyncio.get_running_loop()

        return await loop.run_in_executor(None, optimizer_model_infer, user, sys, temperature)

    async def generate_ea_candidates_evoprompt(
            self,
            population: List[str],
            n_candidates: int
    ) -> List[str]:
        """
        Generates a set of new prompt candidates applying historical textual gradients to the current population.
        """
        print(f"Generating {n_candidates} EA candidates (Async EvoPrompt style)...")
        candidates: Set[str] = set()

        if not population:
            print("Warning: EA population is empty.")
            return []

        while len(candidates) < n_candidates:
            needed = n_candidates - len(candidates)
            batch_size = max(needed, 2)
            tasks = []
            for _ in range(batch_size):
                parent = random.choice(population)
                history = self.prompt_with_gradient_history.get(parent, [])
                gradients_to_apply = random.sample(history, min(len(history), 4))

                gradient_str = "\n".join([f"Gradient {i + 1}: {g}" for i, g in enumerate(gradients_to_apply)])

                tasks.append(self.mutate_add_constraint(parent, gradient_str))

            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            for res in results_list:
                if isinstance(res, list):
                    for p in res:
                        if p not in population:
                            candidates.add(p)
                if len(candidates) >= n_candidates: break

        return list(candidates)[:n_candidates]

    async def expand_candidates_async(self, prompts, tasks, predictors, train_exs_multi_task):
        """
        Expanding current prompts by evaluating current prompts, extracting consensus gradients, and generating new
        evolved candidates.
        """
        infer_tasks = [
            multi_inference(predictor, prompt, random.sample(exs, self.opt['minibatch_size']), task)
            for prompt in prompts for task, predictor, exs in zip(tasks, predictors, train_exs_multi_task)
        ]
        results = await asyncio.gather(*infer_tasks)

        n_tasks = len(tasks)
        for i, prompt in enumerate(prompts):
            p_results = results[i * n_tasks: (i + 1) * n_tasks]
            m_lbls, m_prds, m_txts, m_exs = zip(*[(r[2], r[3], r[1], r[4]) for r in p_results])

            self.prompt_with_gradient_history[prompt] = await self.extract_multi_consensus_gradients(
                prompt, m_lbls, m_prds, m_txts, m_exs, predictors, tasks,
                GLOBAL_CONFIG["FAILURE_N_CLUSTERS"], GLOBAL_CONFIG["GRADIENT_N_CLUSTERS"], self.current_step
            )

        new_cands = await self.generate_ea_candidates_evoprompt(prompts, GLOBAL_CONFIG["N_CANDIDATES"])
        self.current_step += 1
        return list(set(new_cands + prompts))

    def expand_candidates(self, *args, **kwargs):
        return asyncio.run(self.expand_candidates_async(*args, **kwargs))
