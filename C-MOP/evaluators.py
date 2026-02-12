# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import urllib3
import random
from tqdm import tqdm
import concurrent.futures
import requests
import numpy as np


class UCBBandits:
    """ Upper Confidence Bound Bandits """

    def __init__(self, num_prompts, num_samples=5, c=1.0, mode='ucb'):
        self.c = c
        assert mode in {'ucb', 'ucb-e'}
        self.mode = mode
        self.num_prompts = num_prompts
        self.num_samples = num_samples
        self.reset()

    def update(self, chosen, scores):
        for i, score in zip(chosen, scores):
            self.counts[i] += self.num_samples
            self.scores[i] += score * self.num_samples

    def reset(self):
        self.counts = np.zeros(self.num_prompts)
        self.scores = np.zeros(self.num_prompts)

    def get_scores(self):
        # Some counts may be 0, so we need to avoid division by 0.
        return np.divide(self.scores, self.counts, out=np.zeros_like(self.scores), where=self.counts != 0)

    def choose(self, n, t):
        if np.sum(self.counts) == 0:
            # If all counts are 0, choose randomly.
            return random.sample(range(self.num_prompts), n)
        scores = self.get_scores()
        counts = self.counts + 1e-3
        if self.mode == 'ucb':
            ucb_scores = scores + self.c * np.sqrt(np.log(t) / counts)
        elif self.mode == 'ucb-e':
            ucb_scores = scores + self.c * np.sqrt(self.c / counts)
        return np.argsort(ucb_scores)[::-1][:n]

    def get_infos(self):
        return self.counts


class UCBBanditEvaluator:
    """ Upper Confidence Bound Evaluator"""

    def __init__(self, config, embedder=None):
        self.config = config

    def __call__(self, prompts, train_exs_multi_task, tasks, predictors, scorer,
                 rounds=40, num_prompts_per_round=10, samples_per_eval=5, max_threads=1, verbose=True):

        assert self.config['evaluator'] in {'ucb', 'ucb-e'}, f'unk evaluator: {self.config["evaluator"]}'
        bandit_algo = UCBBandits(
            len(prompts), num_samples=samples_per_eval,
            mode=self.config['evaluator'],
            c=self.config['c']
        )

        def data_sampler(l):
            return random.sample(l, samples_per_eval)

        num_prompts_per_round = min(num_prompts_per_round, len(prompts))

        for ri in tqdm(range(rounds), desc=f'Evaluating {len(prompts)} prompts'):
            sampled_prompts_idx = bandit_algo.choose(num_prompts_per_round, ri)
            sampled_prompts = [prompts[i] for i in sampled_prompts_idx]
            sampled_data = []
            for task_train_exs in train_exs_multi_task:
                sampled_data.append(data_sampler(task_train_exs))
            while True:
                try:
                    scores = scorer(predictors, sampled_prompts, sampled_data, max_threads=max_threads, tasks=tasks)
                    break
                except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError,
                        urllib3.exceptions.MaxRetryError):
                    pass
            bandit_algo.update(sampled_prompts_idx, scores)

        return bandit_algo.get_scores().tolist()
