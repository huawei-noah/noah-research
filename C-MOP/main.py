# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import time
import asyncio
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from utils.prompt import DEFAULT_SYSTEM_PROMPT
from utils.helpers import (
    get_task_class,
    get_predictor_instance,
    get_evaluator,
    get_scorer,
    get_args,
    ensure_folder_exists
)
from core.inference import multi_inference
from utils.constants import EMBEDDING_MODEL
import optimizers


async def main():
    args = get_args()
    config = vars(args)

    embedder = SentenceTransformer(EMBEDDING_MODEL)

    multi_tasks = args.task.split(",")
    multi_data_dir = args.data_dir.split(",")
    if len(multi_tasks) != len(multi_data_dir):
        raise ValueError("The number of tasks does not match the number of data directories!")

    tasks_all = []
    predictors_multi = []
    for task_name, data_dir in zip(multi_tasks, multi_data_dir):
        task_cls = get_task_class(task_name)
        task_obj = task_cls(data_dir, args.max_threads)
        tasks_all.append(task_obj)
        predictors_multi.append(get_predictor_instance(task_name, config))

    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config, embedder)

    optimizer = optimizers.CMOP(config, evaluator, scorer, args.max_threads, embedder)

    train_exs_multi = [t.get_train_examples() for t in tasks_all]
    test_exs_multi = [t.get_test_examples() for t in tasks_all]

    out_path = Path(args.out)
    ensure_folder_exists(out_path.parent)

    result_log_path = out_path.parent / f"{multi_tasks[0].strip()}" / "result.txt"
    ensure_folder_exists(result_log_path.parent)

    with open(args.out, 'a') as f:
        f.write(json.dumps(config) + '\n')

    candidates = [DEFAULT_SYSTEM_PROMPT]
    tested_cache = {}

    for round_idx in tqdm(range(config['rounds'] + 1), desc="Optimization Rounds"):
        print(f"\n--- STARTING ROUND {round_idx} ---")
        start_time = time.time()

        # optimizing
        if round_idx > 0:
            candidates = optimizer.expand_candidates(candidates, tasks_all, predictors_multi, train_exs_multi)

        scores = optimizer.score_candidates(candidates, tasks_all, predictors_multi, train_exs_multi)
        sorted_candidates = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        scores, candidates = zip(*sorted_candidates[:config['beam_size']])
        scores, candidates = list(scores), list(candidates)

        with result_log_path.open('a', encoding='utf-8') as outf:
            outf.write(f"======== ROUND {round_idx} ========\n")
            outf.write(f"Time: {time.time() - start_time:.2f}s\n")
            outf.write(f"Candidates: {candidates}\n")
            outf.write(f"Scores: {scores}\n")

        # testing
        metrics = []
        for cand, score in zip(candidates, scores):
            if not cand: continue

            if cand in tested_cache:
                metrics.append(tested_cache[cand])
                continue

            task_metrics = []
            for test_exs, predictor, task in zip(test_exs_multi, predictors_multi, tasks_all):
                f1, texts, labels, preds, exs = await multi_inference(
                    predictor, cand, test_exs, task
                )
                task_metrics.append(f1)

            avg_f1 = np.mean(task_metrics)
            metrics.append(avg_f1)
            tested_cache[cand] = avg_f1

        with result_log_path.open('a', encoding='utf-8') as outf:
            outf.write(f"Test Metrics: {metrics} | Mean: {np.mean(metrics):.4f}\n")

    print("\n✅ DONE! Optimization finished.")


if __name__ == '__main__':

    asyncio.run(main())
