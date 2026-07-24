# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import argparse
import sys
import os
from pathlib import Path
from typing import Union, Dict

current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
import tasks
import evaluators
import scorers
import predictors


def load_global_config() -> Dict:
    instance_n_cluster = os.environ.get('FAILURE_N_CLUSTERS', 14)
    gradient_n_cluster = os.environ.get('GRADIENT_N_CLUSTERS', 10)
    n_candidate = os.environ.get('N_CANDIDATES', 40)

    return {
        "FAILURE_N_CLUSTERS": int(instance_n_cluster),
        "GRADIENT_N_CLUSTERS": int(gradient_n_cluster),
        "N_CANDIDATES": int(n_candidate),
    }


GLOBAL_CONFIG = load_global_config()

TASK_MAPPING = {
    'liar': tasks.DefaultHFBinaryTask,
    'gsm8k': tasks.Gsm8kTask,
    'cfinbench': tasks.CfinBenchTask,
    'bbh': tasks.BBHTask
}

PREDICTOR_MAPPING = {
    'gsm8k': predictors.Gsm8kPredictor,
    'cfinbench': predictors.CfinBenchPredictor,
    'bbh': predictors.BBHPredictor,
    'default': predictors.BinaryPredictor
}

EVALUATOR_MAPPING = {
    'ucb': evaluators.UCBBanditEvaluator,
}

SCORER_MAPPING = {
    'CMOPScorer': scorers.CMOPScorer
}


def get_task_class(task_name: str):
    name = task_name.strip().lower()
    if name not in TASK_MAPPING:
        raise ValueError(f"❌ Unsupported task: {name}")
    return TASK_MAPPING[name]


def get_predictor_instance(task_name: str, config: dict):
    name = task_name.strip().lower()
    predictor_cls = PREDICTOR_MAPPING.get(name, PREDICTOR_MAPPING['default'])
    return predictor_cls(config)


def get_evaluator(name: str):
    return EVALUATOR_MAPPING.get(name.strip().lower())


def get_scorer(name: str):
    return SCORER_MAPPING.get(name)


def ensure_folder_exists(path: Union[str, Path]):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='liar')
    parser.add_argument('--data_dir', default='data/liar')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--max_threads', default=64, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--rounds', default=20, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--minibatch_size', default=128, type=int)

    parser.add_argument('--evaluator', default="ucb", type=str)
    parser.add_argument('--scorer', default="CMOPScorer", type=str)
    parser.add_argument('--eval_rounds', default=9, type=int)
    parser.add_argument('--eval_prompts_per_round', default=10, type=int)

    parser.add_argument('--samples_per_eval', default=256, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')

    args = parser.parse_args()

    return args
