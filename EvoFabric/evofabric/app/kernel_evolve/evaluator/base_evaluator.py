# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import traceback
from abc import abstractmethod, ABC

from loguru import logger

from .kernel_eval import eval_kernel_against_ref
from .evaluation_result import Metrics


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self, initial_code, evolve_code) -> Metrics:
        pass

    def _read_file(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()


class GPUEvaluator(BaseEvaluator):
    def evaluate(self, initial_code, evolve_code) -> Metrics:
        try:
            original = self._read_file(initial_code)
            result = eval_kernel_against_ref(original, verbose=True,
                                             custom_model_path=evolve_code,
                                             measure_performance=True,
                                             num_correct_trials=5, num_perf_trials=100)
            if not result.compiled or not result.correctness:
                metrics = {
                    "speedup": 0.0,
                    "original_time": 1000.0,
                    "optimized_time": 1000.0,
                    "error": str(result.metadata)
                }
                return Metrics(**metrics)
            else:
                avg_org = result.org_runtime
                avg_opt = result.runtime
                speedup = avg_org / avg_opt if avg_opt != 0 else 0
                metrics = {
                    "speedup": speedup,
                    "original_time": avg_org,
                    "optimized_time": avg_opt,
                }
            return Metrics(**metrics)

        except Exception as ex:
            logger.error(ex)
            return Metrics(
                **{
                    "passed": 0.0,
                    "speedup": 0.0,
                    "original_time": 1000.0,
                    "optimized_time": 1000.0,
                    "combined_score": 0.0,
                    "error": str(ex),
                    "traceback": traceback.format_exc(),
                },
            )
