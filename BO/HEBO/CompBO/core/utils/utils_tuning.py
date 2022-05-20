#NEW

import os
from argparse import Namespace
from typing import Callable, Any, Dict, Tuple, List, Optional

import numpy as np
from GPyOpt.core.task.objective import SingleObjective

from core.utils.utils_query import query_exps, get_request_from_config
from utils.utils_save import load_np


class ObjectiveDecorator(object):
    def __init__(self, add_args, add_kwargs):
        self.add_args = add_args
        self.add_kwargs = add_kwargs

    def __call__(self, func):
        def new_func(*fargs, **fkwargs):
            new_args = fargs + self.add_args
            new_kwargs = {**fkwargs, **self.add_kwargs}
            return func(*new_args, **new_kwargs)

        return new_func


class CustomSingleObjective(SingleObjective):
    def __init__(self, func: Callable, *args, num_cores: int = 1, objective_name: str = 'no_name',
                 batch_type: str = 'synchronous', space=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.func = func

        @ObjectiveDecorator(self.args, self.kwargs)
        def custom_func(v, *func_args, **func_kwargs):
            return func(v, *func_args, **func_kwargs)

        super(CustomSingleObjective, self).__init__(
            custom_func, num_cores, objective_name, batch_type, space)


def get_params(config: Namespace):
    d: Dict[str, Any] = config.optimizer_kwargs.copy()
    if hasattr(config, 'scheduler_kwargs'):
        d.update(config.scheduler_kwargs)
    return d


def get_perf(config: Namespace) -> np.array:
    return config.optimal_value - load_np(config.result_dir, 'best_values').mean(0)[-1]


def get_X_Y_tuning(config: Namespace) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    configs = query_exps(config.root_result_dir, requests=[get_request_from_config(config)])
    X, Y = [], []
    for conf in configs:
        X.append(get_params(conf))
        Y.append([get_perf(conf)])
    return X, np.array(Y)


def get_result_dir(config: Namespace) -> Optional[str]:
    configs = query_exps(config.root_result_dir, requests=[get_request_from_config(config)])
    if len(configs) == 0:
        return None
    return os.path.dirname(configs[0].result_dir)
