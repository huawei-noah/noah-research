# NEW
import os
from argparse import Namespace
from typing import Optional, Any, List, Dict

import numpy as np

from utils.utils import get_timestr
from utils.utils_save import save_w_pickle, load_w_pickle, save_np, load_np


def rename_compositional_acq_func(acq_func: str):
    return acq_func.replace('Compositional', '')


def get_exp_id(config: Namespace) -> str:
    """ Get experiment experiment id from its `config`

    Args:
        config: experiment configuration

    Returns: experiment id

    """
    if not hasattr(config, 'ub_offset'):
        config.ub_offset = 0
    if not hasattr(config, 'lb_offset'):
        config.lb_offset = 0
    if not hasattr(config, 'covar'):
        config.covar = 'matern-5/2'
    return get_exp_id_aux(config.test_func, config.input_dim, config.acq_func, config.q, config.optimizer,
                          config.lb_offset, config.ub_offset, config.covar)


def get_exp_id_aux(test_func: str, input_dim: int, acq_func: str, q: int, optimizer: str,
                   lb_offset: Optional[float] = None, ub_offset: Optional[float] = None,
                   covar: Optional[str] = None) -> str:
    """ Get experiment experiment id its main features

    Args:
        test_func:
        input_dim:
        acq_func:
        optimizer:

    Returns: experiment id

    """
    exp_id = f"{test_func}-{input_dim}-{optimizer}-{acq_func}-{q}"
    if lb_offset is not None and lb_offset != 0:
        exp_id += f'-lb{str(lb_offset)}'
    if ub_offset is not None and ub_offset != 0:
        exp_id += f'-ub{str(ub_offset)}'
    if covar is not None:
        exp_id += f"-COV{covar.replace('/', '').replace('-', '')}"
    exp_id += f"-{get_timestr()}"
    return exp_id


def exp_folder_rel_path(config: Namespace) -> str:
    """ Get experiment path from its `config`

    Args:
        config: experiment configuration

    Returns: relative path to experiment folder

    """
    if not hasattr(config, 'ub_offset'):
        config.ub_offset = 0
    if not hasattr(config, 'lb_offset'):
        config.lb_offset = 0

    return exp_folder_rel_path_aux(
        root_result_dir=config.root_result_dir,
        test_func=config.test_func,
        input_dim=config.input_dim,
        acq_func=config.acq_func,
        q=config.q,
        optimizer=config.optimizer,
        lb_offset=config.lb_offset,
        ub_offset=config.ub_offset,
    )


def exp_folder_rel_path_aux(root_result_dir: str, test_func: str, input_dim: int, acq_func: str, q: int,
                            optimizer: str, lb_offset: Optional[float] = None,
                            ub_offset: Optional[float] = None) -> str:
    """ Get experiment path from its main features

    Args:
        root_result_dir:
        test_func:
        input_dim:
        acq_func:
        optimizer:

    Returns: relative path to experiment folder

    """
    test_func_str = f'{test_func}_{input_dim}D'
    if lb_offset is not None and lb_offset != 0:
        test_func_str += f'_lb{str(lb_offset)}'
    if ub_offset is not None and ub_offset != 0:
        test_func_str += f'_ub{str(ub_offset)}'

    return os.path.join(root_result_dir,
                        test_func_str,
                        f'{rename_compositional_acq_func(acq_func)}_{q}q',
                        f'{optimizer}')


def create_exp_folder(config: Namespace, root: Optional[str] = None) -> str:
    """ Create folder where results of the experiment will be stored

    `root`
        |__ `root_result_dir`                   \
            |__`objective funtion_dim_lb_ub`          |--  'Exp folder
                |__`Acquisition function`-`q`q  |--    relative path'
                    |__`Optimizer`             /
                        |__`exp_id`
                            |__ ...
                            |__...

    Args:
        config: configuration of the experiment
        root: root directory from which experiment folder will be built

    Returns:
        result_dir: string corresponding to the exp_path of the created folder
    """
    if root is None:
        root = os.getcwd()
    rel_path: str = exp_folder_rel_path(config)
    exp_id: str = get_exp_id(config)
    result_dir = os.path.join(root, rel_path, exp_id)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def save_config(config: Namespace, filename: str = 'config') -> None:
    save_w_pickle(config, config.result_dir, filename)


def load_config(path_or_filepath: str, filename: str = 'config') -> Namespace:
    if len(path_or_filepath) > 4 and path_or_filepath[-4:] == '.pkl':
        path, filename = os.path.split(path_or_filepath)
    else:
        path = path_or_filepath
    return load_w_pickle(path, filename)


def save_exp_info(info: str, path: str) -> None:
    exp_file = open(os.path.join(path, "info.txt"), "w")
    exp_file.write(info)
    exp_file.close()


def load_exp_info(path: str) -> str:
    with open(os.path.join(path, "info.txt")) as f:
        return f.readline()


def save_trajs(X_trajs_s: List[List[np.ndarray]], path: str) -> None:
    """ Save optimization trajectories """
    X_trajs_s = np.array(X_trajs_s)
    assert X_trajs_s.ndim == 6  # num_trials, num_acq_steps, num_opt_steps, num_restarts, q, d
    traj_path = os.path.join(path, 'trajs')
    os.mkdir(traj_path)
    n_trials = X_trajs_s.shape[0]
    for trial in range(n_trials):
        trial_path = os.path.join(traj_path, str(trial))
        os.mkdir(trial_path)
        X_trajs = X_trajs_s[trial]
        save_np(X_trajs, trial_path, 'trajs')


def load_trajs(exp_path: str, trial: int = 0, filename: Optional[str] = "trajs") -> np.ndarray:
    """ Load optimization trajectories for a specific trial """
    trajs_path = os.path.join(exp_path, "trajs", str(trial))
    return load_np(trajs_path, filename)


def save_losses(losses_s: List[List[List[float]]], path: str, filename: Optional[str] = "losses") -> None:
    losses_path = os.path.join(path, 'losses')
    os.mkdir(losses_path)
    losses_s = np.array(losses_s)
    assert losses_s.ndim == 4  # num_trials, num_acq_steps, num_opt_steps, num_restarts
    save_np(losses_s, losses_path, filename)


def load_losses(exp_path: str, filename: Optional[str] = "losses") -> np.ndarray:
    """ Load optimizer loss as an array of shape (num_trials x num_acq_steps x num_opt_steps)"""
    losses_path = os.path.join(exp_path, "losses")
    return load_np(losses_path, filename)


def save_execution_times(execution_times: List[List[float]], path: str, filename: str = "execution_times"):
    execution_times = np.array(execution_times)
    assert execution_times.ndim == 2  # num_trials x num_acq_steps
    save_np(execution_times, path, filename)


def load_execution_times(path: str, filename: str = "execution_times") -> np.ndarray:
    return load_np(path, filename)


def save_best_hyperparams(hyperparams: Dict[str, Any], config: Namespace):
    result_dir = exp_folder_rel_path(config)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    save_w_pickle(hyperparams, result_dir, "best_hyperparams")


def load_best_hyperparams(config: Namespace) -> Dict[str, Any]:
    result_dir = exp_folder_rel_path(config) + '/'
    return load_w_pickle(result_dir, "best_hyperparams")
