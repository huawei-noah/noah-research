#NEW

import glob
import os
import sys
from argparse import Namespace
from typing import Optional, List

import torch

sys.path[0] = os.path.join(os.path.normpath(os.path.join(os.getcwd(), "..")))

from core.bo_runner import run_exp_bo
from core.utils.utils_parsing import get_test_best_config
from core.utils.utils_query import query_exp_best, query_exps
from core.utils.utils_save import exp_folder_rel_path


def main(obj_func_name: str, input_dim: int, acq_func_name: str, num_trials: int, q: int,
         select_optimizers: Optional[List[str]] = None, exclude_optimizers: Optional[List[str]] = None, seed: int = 0,
         do_random_search: bool = 0, root_tuning_dir: str = None, root_test_dir: str = None, save_losses: bool = False,
         save_trajs: bool = False, verbose: float = 1., lb_offset: float = 0, ub_offset: float = 0,
         covar: str = 'matern-5/2', cuda: Optional[int] = None
         ):
    """ Run Bayes optimization using optimizers whose hyper-parameters have been tuned beforehand

    Args:
        obj_func_name: objective function to optimize
        lb_offset: lower bound offset
        ub_offset: upper bound offset
        input_dim: dimension of feature space
        acq_func_name: name of the acquisition function used fo BO
        num_trials: number of trials (each having a different seed)
        q: number of points acquired at each acquisition step
        select_optimizers: list of optimizers to test
        exclude_optimizers: list of optimizers that should not be tested
        seed: initial seed (no need to specify one seed per trial)
        do_random_search: whether to run an experiment with random search
        root_tuning_dir: directory where to find tuning experiments
        root_test_dir: directory where to store this test experiment
    """

    if exclude_optimizers is None:
        exclude_optimizers = []
    if select_optimizers is None:
        select_optimizers = []

    acq_func_name = acq_func_name.replace('Compositional', '')
    test_func_str = f"{obj_func_name}_{input_dim}D"
    if lb_offset is not None and lb_offset != 0:
        test_func_str += f'_lb{str(lb_offset)}'
    if ub_offset is not None and ub_offset != 0:
        test_func_str += f'_ub{str(ub_offset)}'
    covar = covar.lower()
    tuning_dir = os.path.join(root_tuning_dir, test_func_str, f"{acq_func_name}_{q}q")
    optimizer_dirs = glob.glob(tuning_dir + '/*')

    for i, optimizer_dir in enumerate(optimizer_dirs):
        print(optimizer_dir)
        result_dirs = glob.glob(optimizer_dir + '/*/')
        best_opt_config: Namespace = query_exp_best(query_exps(result_dirs, {'covar': covar}))
        if best_opt_config is None or best_opt_config.optimizer in exclude_optimizers:
            continue
        if len(select_optimizers) > 0 and best_opt_config.optimizer not in select_optimizers:
            continue
        if not os.path.exists(os.path.join(optimizer_dir, 'best_hyperparams.pkl')):
            raise ValueError(f"{optimizer_dir} does not contain best_hyperparams.pkl: tuning stopped before end")
        exp_config = best_opt_config
        exp_config._former_cmd = exp_config._cmdline
        exp_config._cmdline = ''
        exp_config.root_result_dir = root_test_dir
        exp_config.num_trials = num_trials
        exp_config.seed = seed
        exp_config.tuned = 1
        exp_config.save_losses = save_losses
        exp_config.save_trajs = save_trajs

        exp_save_path: str = exp_folder_rel_path(exp_config)

        if len(query_exps(exp_save_path, exp_config)) > 0:
            print(f'Best hyperparameters have already been tested: {exp_save_path}')
            continue
        exp_config.verbose = verbose
        exp_config.cuda = cuda
        if not hasattr(exp_config, 'early_stop'):
            exp_config.early_stop = False
        run_exp_bo(exp_config)

    if do_random_search and len(optimizer_dirs) > 0:
        run_random_search(config=best_opt_config, root_test_dir=root_test_dir, num_trials=num_trials, seed=seed)


def run_random_search(config, root_test_dir, num_trials, seed):
    config_random = config
    config_random.optimizer = 'RandomSearch'
    config_random._former_cmd = ''
    config_random._cmdline = ''
    config_random.root_result_dir = root_test_dir
    config_random.num_trials = num_trials
    config_random.acq_func = config.acq_func.replace('Compositional', '')
    config_random.seed = seed
    config_random.tuned = 0
    config_random.cuda = None
    if not hasattr(config_random, 'covar'):
        config_random.covar = 'matern-5/2'

    exp_save_path: str = exp_folder_rel_path(config_random)
    exp_confs: List[Namespace] = query_exps(exp_save_path, {'covar': config_random.covar})
    if len(exp_confs) > 0:
        print(f'Random search has already been tested: {exp_confs[0].result_dir}')
        return

    run_exp_bo(config_random)


def generalize_one(obj_func: str, root_tuning_dir: str, root_test_dir: str, requests, seed: int,
                   num_trials: int) -> Namespace:
    config = query_exp_best(query_exps(root_tuning_dir, requests))
    config.test_func = obj_func
    config._former_cmd = config._cmdline
    config._cmdline = ''
    config.root_result_dir = root_test_dir
    config.num_trials = num_trials
    config.seed = seed
    config.tuned = 0

    run_exp_bo(config)

    return config


def generalize(obj_func: str, root_tuning_dir: str, root_test_dir: str, acq_func: str, input_dim: int, q: int,
               selected_optimizers: List[str], seed: int, num_trials: int, do_random_search: int = 0):
    assert len(selected_optimizers) > 0
    for optimizer in selected_optimizers:
        requests = dict(
            input_dim=input_dim,
            acq_func=acq_func,
            q=q,
            optimizer=optimizer
        )

        config = generalize_one(obj_func, root_tuning_dir, root_test_dir, requests, seed, num_trials)

    if do_random_search:
        run_random_search(config, root_test_dir, num_trials, seed)


if __name__ == '__main__':
    config_test_best_ = get_test_best_config()

    num_cpu = len(os.sched_getaffinity(0))
    torch.set_num_threads(max(1, (3 * num_cpu) // 4))
    torch.set_num_interop_threads(max(1, num_cpu - torch.get_num_threads()))
    print(torch.get_num_threads(), torch.get_num_interop_threads())

    print(config_test_best_.selected_optimizers)
    main(obj_func_name=config_test_best_.test_func, input_dim=config_test_best_.input_dim,
         acq_func_name=config_test_best_.acq_func, num_trials=config_test_best_.num_trials,
         q=config_test_best_.q, select_optimizers=config_test_best_.selected_optimizers,
         exclude_optimizers=config_test_best_.excluded_optimizers, seed=config_test_best_.seed,
         do_random_search=config_test_best_.do_random_search, root_tuning_dir=config_test_best_.root_tuning_dir,
         root_test_dir=config_test_best_.root_test_dir, save_losses=config_test_best_.save_losses,
         save_trajs=config_test_best_.save_trajs,
         verbose=config_test_best_.verbose, lb_offset=config_test_best_.lb_offset,
         ub_offset=config_test_best_.ub_offset, covar=config_test_best_.covar,
         cuda=config_test_best_.cuda)
