# NEW

import os
from argparse import Namespace
from glob import glob
from typing import Type, Any, List, Callable, Union, Dict, Optional, Tuple
import numpy as np
import torch
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, Kernel
from gpytorch.priors import GammaPrior

from botorch import test_functions, acquisition
from botorch.acquisition import MCAcquisitionFunction
from botorch.optim.initializers import initialize_q_batch_nonneg
from botorch.test_functions import SyntheticTestFunction
from torch.optim.optimizer import Optimizer
from torch.optim import lr_scheduler

from core.es import evolution_opt

import custom_optimizer
from core import comp_acquisition
from core.es import evolution_opt
from utils.utils_save import load_np, load_config, save_config


def query_test_func(test_func_name: str, input_dim: int, negate: bool, l_bound_offset: Optional[float] = None,
                    u_bound_offset: Optional[float] = None) -> SyntheticTestFunction:
    """ Get instance of `botorch` synthetic function

     Args:
         test_func_name: name of the synthetic function `f` (Levy, DixonPrice...)
         input_dim: input dimension
         negate: whether to con consider `-f` instead of `f`
         l_bound_offset: offset added to the lower bound of the hypercube domain of `f`
         u_bound_offset: offset added to the upper bound of the hypercube domain of `f`
     """
    func = getattr(test_functions, test_func_name)
    if hasattr(func, "dim"):
        if func.dim != input_dim:
            raise ValueError(f"{test_func_name} does not allow input dimension {input_dim}, only {func.dim}")
        f = func(negate=negate)
    else:
        f = func(dim=input_dim, negate=negate)
    if l_bound_offset is None:
        l_bound_offset = 0
    if u_bound_offset is None:
        u_bound_offset = 0
    if u_bound_offset != 0 or l_bound_offset != 0:
        print(f'Former bounds: {f.bounds}\nApply offsets: {l_bound_offset, u_bound_offset}')
        for i in range(f.dim):
            f._bounds[i] = (f._bounds[i][0] + l_bound_offset, f._bounds[i][1] + u_bound_offset)
            assert f._bounds[i][1] > f._bounds[i][0]
        f.register_buffer(
            "bounds", torch.tensor(f._bounds, dtype=torch.float).transpose(-1, -2)
        )
        print(f'New bounds: {f.bounds}')
        # Make sure that there is still at least one optimizer in the domain
        optimizers = []
        for optimizer in f._optimizers:
            in_domain = True
            for i in range(len(optimizer)):
                in_domain &= f._bounds[i][0] <= optimizer[i] <= f._bounds[i][1]
            if in_domain:
                optimizers.append(optimizer)
        if len(optimizers) == 0:
            raise ValueError('New bounds are such that no optimizers lay in the new domain.')
        f._optimizers = optimizers
        f.register_buffer(
            "optimizers", torch.tensor(f._optimizers, dtype=torch.float))
    return f


def query_test_func_class(test_func: str) -> Type[SyntheticTestFunction]:
    return getattr(test_functions, test_func)


def query_covar(covar_name: str, X, Y, scale: bool, **kwargs) -> Kernel:
    """ Get covariance module

    Args:
        covar_name: name of the kernel to use ('matern-5/2', 'rbf')
        X: input points at which observations have been gathered
        Y: observations
        scale: whether to use a scaled GP

    Returns:
        An instance of GPyTorch kernel
    """
    ard_num_dims = X.shape[-1]
    aug_batch_shape = X.shape[:-2]
    num_outputs = Y.shape[-1]
    if num_outputs > 1:
        aug_batch_shape += torch.Size([num_outputs])
    lengthscale_prior = GammaPrior(3.0, 6.0)

    kws = dict(ard_num_dims=ard_num_dims, batch_shape=aug_batch_shape,
               lengthscale_prior=kwargs.pop('lengthscale_prior', lengthscale_prior))
    if covar_name.lower()[:6] == 'matern':
        kernel_class = MaternKernel
        if covar_name[-3:] == '5/2':
            kws['nu'] = 2.5
        elif covar_name[-3:] == '3/2':
            kws['nu'] = 1.5
        elif covar_name[-3:] == '1/2':
            kws['nu'] = .5
        else:
            raise ValueError(covar_name)
    elif covar_name.lower() == 'rbf':
        kernel_class = RBFKernel
    else:
        raise ValueError(covar_name)
    kws.update(**kwargs)
    outputscale_prior = kws.pop('outputscale_prior', GammaPrior(2.0, 0.15))

    base_kernel = kernel_class(**kws)
    if not scale:
        return base_kernel

    return ScaleKernel(base_kernel, batch_shape=aug_batch_shape, outputscale_prior=outputscale_prior)


def query_AcqFunc(acq_func: str, **acq_func_kwargs) -> Type[MCAcquisitionFunction]:
    """ Return the class of Acquisition function """
    if acq_func == "qMaxValueEntropy":
        acq_func_kwargs["num_candidates"] = acq_func_kwargs.get("candidate_set", 100)
    if hasattr(acquisition, acq_func):
        return getattr(acquisition, acq_func)
    elif hasattr(comp_acquisition, acq_func):
        return getattr(comp_acquisition, acq_func)
    else:
        raise ValueError(f'{acq_func} not found.')


def query_optimizer(optimizer: str) -> Type[Optimizer]:
    """ Get the class of Optimizer associated to `optimizer` """

    if hasattr(torch.optim, optimizer):
        return getattr(torch.optim, optimizer)
    elif hasattr(custom_optimizer, optimizer):
        return getattr(custom_optimizer, optimizer)
    if hasattr(evolution_opt, optimizer):
        return getattr(evolution_opt, optimizer)
    else:
        raise ValueError(f"Unavailable optimizer: {optimizer}")


def query_scheduler(scheduler: str) -> Optional[Type[lr_scheduler._LRScheduler]]:
    """ Get the class of optimizer scheduler associated to `scheduler` """
    if scheduler is None or scheduler == 'None':
        return None
    if hasattr(lr_scheduler, scheduler):
        return getattr(lr_scheduler, scheduler)
    else:
        raise ValueError(f"Unavailable optimizer: {scheduler}")


def query_ts_data(root_result_dir: str, test_func: str, input_dim: int, typ: str):
    """ Get Thompson Sampling results """
    mean = np.loadtxt(os.path.join(root_result_dir, f'log_regret_{typ}_{test_func}_{input_dim}.txt'))
    uc = np.loadtxt(os.path.join(root_result_dir, f'log_regret_{typ}_upper_{test_func}_{input_dim}.txt'))
    lc = np.loadtxt(os.path.join(root_result_dir, f'log_regret_{typ}_lower_{test_func}_{input_dim}.txt'))
    return mean, uc, lc


def query_init_q_batch_func() -> Callable:
    """ Return a heuristic for selecting initial conditions for acquisition functions """
    return initialize_q_batch_nonneg


def get_request_from_config(config: Namespace) -> Dict[str, Any]:
    if not hasattr(config, 'lb_offset'):
        config.lb_offset = 0
    if not hasattr(config, 'ub_offset'):
        config.ub_offset = 0
    if not hasattr(config, 'covar'):
        config.covar = 'matern-5/2'
    return dict(
        input_dim=config.input_dim,
        acq_func=config.acq_func,
        test_func=config.test_func,
        q=config.q,
        optimizer=config.optimizer,
        lb_offset=config.lb_offset,
        ub_offset=config.ub_offset,
        covar=config.covar
    )


def check_request(config: Namespace, request: Union[Namespace, Dict[str, Any], Callable[[Namespace], bool]]) -> bool:
    if isinstance(request, Namespace):
        request = get_request_from_config(request)

    if isinstance(request, dict):
        good = True
        for k, v in request.items():
            if k in ['lb_offset', 'ub_offset']:
                if not hasattr(config, k) and v == 0:
                    continue
            if k == 'covar':
                if not hasattr(config, 'covar') and v == 'matern-5/2':
                    continue
            good &= hasattr(config, k) and getattr(config, k) == v
            if not good:
                break
        return good
    return request(config)


# -------------------------------- Exp-related -------------------------------- #

def query_exps(result_dirs: Union[str, List[str]],
               requests: Optional[
                   Union[Namespace, List[Union[Dict[str, Any], Callable[[Namespace], bool]]], Dict[str, Any]]] = None,
               recursive=True,
               key: Optional[Union[Callable, str]] = None) \
        -> List[Namespace]:
    """ Get list of `config` stored in one of the specified `result_dirs` (and their subfolders if `recursive`)
        matching the `requests`

    Args:
        result_dirs: path(s) in which to look for config files
        requests: list of constraints specifying the kind of configs to select (e.g. query for an optimizer,...)
        recursive: if true, will look into subfolders too
        key: criterion on which to sort selected `configs`

    Returns:
         A list of configs found in the `result_dirs` and matching all the requests
    """

    configs: List[Namespace] = []
    if requests is None:
        requests = []
    if not isinstance(requests, list):
        requests = [requests]
    if isinstance(result_dirs, str):
        result_dirs = [result_dirs]
    for result_dir in result_dirs:
        config_dirs = glob(result_dir + '/**/*.pkl', recursive=recursive)
        for config_dir in config_dirs:
            try:
                config = load_config(config_dir)
            except EOFError as e:
                print(config_dir)
                raise (e)
            if not isinstance(config, Namespace):
                continue
            add_config = True
            for request in requests:
                add_config &= check_request(config, request)
                if not add_config:
                    break
            if add_config:
                # modify registered result directory if needs be
                dirname = os.path.dirname(config_dir)
                if dirname != config.result_dir:
                    config.result_dir = dirname
                    save_config(config)
                configs.append(config)
    if len(configs) > 0 and key is not None:
        if isinstance(key, str):
            key: Callable = lambda conf: getattr(conf, key)
        configs.sort(key=key)
    return configs


def query_exp_best(configs: List[Namespace], return_best_regret=False) -> Union[Namespace, Tuple[Namespace, float]]:
    """ Return best experiment in terms of regret given a list of experiment configurations

    Args:
        configs: list of experiment configurations to consider
        return_best_regret: if True, not only return best configuration but also the best regret obtained

    Returns:
        Best configuration and optionally best regret
    """

    best_regret = np.inf
    best_config = None
    for config in configs:
        result_dir: str = config.result_dir
        best_values = load_np(result_dir, "best_values")
        regret = np.abs(best_values.mean(axis=0)[-1] - config.optimal_value)
        if regret < best_regret:
            best_config = config
            best_regret = regret
    if return_best_regret:
        return best_config, best_regret
    return best_config


def query_best_step_tuning(configs: List[Namespace]) -> int:
    """ Query the step at which best hyperparameters have been found

    Args:
        configs: list of experiment configurations to consider (should correspond to the results of hyperparameters
                 tuning)

    Returns:
        Step at which best hyperparameters have been found
    """
    sorted_configs = sorted(configs, key=lambda config: config.result_dir)
    best_config, best_regret = query_exp_best(configs, return_best_regret=True)
    return sorted_configs.index(best_config)


def query_final_immediate_regret(config: Namespace, log: bool = True, std: bool = False) -> \
        Union[float, Tuple[float, float]]:
    """ Query final immediate regret averaged over trials

    Args:
        config: experiment configuration
        log: whether to work with log (base 10) of immediate regrets
        std: whether to return also standard deviation

    Returns:
          final_regret_mean: mean of final regrets
          final_regret_std (Optional): std of final regrets
    """
    final_regrets = np.abs(load_np(config.result_dir, 'best_values')[:, -1] - config.optimal_value)
    if log:
        final_regrets = np.log10(final_regrets)
    final_regret_mean = final_regrets.mean(0)
    if std:
        final_regret_std = final_regrets.std(0)
        return final_regret_mean, final_regret_std
    return final_regret_mean
