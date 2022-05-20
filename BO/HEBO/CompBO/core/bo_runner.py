#NEW
import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict, Any, Type, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction, qExpectedImprovement, OneShotAcquisitionFunction
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement, qUpperConfidenceBound
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import gen_batch_initial_conditions, ExpMAStoppingCriterion
from botorch.optim.utils import columnwise_clamp
from botorch.sampling import MCSampler
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer

sys.path[0] = str(Path(os.path.realpath(__file__)).parent.parent)

from core.bayes_opt import BayesOptimization
from core.params_helper import ParamSpace
from utils import utils_save
from core.utils.utils_parsing import get_exp_config
from core.utils.utils_query import query_test_func, query_exps
from core.utils.utils_save import create_exp_folder, save_exp_info, save_config, save_trajs, save_execution_times
from utils.utils_save import save_np, load_np

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)


def get_initial_starts(acq_func: AcquisitionFunction, raw_samples: int, num_starts: int, q: int, bounds: Tensor,
                       init_func: Callable) -> Tensor:
    """ Optimizing acquisition function
        Args:
            acq_func:  acquisition function to optimize
            raw_samples: The number of random restart (multi-start sgd)
            num_starts: The number of initial condition to be generated. Must be less than `b`
            q:  number of new acquisition points we look for
            bounds:  bounds on the acquisition points (shape: 2 x d)
            init_func:  function called to initialize the multi-start sgd

        Returns:
             X tensor of shape N x q x d from which initialization will start
    """
    dim = bounds.shape[1]
    Xraw = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(raw_samples, q, dim, dtype=float)
    with torch.no_grad():
        Yraw = acq_func(Xraw)  # evaluate the acquisition function on these q-batches
    # apply the heuristic for sampling promising initial conditions
    X = init_func(Xraw, Yraw, num_starts)
    return X


def norm_bounds(d: int) -> Tensor:
    """ Return tensor of shape 2 x d corresponding to the bounds of [0, 1]^d """
    return torch.stack([torch.zeros(d), torch.ones(d)])


def optimize_acqf_and_get_observation(acq_func: AcquisitionFunction, num_starts: int, raw_samples: int, q: int,
                                      bounds: Tensor, init_func: Callable, opt: Type[Optimizer],
                                      opt_kwargs: Dict[str, Any] = None,
                                      scheduler_class: Optional[Type[_LRScheduler]] = None,
                                      scheduler_kwargs: Dict[str, Any] = None,
                                      num_opt_steps: int = 50, verbose: int = 0, return_traj: bool = False,
                                      return_loss: bool = False, seed: Optional[int] = None) -> Tuple[
    Tensor, Dict[str, Any]]:
    """ Optimize acquisition function

    Args:
        acq_func:  The acquisition function to optimize
        num_starts: The number of starting points for multistart acquisition function optimization.
        raw_samples:  The number of samples for initialization.
        q:  number of new acquisition points we look for
        bounds:  A `2 x d` tensor of lower and upper bounds for each column of `X`.
        init_func:  A function called to initialize the multi-start sgd
        opt:  A torch optimizer from subclass (Optimizer)
        opt_kwargs: A string-specified dictionary of parameters for optimizer
        scheduler_class: A class of torch scheduler associated to instentiate with the optimizer
        scheduler_kwargs: A string-specified dictionary of parameters for scheduler
        num_opt_steps: The number of optimization step
        verbose: level of message passed (0: nothing is printed)
        return_traj: Whether to return optimization trajectory
        return_loss: whether to return optimizer loss
        seed: seed for reproducibility

    Returns:
         X tensor of shape q x d best candidate maximize of acquisition function
         meta_dic : dictionary that may contains entries:
            - 'trajs': store X trajectory if `return_traj` set to True
            - 'loss': store optimizer loss across trajectory if `return_loss` set to True
    """
    # we'll want gradients for the input
    q_q_fantasies = q  # q + q_fantaisies when using Knowledge Gradient
    if isinstance(acq_func, OneShotAcquisitionFunction):
        q_q_fantasies += acq_func.num_fantasies
    X: Tensor = gen_batch_initial_conditions(acq_func, bounds.double(), q_q_fantasies, num_starts, raw_samples,
                                             options={'seed': seed})
    # get_initial_starts(acq_func, raw_samples, num_starts, q_q_fantasies, bounds,init_func)  # shape num_starts x q_q_fantasies x d
    assert X.shape == (num_starts, q_q_fantasies, bounds.shape[
        -1]), f"X.shape should be {(num_starts, q_q_fantasies, bounds.shape[-1])} but got {X.shape}"
    X.requires_grad_(True)
    # set up the optimizer, make sure to only pass in the candidate set here
    if opt_kwargs is None:
        opt_kwargs = {}
    if scheduler_kwargs is None:
        scheduler_kwargs = {}

    # set parameters to optimize
    params = (dict(params=X),)
    if hasattr(acq_func, "is_compositional"):
        with torch.no_grad():
            Y: Tensor = acq_func.oracle_g(X.clone())
        Y.requires_grad_(True)
        params += (dict(params=Y),)
    optimizer = opt(params, **opt_kwargs)
    scheduler = None
    if scheduler_class:
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    if return_traj:
        X_traj: List[Tensor] = [X[:, :q].detach().clone()]
    if return_loss:
        losses = []

    stopping_criterion = ExpMAStoppingCriterion(maxiter=num_opt_steps)

    # run a basic optimization loop
    for i in range(num_opt_steps):
        optimizer.zero_grad()

        if hasattr(acq_func, "is_compositional"):
            eval_J = False  # whether to evaluate f(g(X)) during `opt_forward`
            forw_results = acq_func.opt_forward(X, Y, eval_J=eval_J)
            g, f_Y = forw_results[:2]
            if eval_J:
                losses_step: Tensor = - forw_results[-1]
                with torch.no_grad():
                    loss: Tensor = losses_step.sum().detach()  # if J(x) = f(g(x)) has been evaluated
            f_Y = - f_Y.sum()  # we want to maximize
            f_Y.backward()
            g_x: Optional[Tensor] = optimizer.step(oracle_g=g)
            if not eval_J:
                losses_step: Tensor = - acq_func.oracle_f(g_x)  # (`batch_size`,) tensor
                loss: Tensor = losses_step.sum().detach()
        else:
            # this performs batch evaluation, so this is an `batch_size`-dim tensor
            losses_step: Tensor = - acq_func(X)  # shape: (N,)
            loss = losses_step.sum()

            loss.backward()  # perform backward pass
            optimizer.step()  # take a step

        if scheduler:
            scheduler.step()

        stop: bool = stopping_criterion.evaluate(fvals=loss.detach())

        # clamp values to the feasible set
        X.data = columnwise_clamp(X, bounds[0], bounds[1])
        # for j, (lb, ub) in enumerate(zip(*bounds)):
        #     X.data[..., j].clamp_(lb, ub)  # need to do this on the data not X itself

        if return_traj:
            # store the optimization trajectory
            X_traj.append(X[:, :q].detach().clone())

        if return_loss:
            losses.append(loss.detach().item())

        if i % 10 == 0 and verbose > 0:
            print(f"Iteration {i + 1:>3}/{num_opt_steps:<3d} - Loss: {loss.item():>5.5f}")

        # if stop:
        #     continue
        # break

    # return only best among num_starts candidates
    with torch.no_grad():
        best_ind = torch.argmax(acq_func(X)).item()
    X = X[:, :q]  # n_starts x q x d

    meta_dic = {}
    if return_traj:
        meta_dic['traj'] = torch.stack(X_traj)[:, best_ind]
    if return_loss:
        meta_dic['loss'] = losses
    # loss obtained for the selected point at the last optimization step
    meta_dic['info'] = dict(last_loss=losses_step[best_ind].detach().item())

    return X[best_ind], meta_dic


def generate_initial_data(n: int, bounds: Tensor, obj_func: SyntheticTestFunction) -> Tuple[Tensor, Tensor]:
    train_X = unnormalize(torch.rand((n, bounds.shape[1])), bounds)
    train_Y = obj_func(train_X).reshape(-1, 1)
    return train_X, train_Y


def update_best(best_candidates: List[Tensor], best_values: List[float], new_candidates: Tensor,
                new_values: Tensor) -> None:
    assert 0 < len(best_candidates) == len(best_values)
    assert new_candidates.ndim == (new_values.ndim + 1)
    best_new_candidate, best_new_value = new_candidates[torch.argmax(new_values)].clone(), torch.max(new_values).item()
    best_values.append(max(best_values[-1], best_new_value))
    best_candidates.append(best_candidates[-1] if best_values[-1] == best_values[-2] else best_new_candidate)


def one_acq_step(X: Union[Tensor, np.ndarray], Y: [Tensor, np.ndarray],
                 np_bounds: np.ndarray,
                 do_normalize: bool, do_standardize: bool,
                 sampler: MCSampler,
                 q: int,
                 AcqFunc: Type[AcquisitionFunction],
                 acqf_opt_config: Dict[str, Any],
                 acq_func_kwargs: Dict[str, Any]) -> np.ndarray:
    dim: int = X.shape[1]
    bounds = torch.tensor(np_bounds).double()
    train_X, train_Y = torch.tensor(X), torch.tensor(Y)  # train_Y.ndim == 2
    outer_dim = train_Y.shape[1]  # typically 1 if real-valued objective function

    meta_dic = {}
    best_candidates, best_values = [train_X[torch.argmax(train_Y)].clone()], [torch.max(train_Y).item()]

    # prepare data
    train_X_it = normalize(train_X, bounds) if do_normalize else train_X
    train_Y_it = train_Y

    if acq_func_kwargs is None:
        acq_func_kwargs: Dict[str, Any] = {}
    num_candidates: Optional[int] = acq_func_kwargs.pop("num_candidates", None)
    previous_entropy, current_entropy = None, None  # for Entropy Search

    acq_bounds = norm_bounds(dim) if do_normalize else bounds

    begin_it_t: float = time.time()

    # fit surrogate model given the data
    model = SingleTaskGP(train_X_it, train_Y_it,
                         outcome_transform=Standardize(outer_dim) if do_standardize else None)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    if num_candidates:  # should be the case when using qCompositionalEntropySearch
        set_candidates = normalize(torch.rand(num_candidates, acq_bounds.shape[1], dtype=train_X.dtype),
                                   acq_bounds).requires_grad_(False)
        acq_func_kwargs["candidate_set"] = set_candidates
    if issubclass(AcqFunc, qUpperConfidenceBound):
        acq_func_kwargs["sampler"] = sampler
    if issubclass(AcqFunc, qExpectedImprovement) or issubclass(AcqFunc, qProbabilityOfImprovement):
        acq_func_kwargs["best_f"] = best_values[-1]
        acq_func_kwargs["sampler"] = sampler
    acq_func = AcqFunc(model, **acq_func_kwargs)

    new_X, meta_dic = optimize_acqf_and_get_observation(acq_func, q=q, bounds=acq_bounds,
                                                        return_traj=False, **acqf_opt_config)
    execution_time = time.time() - begin_it_t
    meta_dic['execution_time'] = execution_time

    new_X = new_X.detach()  # shape q_q_fantasies x dim
    if do_normalize:
        new_X = unnormalize(new_X, bounds)
    return new_X.detach().cpu().numpy()


def save_exp(config: argparse.Namespace, best_values_s: np.ndarray, train_X_s: np.ndarray, train_Y_s: np.ndarray,
             execution_times_s: Optional[List[List[float]]] = None,
             losses_s: Optional[List[List[List[float]]]] = None, X_trajs_s: Optional[List[List[np.ndarray]]] = None):
    print(f"Saving results in: {config.result_dir}")

    result_dir = config.result_dir
    save_exp_info(config.exp_title, result_dir)
    save_config(config)

    # write result values
    save_np(best_values_s, result_dir, "best_values")
    if not hasattr(config, 'time_only'):
        save_np(train_X_s, result_dir, "train_X")
        save_np(train_Y_s, result_dir, "train_Y")
    save_execution_times(execution_times_s, result_dir)

    if config.save_losses:
        utils_save.save_losses(losses_s, result_dir)
    if config.save_trajs:
        save_trajs(X_trajs_s, result_dir)


def run_exp_bo(config):
    start_time = time.time()

    # goal is to MAXIMIZE config.negate * test_func
    if not hasattr(config, 'lb_offset'):
        config.lb_offset = 0
    if not hasattr(config, 'ub_offset'):
        config.ub_offset = 0
    if not hasattr(config, 'cuda'):
        config.cuda = None
    if not hasattr(config, 'noise_free'):
        config.noise_free = False
    if not hasattr(config, 'covar'):
        config.covar = 'matern-5/2'
        config.covar_kw = {}
    config.covar = config.covar.lower()
    tkwargs = {
        "dtype": torch.double,
        "device": torch.device(
            f"cuda:{config.cuda}" if torch.cuda.is_available() and config.cuda is not None else "cpu"),
    }
    if torch.cuda.is_available() and config.cuda is not None:
        torch.cuda.set_device(tkwargs['device'])

    testfunc: SyntheticTestFunction = query_test_func(config.test_func, config.input_dim, config.negate,
                                                      config.lb_offset, config.ub_offset).to(**tkwargs)
    config.optimal_value = testfunc.optimal_value
    config.input_dim = testfunc.dim

    bounds: Tensor = testfunc.bounds  # shape (2, d)

    config.exp_title = f"{'Min' if testfunc.negate else 'Max'}imize {testfunc._get_name()} (dimension: {testfunc.dim:d}) using {config.optimizer} to optimize acquisition function {config.acq_func}"

    print("\n``` " + config.exp_title + " ```\n")

    search_bounds = [
        {'name': f'x_{i}', 'type': 'continuous', 'domain': (bounds[0, i].item(), bounds[1, i].item())} for i in
        range(config.input_dim)
    ]
    search_real_transfo = [
        {f'x_{i}': lambda w: w} for i in range(config.input_dim)
    ]
    params_h: ParamSpace = ParamSpace(bounds=search_bounds, transfos=search_real_transfo)

    # run bayesian optimization routine
    train_X_s: List[np.ndarray] = []
    train_Y_s: List[np.ndarray] = []
    best_values_s: List[np.ndarray] = []

    execution_times_s: List[List[float]] = []

    config.save_losses &= config.save  # don't save results if don't save experiment
    config.save_trajs &= config.save  # don't save trajs if don't save experiment
    losses_s, X_trajs_s = None, None
    if config.save_trajs:
        X_trajs_s: List[List[np.ndarray]] = []
    if config.save_losses:
        losses_s: List[List[List[float]]] = []

    # seed for reproducibility
    seed = config.seed

    if not hasattr(config, 'early_stop'):
        config.early_stop = False

    if config.save:
        result_dir = create_exp_folder(config, os.getcwd())
        config.result_dir = result_dir
        if hasattr(config, 'ignore_existing') and not config.ignore_existing:
            exp_dir = os.path.dirname(config.result_dir)
            exps = query_exps(exp_dir, config)
            if len(exps) > 0:
                print(f'Experiments with similar configurations exist in {exp_dir}')
                best_values: np.ndarray = load_np(exps[0].result_dir, 'best_values')
                return config.optimal_value - best_values.mean(0)[-1]

    for trial in range(config.num_trials):
        print(f'Trial {trial + 1:>3d}/{config.num_trials:<3d}')
        best_values: List[float] = []

        bo: BayesOptimization = BayesOptimization(
            params_h=params_h,
            negate=False,  # we already negate test function if needed
            noise_free=config.noise_free,
            covar=config.covar,
            covar_kw=config.covar_kw,
            optimizer=config.optimizer,
            acq_func=config.acq_func,
            scheduler=config.scheduler,
            optimizer_kwargs=config.optimizer_kwargs,
            acq_func_kwargs=config.acq_func_kwargs,
            initial_design_numdata=config.num_initial,
            num_MC_samples_acq=config.num_MC_samples_acq,
            num_raw_samples=config.num_raw_samples,
            num_starts=config.num_starts,
            num_opt_steps=config.num_opt_steps,
            scheduler_kwargs=config.scheduler_kwargs,
            verbose=config.verbose - 1,
            seed=seed,
            early_stop=config.early_stop,
            device=config.cuda
        )

        # initial random points:
        Xs: List[Dict[str, float]] = bo.gen(config.num_initial)
        X_to_eval: Tensor = torch.tensor(np.array([params_h.get_list_params_from_dict(X) for X in Xs])).to(**tkwargs)
        bo.observe(Xs, testfunc(X_to_eval))
        bo.num_acq_steps = 0
        best_values.append(bo.data_Y.max().item())

        for acq_step in range(config.num_acq_steps):
            Xs: List[Dict[str, float]] = bo.gen(config.q)
            X_to_eval: Tensor = torch.tensor(np.array([params_h.get_list_params_from_dict(X) for X in Xs]))
            new_values = testfunc(X_to_eval)
            bo.observe(Xs, new_values)
            best_values.append(max(best_values[-1], new_values.max().item()))

        train_X: np.ndarray = bo.data_X.detach().cpu().numpy()
        train_Y: Tensor = bo.data_Y.detach().cpu().numpy()
        best_values_s.append(np.array(best_values))

        if config.save:
            train_X_s.append(train_X)
            train_Y_s.append(train_Y)
            execution_times_s.append(bo.execution_times_s[1:])
        if config.save_losses:
            losses_s.append(np.array(bo.losses))
        seed += 1

    best_values = np.array(best_values_s).reshape(config.num_trials, -1)

    if config.save:
        train_X_s: np.ndarray = np.array(train_X_s).reshape((config.num_trials, -1, config.input_dim))
        train_Y_s: np.ndarray = np.array(train_Y_s).reshape(config.num_trials, -1)

        save_exp(config, best_values, train_X_s=train_X_s, train_Y_s=train_Y_s, execution_times_s=execution_times_s,
                 losses_s=losses_s, X_trajs_s=X_trajs_s)

    if config.verbose > 0:
        print(f"Experiment took {time.time() - start_time:.1f} s")
    return config.optimal_value - best_values.mean(0)[-1]


if __name__ == '__main__':
    config_ = get_exp_config()

    run_exp_bo(config_)
