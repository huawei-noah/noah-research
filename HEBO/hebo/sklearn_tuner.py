"""Sklearn Tuner API for easy integration with sklearn models and cv."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from sklearn.model_selection import cross_val_predict

from pymoo.model.problem import Problem
from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.algorithms.nsga2 import NSGA2

warnings.filterwarnings('ignore')

def minimise_me(
        eqc,
        neqc,
        lower_bounds,
        upper_bounds,
):
    sampling = get_sampling('real_lhs')
    crossover = get_crossover('real_sbx', eta=15, prob=0.9)
    mutation = get_mutation('real_pm', eta=20)

    n_var = len(lower_bounds)
    n_obj = len(eqc)
    n_constr = len(neqc)
    
    class MultiObjMultiConstProblem(Problem):
        def __init__(self, n_var=0, n_obj=0, n_constr=0, xl=[], xu=[]):
            super().__init__(n_var=n_var,
                             n_obj=n_obj,
                             n_constr=n_constr,
                             xl=xl,
                             xu=xu)

        def _evaluate(self, X, out, *args, **kwargs):
            # Here we have a GP for each constraint.
            # If one constraint is binary, see below.
            
            list_g = []
            for f_cont in neqc:
                list_g.append(f_cont(X))
            out["G"] = np.column_stack(list_g)

            list_f = []
            for f_func in eqc:
                list_f.append(f_func(X))

            out["F"] = np.column_stack(list_f)
            
    problem = MultiObjMultiConstProblem(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=lower_bounds, xu=upper_bounds)
    
    obj = NSGA2(
        pop_size=100,  # change start children
        n_offsprings=100,  # change the number of end children
        sampling=sampling,  # TODO: can replace with get_init_pop(pop) once made
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True
    )

    # perform a copy of the algorithm to ensure reproducibility
    N_iterations = 1000
    termination = get_termination("n_gen", N_iterations)
    # let the algorithm know what problem we are intending to solve and provide other attributes

    # obj.setup(problem, termination=termination, seed=1)
    obj.setup(problem, termination=termination, seed=0)
    i = 0
    while obj.has_next():
        i += 1
        # perform an iteration of the algorithm
        obj.next()
    
    result = obj.result()
    return result

def sklearn_tuner(
        model_class,
        space_config: [dict],
        X: np.ndarray,
        y: np.ndarray,
        metric: Callable,
        greater_is_better: bool = True,
        cv=None,
        max_iter=16,
        report=False,
        hebo_cfg=None,
) -> (dict, pd.DataFrame):
    """Tuning sklearn estimator.

    Parameters:
    -------------------
    model_class: class of sklearn estimator
    space_config: list of dict, specifying search space
    X, y: data used to for cross-valiation
    metrics: metric function in sklearn.metrics
    greater_is_better: whether a larger metric value is better
    cv: the 'cv' parameter in `cross_val_predict`
    max_iter: number of trials

    Returns:
    -------------------
    Best hyper-parameters and all visited data


    Example:
    -------------------
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from hebo.sklearn_tuner import sklearn_tuner

    space_cfg = [
            {'name' : 'max_depth',        'type' : 'int', 'lb' : 1, 'ub' : 20},
            {'name' : 'min_samples_leaf', 'type' : 'num', 'lb' : 1e-4, 'ub' : 0.5},
            {'name' : 'max_features',     'type' : 'cat', 'categories' : ['auto', 'sqrt', 'log2']},
            {'name' : 'bootstrap',        'type' : 'bool'},
            {'name' : 'min_impurity_decrease', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1.0},
            ]
    X, y   = load_boston(return_X_y = True)
    result = sklearn_tuner(RandomForestRegressor, space_cfg, X, y, metric = r2_score, max_iter = 16)
    """
    if hebo_cfg is None:
        hebo_cfg = {}
    space = DesignSpace().parse(space_config)
    opt = HEBO(space, **hebo_cfg)
    for i in range(max_iter):
        rec = opt.suggest()
        model = model_class(**rec.iloc[0].to_dict())
        pred = cross_val_predict(model, X, y, cv=cv)
        score_v = metric(y, pred)
        sign = -1. if greater_is_better else 1.0
        opt.observe(rec, np.array([sign * score_v]))
        print('Iter %d, best metric: %g' % (i, sign * opt.y.min()))
    best_id = np.argmin(opt.y.reshape(-1))
    best_hyp = opt.X.iloc[best_id]
    df_report = opt.X.copy()
    df_report['metric'] = sign * opt.y
    if report:
        return best_hyp.to_dict(), df_report
    else:
        return best_hyp.to_dict()
