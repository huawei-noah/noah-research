"""Evolutionary optimiser for acq functions."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from hebo.acquisitions.acq import Acquisition
from hebo.design_space.design_space import DesignSpace
import hebo.mindspore as hebo_ms
from pymoo.factory import get_algorithm
import numpy as np
import pandas as pd
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor
from pyDOE2 import lhs
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem, get_mutation, get_crossover
from pymoo.operators.mixed_variable_operator import MixedVariableMutation, MixedVariableCrossover
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.configuration import Configuration

Configuration.show_compile_hint = False


class BOProblem(Problem):
    """BO problem specification."""
    
    def __init__(self,
                 lb: np.ndarray,
                 ub: np.ndarray,
                 acq: Acquisition,
                 space: DesignSpace,
                 fix: dict = None
                 ):
        super().__init__(len(lb), xl=lb, xu=ub, n_obj=acq.num_obj, n_constr=acq.num_constr)
        self.acq = acq
        self.space = space
        self.fix = fix  # NOTE: use self.fix to enable contextual BO
    
    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        """Evaluate acq value of input."""
        num_x = x.shape[0]
        xcont = Tensor(x[:, :self.space.num_numeric].astype(
            float)) if self.space.num_numeric > 0 else mnp.zeros((num_x, 0))
        xenum = Tensor(x[:, self.space.num_numeric:].astype(
            int)) if self.space.num_categorical > 0 else mnp.zeros((num_x, 0))
        df_x = self.space.inverse_transform(xcont, xenum)
        if self.fix is not None:  # invalidate fixed input, replace with fixed values
            for k, v in self.fix.items():
                df_x[k] = v
        xcont, xenum = self.space.transform(df_x)
        
        acq_eval = self.acq(
            xcont,
            xenum).asnumpy().reshape(num_x, self.acq.num_obj + self.acq.num_constr)
        out['F'] = acq_eval[:, :self.acq.num_obj]
        
        if self.acq.num_constr > 0:
            out['G'] = acq_eval[:, -1 * self.acq.num_constr:]


class EvolutionOpt:
    """Evolutionary Optimiser."""
    
    def __init__(self,
                 design_space: DesignSpace,
                 acq: Acquisition,
                 es: str = None,
                 **conf):
        self.space = design_space
        self.es = es
        self.acq = acq
        self.pop = conf.get('pop', 100)
        self.iter = conf.get('iters', 500)
        self.verbose = conf.get('verbose', False)
        self.repair = conf.get('repair', None)
        self.lhs_init = conf.get('lhs_init', True)
        assert (self.acq.num_obj > 0)
        
        if self.es is None:
            self.es = 'nsga2' if self.acq.num_obj > 1 else 'ga'
    
    def get_init_pop(self, initial_suggest: pd.DataFrame = None) -> np.ndarray:
        """Return init pop."""
        if not self.lhs_init:
            init_pop = self.space.sample(self.pop)
        else:
            lhs_samp = lhs(self.space.num_paras, self.pop)
            lhs_samp = lhs_samp * \
                       (self.space.opt_ub - self.space.opt_lb).asnumpy() + self.space.opt_lb.asnumpy()
            x = lhs_samp[:, :self.space.num_numeric]
            xe = lhs_samp[:, self.space.num_numeric:].astype(int)
            for i, n in enumerate(self.space.numeric_names):
                if self.space.paras[n].is_discrete_after_transform:
                    x[:, i] = x[:, i].round()
            x = Tensor(x) if x.shape[1] > 0 else mnp.zeros((x.shape[0], 0))
            xe = Tensor(xe) if xe.shape[1] > 0 else mnp.zeros((xe.shape[0], 0))
            init_pop = self.space.inverse_transform(Tensor(x), Tensor(xe))
        if initial_suggest is not None:
            init_pop = pd.concat(
                [initial_suggest, init_pop], axis=0).head(self.pop)
        x, xe = self.space.transform(init_pop)
        return np.hstack([x.asnumpy(), xe.asnumpy().astype(float)])
    
    def get_mutation(self):
        """Return mutation."""
        mask = []
        for name in (self.space.numeric_names + self.space.enum_names):
            if self.space.paras[name].is_discrete_after_transform:
                mask.append('int')
            else:
                mask.append('real')
        
        mutation = MixedVariableMutation(mask, {
            'real': get_mutation('real_pm', eta=20),
            'int': get_mutation('int_pm', eta=20)
        })
        return mutation
    
    def get_crossover(self):
        """Return crossover."""
        mask = []
        for name in (self.space.numeric_names + self.space.enum_names):
            if self.space.paras[name].is_discrete_after_transform:
                mask.append('int')
            else:
                mask.append('real')
        
        crossover = MixedVariableCrossover(mask, {
            'real': get_crossover('real_sbx', eta=15, prob=0.9),
            'int': get_crossover('int_sbx', eta=15, prob=0.9)
        })
        return crossover
    
    def optimize(self, initial_suggest: pd.DataFrame = None,
                 fix_input: dict = None) -> pd.DataFrame:
        """Maximise acq functions."""
        lb = self.space.opt_lb.asnumpy()
        ub = self.space.opt_ub.asnumpy()
        prob = BOProblem(lb, ub, self.acq, self.space, fix_input)
        init_pop = self.get_init_pop(initial_suggest)
        mutation = self.get_mutation()
        crossover = self.get_crossover()
        algo = get_algorithm(
            self.es,
            pop_size=self.pop,
            sampling=init_pop,
            mutation=mutation,
            crossover=crossover,
            repair=self.repair)
        res = minimize(prob, algo, ('n_gen', self.iter), verbose=self.verbose)
        if res.X is not None:
            opt_x = res.X.reshape(-1, len(lb)).astype(float)
        else:
            opt_x = np.array([p.X for p in res.pop]).astype(float)
            if self.acq.num_obj == 1:
                opt_x = opt_x[[np.random.choice(opt_x.shape[0])]]
        
        opt_xcont = hebo_ms.from_numpy(
            opt_x[:, :self.space.num_numeric].astype(float))
        opt_xenum = hebo_ms.from_numpy(
            opt_x[:, self.space.num_numeric:].astype(int))
        df_opt = self.space.inverse_transform(opt_xcont, opt_xenum)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_opt[k] = v
        return df_opt
