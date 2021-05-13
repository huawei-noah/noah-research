"""HEBO Optimizer."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import numpy as np
import pandas as pd
import mindspore as ms
from mindspore import Tensor
from pyDOE2 import lhs
from sklearn.preprocessing import power_transform

from hebo.design_space.design_space import DesignSpace
from hebo.models.model_factory import get_model
from hebo.acquisitions.acq import MACE, Mean, Sigma
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
import hebo.mindspore as hebo_ms

from .abstract_optimizer import AbstractOptimizer


class HEBO(AbstractOptimizer):
    """HEBO Optimizer class."""
    
    support_parallel_opt = True
    support_combinatorial = True
    support_contextual = True
    
    def __init__(
            self,
            space,
            model_name='gpy',
            rand_sample=None,
            acq_cls=MACE,
            es='nsga2',
            verbose=False):
        """Init.

        model_name : surrogate model to be used
        rand_iter  : iterations to perform random sampling
        """
        super().__init__(space)
        self.space = space
        self.es = es
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.model_name = model_name
        self.rand_sample = 1 + \
                           self.space.num_paras if rand_sample is None else max(2, rand_sample)
        self.acq_cls = acq_cls
        self.verbose = verbose
    
    def quasi_sample(self, n, fix_input=None) -> pd.DataFrame:
        """Quasi-random sampling function."""
        samp = lhs(self.space.num_paras, n)
        samp = samp * (self.space.opt_ub - self.space.opt_lb).asnumpy() + self.space.opt_lb.asnumpy()
        x = samp[:, :self.space.num_numeric]
        xe = samp[:, self.space.num_numeric:]
        for i, n in enumerate(self.space.numeric_names):
            if self.space.paras[n].is_discrete_after_transform:
                x[:, i] = x[:, i].round()
        x = hebo_ms.from_numpy(x)
        xe = hebo_ms.from_numpy(xe)
        df_samp = self.space.inverse_transform(x, xe)
        if fix_input is not None:
            for k, v in fix_input.items():
                df_samp[k] = v
        return df_samp
    
    @property
    def model_config(self):
        """Add additional arguments for surrogate model."""
        if self.model_name == 'gpy':
            cfg = {
                'verbose': self.verbose,
                'warp': True,
                'space': self.space
            }
        elif self.model_name == 'gpy_mlp':
            cfg = {
                'verbose': self.verbose
            }
        elif self.model_name == 'rf':
            cfg = {
                'n_estimators': 20
            }
        else:
            cfg = {}
        if self.space.num_categorical > 0:
            cfg['num_uniqs'] = [len(self.space.paras[name].categories)
                                for name in self.space.enum_names]
        return cfg
    
    def suggest(self, n_suggestions=1, fix_input=None):
        """Suggest params."""
        if self.X.shape[0] < self.rand_sample:
            sample = self.quasi_sample(n_suggestions, fix_input)
            return sample
        else:
            X, Xe = self.space.transform(self.X)
            try:
                if self.y.min() <= 0:
                    y = Tensor(
                        power_transform(
                            self.y / self.y.std(),
                            method='yeo-johnson'))
                else:
                    y = Tensor(
                        power_transform(
                            self.y / self.y.std(),
                            method='box-cox'))
                    if y.asnumpy().std() < 0.5:
                        y = Tensor(
                            power_transform(
                                self.y / self.y.std(),
                                method='yeo-johnson'))
                if y.asnumpy().std() < 0.5:
                    raise RuntimeError('Power transformation failed')
                model = get_model(
                    self.model_name,
                    self.space.num_numeric,
                    self.space.num_categorical,
                    1,
                    **self.model_config)
                model.fit(X, Xe, y)
            except BaseException:
                y = Tensor(self.y.copy())
                model = get_model(
                    self.model_name,
                    self.space.num_numeric,
                    self.space.num_categorical,
                    1,
                    **self.model_config)
                model.fit(X, Xe, y)
            
            best_id = np.argmin(self.y.squeeze())
            best_x = self.X.iloc[[best_id]]
            _ = self.y.min()
            py_best, ps2_best = model.predict(*self.space.transform(best_x))
            py_best = py_best.asnumpy().squeeze()
            ps2_best = ps2_best.asnumpy().squeeze()
            _ = np.sqrt(ps2_best)
            
            iter = max(1, self.X.shape[0] // n_suggestions)
            upsi = 0.5
            delta = 0.01
            # kappa = np.sqrt(upsi * 2 * np.log(iter **  (2.0 + self.X.shape[1] / 2.0) * 3 * np.pi**2 / (3 * delta)))
            kappa_sq = ((2.0 + self.X.shape[1] / 2.0) * np.log(iter) + np.log(3 * np.pi ** 2 / (3 * delta)))
            kappa = np.sqrt(upsi * 2 * kappa_sq)
            acq = self.acq_cls(model, py_best, kappa=kappa)  # LCB < py_best
            assert acq.num_obj > 1
            mu = Mean(model)
            sig = Sigma(model, linear_a=-1.)
            opt = EvolutionOpt(
                self.space,
                acq,
                pop=100,
                iters=100,
                verbose=self.verbose,
                es=self.es)
            rec = opt.optimize(
                initial_suggest=best_x,
                fix_input=fix_input).drop_duplicates()
            rec = rec[self.check_unique(rec)]
            
            cnt = 0
            while rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(
                    n_suggestions - rec.shape[0], fix_input)
                rand_rec = rand_rec[self.check_unique(rand_rec)]
                rec = rec.append(rand_rec, ignore_index=True)
                cnt += 1
                if cnt > 3:
                    # sometimes the design space is so small that duplicated
                    # sampling is unavoidable
                    break
            if rec.shape[0] < n_suggestions:
                rand_rec = self.quasi_sample(
                    n_suggestions - rec.shape[0], fix_input)
                rec = rec.append(rand_rec, ignore_index=True)
            
            select_id = np.random.choice(
                rec.shape[0], n_suggestions, replace=False).tolist()
            
            py_all = mu(*self.space.transform(rec)).squeeze().asnumpy()
            ps_all = -1 * sig(*self.space.transform(rec)).squeeze().asnumpy()
            best_pred_id = np.argmin(py_all)
            best_unce_id = np.argmax(ps_all)
            if best_unce_id not in select_id and n_suggestions > 2:
                select_id[0] = best_unce_id
            if best_pred_id not in select_id and n_suggestions > 2:
                select_id[1] = best_pred_id
            if n_suggestions == 1:
                select_id = np.random.choice([select_id[0], best_pred_id, best_unce_id], 1, p=[
                    6 / 8, 1 / 8, 1 / 8]).tolist()
            rec_selected = rec.iloc[select_id].copy()
            return rec_selected
    
    def check_unique(self, rec: pd.DataFrame) -> [bool]:
        """Check if parameter sets are unique."""
        return (~pd.concat([self.X, rec], axis=0).duplicated().tail(
            rec.shape[0]).values).tolist()
    
    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : pandas DataFrame
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,1)
            Corresponding values where objective has been evaluated
        """
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id]
        yy = y[valid_id].reshape(-1, 1)
        self.X = self.X.append(XX, ignore_index=True)
        self.y = np.vstack([self.y, yy])
