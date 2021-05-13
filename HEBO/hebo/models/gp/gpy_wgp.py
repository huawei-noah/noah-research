"""GPy Warped GP."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.


import warnings
import GPy
import mindspore as ms
import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

import hebo.mindspore as hebo_ms
from ..base_model import BaseModel
from ..layers import OneHotTransform
from ..scalers import MindSporeMinMaxScaler, MindSporeStandardScaler
from ..util import filter_nan

import logging

logging.disable(logging.WARNING)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class GPyGP(BaseModel):
    """Input warped GP model implemented using GPy.

    Why doing so:
    - Input warped GP
    """
    
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        if num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])
        self.xscaler = MindSporeMinMaxScaler((-1, 1))
        self.yscaler = MindSporeStandardScaler()
        self.verbose = self.conf.get('verbose', False)
        self.num_epochs = self.conf.get('num_epochs', 200)
        self.warp = self.conf.get('warp', True)
        self.space = self.conf.get('space')  # DesignSpace
        if self.space is None and self.warp:
            warnings.warn('Space not provided, set warp to False')
            self.warp = False
    
    def fit_scaler(self, Xc: Tensor, y: Tensor):
        """fit_scaler."""
        if Xc is not None and Xc.shape[1] > 0:
            if self.space is not None:
                # NOTE: we must know the strict lower and upper bound to use
                # input warping
                cont_lb = self.space.opt_lb[:self.space.num_numeric].view(
                    1, -1)
                cont_ub = self.space.opt_ub[:self.space.num_numeric].view(
                    1, -1)
                self.xscaler.fit(ms.ops.Concat(axis=0)([Xc, cont_lb, cont_ub]))
            else:
                self.xscaler.fit(Xc)
        self.yscaler.fit(y)
    
    def trans(self, Xc: Tensor, Xe: Tensor, y: Tensor = None):
        """trans."""
        if Xc is not None and Xc.shape[1] > 0:
            Xc_t = self.xscaler.transform(Xc)
        else:
            Xc_t = hebo_ms.zeros((Xe.shape[0], 0))
        
        if Xe is None or Xe.shape[1] == 0:
            Xe_t = hebo_ms.zeros((Xc.shape[0], 0))
        else:
            Xe_t = self.one_hot(Xe.astype(ms.int32))
        
        Xall = np.hstack([Xc_t.asnumpy(), Xe_t.asnumpy()])
        
        if y is not None:
            y_t = self.yscaler.transform(y).asnumpy()
            return Xall, y_t
        return Xall
    
    def fit(self, Xc: Tensor, Xe: Tensor, y: Tensor):
        """fit."""
        Xc, Xe, y = filter_nan(Xc, Xe, y, 'all')
        self.fit_scaler(Xc, y)
        X, y = self.trans(Xc, Xe, y)
        
        k1 = GPy.kern.Linear(X.shape[1], ARD=False)
        k2 = GPy.kern.Matern32(X.shape[1], ARD=True)
        k2.lengthscale = np.std(X, axis=0).clip(min=0.02)
        k2.variance = 0.5
        k2.variance.set_prior(GPy.priors.Gamma(0.5, 1), warning=False)
        kern = k1 + k2
        if not self.warp:
            self.gp = GPy.models.GPRegression(X, y, kern)
        else:
            xmin = np.zeros(X.shape[1])
            xmax = np.ones(X.shape[1])
            xmin[:Xc.shape[1]] = -1
            warp_f = GPy.util.input_warping_functions.KumarWarping(
                X, Xmin=xmin, Xmax=xmax)
            self.gp = GPy.models.InputWarpedGP(
                X, y, kern, warping_function=warp_f)
        self.gp.likelihood.variance.set_prior(
            GPy.priors.LogGaussian(-4.63, 0.5), warning=False)
        
        self.gp.optimize_restarts(
            max_iters=self.num_epochs,
            verbose=self.verbose,
            num_restarts=10,
            robust=True)
        return self
    
    def predict(self, Xc: Tensor, Xe: Tensor) -> (Tensor, Tensor):
        """predict."""
        Xall = self.trans(Xc, Xe)
        py, ps2 = self.gp.predict(Xall)
        mu = self.yscaler.inverse_transform(Tensor(py).view(-1, 1))
        var = ms.ops.clip_by_value(
            self.yscaler.std ** 2 * Tensor(ps2).view(-1, 1), 1e-6, np.inf)
        return mu, var
    
    def sample_f(self):
        """sample_f."""
        raise NotImplementedError(
            'Thompson sampling is not supported for GP, use `sample_y` instead')
    
    @property
    def noise(self):
        """noise."""
        var_normalized = Tensor(self.gp.likelihood.variance[0], ms.float32)
        noise = (var_normalized * self.yscaler.std ** 2).view((self.num_out,))
        return noise
