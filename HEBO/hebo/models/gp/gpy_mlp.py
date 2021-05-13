"""GPy MLP GP."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import GPy
import hebo.mindspore as hebo_ms
import mindspore as ms
import numpy as np
from mindspore import Tensor

from ..base_model import BaseModel
from ..layers import OneHotTransform
from ..scalers import MindSporeMinMaxScaler, MindSporeStandardScaler
from ..util import filter_nan


class GPyMLPGP(BaseModel):
    """GPyMLPGP."""
    
    def __init__(self, num_cont, num_enum, num_out, **conf):
        super().__init__(num_cont, num_enum, num_out, **conf)
        if num_enum > 0:
            self.one_hot = OneHotTransform(self.conf['num_uniqs'])
        self.xscaler = MindSporeMinMaxScaler((-1, 1))
        self.yscaler = MindSporeStandardScaler()
        self.verbose = self.conf.get('verbose', False)
        self.num_epochs = self.conf.get('num_epochs', 200)
    
    def fit_scaler(self, Xc: Tensor, y: Tensor):
        """fit_scaler."""
        if Xc is not None and Xc.shape[1] > 0:
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
        
        kern = GPy.kern.src.mlp.MLP(input_dim=X.shape[1], ARD=True)
        self.gp = GPy.models.GPRegression(X, y, kern)
        self.gp.kern.variance = np.var(y)
        self.gp.kern.lengthscale = np.std(X, axis=0).clip(min=0.02)
        self.gp.likelihood.variance = 1e-2 * np.var(y)
        
        self.gp.kern.variance.set_prior(GPy.priors.Gamma(0.5, 0.5))
        self.gp.likelihood.variance.set_prior(
            GPy.priors.LogGaussian(-4.63, 0.5))
        
        self.gp.optimize_restarts(
            max_iters=self.num_epochs,
            messages=self.verbose,
            num_restarts=10,
            robust=True)
        print(self.gp.likelihood.variance, flush=True)
        print(self.gp.likelihood.variance[0], flush=True)
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
        return (var_normalized * self.yscaler.std ** 2).view((self.num_out,))
