"""Acquisition function library."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from abc import ABC, abstractmethod

import hebo.mindspore as hebo_ms
import numpy as np
from mindspore import Tensor
from scipy.stats import norm

from ..models.base_model import BaseModel


class Acquisition(ABC):
    """Acquisition base class."""
    
    def __init__(self, model, **conf):
        self.model = model
    
    @property
    @abstractmethod
    def num_obj(self):
        """num_obj."""
        pass
    
    @property
    @abstractmethod
    def num_constr(self):
        """num_constr."""
        pass
    
    @abstractmethod
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """Shape of output tensor: (x.shape[0], self.num_obj + self.num_constr)."""
        pass
    
    def __call__(self, x: Tensor, xe: Tensor):
        """__call__ on continuous and integer tensors."""
        return self.eval(x, xe)


class SingleObjectiveAcq(Acquisition):
    """Single-objective, unconstrained acquisition."""
    
    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)
    
    @property
    def num_obj(self):
        """num_obj."""
        return 1
    
    @property
    def num_constr(self):
        """num_constr."""
        return 0


class LCB(SingleObjectiveAcq):
    """Lower confidence bound acq."""
    
    def __init__(self, model: BaseModel, kappa=3.0, **conf):
        super().__init__(model, **conf)
        self.kappa = kappa
        assert (model.num_out == 1)
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """eval."""
        py, ps2 = self.model.predict(x, xe)
        return py - self.kappa * hebo_ms.sqrt(ps2)


class Mean(SingleObjectiveAcq):
    """Mean acquisition."""
    
    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)
        assert (model.num_out == 1)
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """eval."""
        py, _ = self.model.predict(x, xe)
        return py


class Sigma(SingleObjectiveAcq):
    """Sigma acq."""
    
    def __init__(self, model: BaseModel, **conf):
        super().__init__(model, **conf)
        assert (model.num_out == 1)
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """eval."""
        _, ps2 = self.model.predict(x, xe)
        return -1 * hebo_ms.sqrt(ps2)


class EI(SingleObjectiveAcq):
    """EI."""
    
    pass


class logEI(SingleObjectiveAcq):
    """logEI."""
    
    pass


class WEI(Acquisition):
    """EI."""
    
    pass


class Log_WEI(Acquisition):
    """Log_WEI."""
    
    pass


class MES(SingleObjectiveAcq):
    """MES."""
    
    pass


class MOMeanSigmaLCB(Acquisition):
    """MOMeanSigmaLCB."""
    
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.best_y = best_y
        self.kappa = conf.get('kappa', 2.0)
        assert (self.model.num_out == 1)
    
    @property
    def num_obj(self):
        """num_obj."""
        return 2
    
    @property
    def num_constr(self):
        """num_constr."""
        return 1
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """Eval.

        minimize (py, -1 * ps)
        s.t.     LCB  < best_y
        """
        out = np.zeros((x.shape[0], self.num_obj + self.num_constr))
        py, ps2 = self.model.predict(x, xe)
        py = py.asnumpy()
        ps2 = ps2.asnumpy()
        ps = np.sqrt(ps2)
        noise = np.sqrt(self.model.noise.asnumpy())
        py += noise * np.random.randn(*py.shape)
        lcb = py - self.kappa * ps
        out[:, 0] = py.squeeze()
        out[:, 1] = -1 * ps.squeeze()
        out[:, 2] = lcb.squeeze() - self.best_y  # lcb - best_y < 0
        return Tensor(out)


class MACE(Acquisition):
    """MACE."""
    
    def __init__(self, model, best_y, **conf):
        super().__init__(model, **conf)
        self.kappa = conf.get('kappa', 2.0)
        self.eps = conf.get('eps', 1e-4)
        self.tau = best_y
        self.dist = norm()
    
    @property
    def num_constr(self):
        """num_constr."""
        return 0
    
    @property
    def num_obj(self):
        """num_obj."""
        return 3
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        """eval.

        minimize (-1 * EI,  -1 * PI, lcb).
        """
        py, ps2 = self.model.predict(x, xe)
        py = py.asnumpy()
        ps2 = ps2.asnumpy()
        noise = np.sqrt(2.0 * self.model.noise.asnumpy())
        ps = np.sqrt(ps2)
        lcb = (py + noise * np.random.randn(*py.shape)) - self.kappa * ps
        
        normed = ((self.tau - self.eps - py - noise * np.random.randn(*py.shape)) / ps)
        log_phi = self.dist.logpdf(normed)
        Phi = self.dist.cdf(normed)
        PI = Phi
        EI = ps * (Phi * normed + np.exp(log_phi))
        logEIapp = np.log(ps) - 0.5 * normed ** 2 - np.log(normed ** 2 - 1)
        logPIapp = -0.5 * normed ** 2 - \
                   np.log(-1 * normed) - np.log(np.sqrt(2 * np.pi))
        
        use_app = ~((normed > -6) & np.isfinite(np.log(EI)) & np.isfinite(np.log(PI))).reshape(-1)
        out = np.zeros((x.shape[0], 3))
        out[:, 0] = lcb.reshape(-1)
        out[:, 1][use_app] = -1 * logEIapp[use_app].reshape(-1)
        out[:, 2][use_app] = -1 * logPIapp[use_app].reshape(-1)
        out[:, 1][~use_app] = -1 * np.log(EI[~use_app].reshape(-1))
        out[:, 2][~use_app] = -1 * np.log(PI[~use_app].reshape(-1))
        return Tensor(out)


class GeneralAcq(Acquisition):
    """GeneralAcq."""
    
    def __init__(self, model, num_obj, num_constr, **conf):
        super().__init__(model, **conf)
        self._num_obj = num_obj
        self._num_constr = num_constr
        self.kappa = conf.get('kappa', 2.0)
        self.c_kappa = conf.get('c_kappa', 0.)
        self.use_noise = conf.get('use_noise', True)
        assert self.model.num_out == self.num_obj + self.num_constr
        assert self.num_obj >= 1
    
    @property
    def num_obj(self) -> int:
        """num_obj."""
        return self._num_obj
    
    @property
    def num_constr(self) -> int:
        """num_constr."""
        return self._num_constr
    
    def eval(self, x: Tensor, xe: Tensor) -> Tensor:
        r"""Acquisition function to deal with general constrained, multi-objective optimization problems.

        Suppose we have $om$ objectives and $cn$ constraints, the problem should has been transformed to :

        Minimize (o1, o1, \\dots,  om)
        S.t.     c1 < 0,
                 c2 < 0,
                 \\dots
                 cb_cn < 0

        In this `GeneralAcq` acquisition function, we calculate lower
        confidence bound of objectives and constraints, and solve the following
        problem:

        Minimize (lcb_o1, lcb_o2, \\dots,  lcb_om)
        S.t.     lcb_c1 < 0,
                 lcb_c2 < 0,
                 \\dots
                 lcb_cn < 0
        """
        py, ps2 = self.model.predict(x, xe)
        py = py.asnumpy()
        ps2 = ps2.asnumpy()
        ps = np.sqrt(ps2)
        if self.use_noise:
            noise = np.sqrt(self.model.noise.asnumpy())
            py += noise * np.random.randn(*py.shape)
        out = np.ones(py.shape)
        out[:, :self.num_obj] = py[:, :self.num_obj] - \
                                self.kappa * ps[:, :self.num_obj]
        out[:, self.num_obj:] = py[:, self.num_obj:] - \
                                self.c_kappa * ps[:, self.num_obj:]
        return Tensor(out)
