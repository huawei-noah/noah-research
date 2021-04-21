"""Step Integer Parameter."""

import numpy as np
from .param import Parameter


class StepIntPara(Parameter):
    """Integer parameter, that increments with a fixed step, like `[4, 8, 12, 16]`.

    The config would be like `{'name' : 'x', 'type' : 'step_int', 'lb' : 4, 'ub' : 16, 'step' : 4}`.
    """

    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb = round(param_dict['lb'])
        self.ub = round(param_dict['ub'])
        self.step = round(param_dict['step'])
        self.num_step = (self.ub - self.lb) // self.step

    def sample(self, num=1):
        """Sample."""
        return np.random.randint(
            0, self.num_step + 1, num) * self.step + self.lb

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform."""
        return (x - self.lb) / self.step

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        """Inverse Transform."""
        x_recover = x * self.step + self.lb
        return x_recover.round().astype(int)

    @property
    def is_numeric(self):
        """Is Numeric."""
        return True

    @property
    def opt_lb(self):
        """Lower bound."""
        return 0.

    @property
    def opt_ub(self):
        """Upper bound."""
        return 1. * self.num_step

    @property
    def is_discrete(self):
        """Is Discrete?."""
        return True

    @property
    def is_discrete_after_transform(self):
        """Is Discrete After Transform?."""
        return True
