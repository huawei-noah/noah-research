"""Parameter."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A

# PARTICULAR PURPOSE. See the MIT License for more details.

import pandas as pd
import numpy  as np
from abc import ABC, abstractmethod


class Parameter(ABC):
    """Base Parameter Class."""
    
    def __init__(self, param_dict):
        self.param_dict = param_dict
        self.name = param_dict['name']
        pass
    
    @abstractmethod
    def sample(self, num=1) -> pd.DataFrame:
        """Sample."""
        pass
    
    @abstractmethod
    def transform(self, x: np.array) -> np.array:
        """Transform."""
        pass
    
    @abstractmethod
    def inverse_transform(self, x: np.array) -> np.array:
        """Inverse Transform."""
        pass
    
    @property
    @abstractmethod
    def is_numeric(self) -> bool:
        """Is Numeric."""
        pass
    
    @property
    @abstractmethod
    def is_discrete(self) -> bool:
        """Integer and categorical variable."""
        pass
    
    @property
    @abstractmethod
    def is_discrete_after_transform(self) -> bool:
        """Is Discrete After Transform."""
        pass
    
    @property
    def is_categorical(self) -> bool:
        """Is categorical."""
        return not self.is_numeric
    
    @property
    @abstractmethod
    def opt_lb(self) -> float:
        """Lower bound."""
        pass
    
    @property
    @abstractmethod
    def opt_ub(self) -> float:
        """Upper bound."""
        pass
