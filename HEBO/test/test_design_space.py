# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

import pytest
from pytest import approx

from hebo.design_space.design_space      import DesignSpace
from hebo.design_space.numeric_param     import NumericPara
from hebo.design_space.integer_param     import IntegerPara
from hebo.design_space.pow_param         import PowPara
from hebo.design_space.categorical_param import CategoricalPara
from hebo.design_space.bool_param        import BoolPara

def test_design_space():
    space = DesignSpace().parse([
        {'name' : 'x0', 'type' : 'num', 'lb' : 0, 'ub' : 7}, 
        {'name' : 'x1', 'type' : 'int', 'lb' : 0, 'ub' : 7}, 
        {'name' : 'x2', 'type' : 'pow', 'lb' : 1e-4, 'ub' : 1e-2, 'base' : 10}, 
        {'name' : 'x3', 'type' : 'cat', 'categories' : ['a', 'b', 'c']}, 
        {'name' : 'x4', 'type' : 'bool'}
    ])
    assert space.numeric_names   == ['x0', 'x1', 'x2', 'x4']
    assert space.enum_names      == ['x3']
    assert space.num_paras       == 5
    assert space.num_numeric     == 4
    assert space.num_categorical == 1

    samp    = space.sample(10)
    x, xe   = space.transform(samp)
    x_, xe_ = space.transform(space.inverse_transform(x, xe))
    assert (x - x_).abs().max() < 1e-4

    assert (space.opt_lb <= space.opt_ub).all()

    assert not space.paras['x0'].is_discrete
    assert space.paras['x1'].is_discrete
    assert not space.paras['x2'].is_discrete
    assert space.paras['x3'].is_discrete
    assert space.paras['x4'].is_discrete
