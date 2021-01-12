# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest
import torch
import pdb

from pytest import approx

from bo.models.nn.conditional_deep_ensemble import ConditionalDeepEnsemble
from bo.design_space.design_space import DesignSpace

@pytest.mark.parametrize('output_noise', [True, False], ids = ['nll', 'mse'])
@pytest.mark.parametrize('rand_prior',   [True, False], ids = ['with_prior', 'no_prior'])
@pytest.mark.parametrize('num_processes', [1, 5], ids = ['seq', 'para'])
def test_cond(output_noise, rand_prior, num_processes):
    space = DesignSpace().parse([
        {'name' : 'param_A' , 'type' : 'bool'}, 
        {'name' : 'param_B' , 'type' : 'num', 'lb': 0, 'ub' : 1}, 
        {'name' : 'param_C' , 'type' : 'num', 'lb': 0, 'ub' : 1}, 
        {'name' : 'param_D' , 'type' : 'num', 'lb': 0, 'ub' : 1}, 
        {'name' : 'param_A2' , 'type' : 'cat', 'categories' : ['a', 'b', 'c']}, 
        {'name' : 'param_B2' , 'type' : 'num', 'lb': 0, 'ub' : 1}, 
        ])
    X = space.sample(50)
    y = torch.randn(50, 1)
    cond  = {'param_A'  : None,
             'param_B'  : ('param_A', True),
             'param_C'  : ('param_A', True),
             'param_D'  : ('param_A', [True]),  # XXX: enable value could be scaler value or list of values
             'param_A2' : None, 
             'param_B2' : ('param_A2', ['a'])
            }
    model = ConditionalDeepEnsemble(4, 0, 1, space, cond, num_epoch=1, rand_prior = rand_prior, num_processes = num_processes, output_noise = output_noise)
    model.fit(*space.transform(X), y)
    model.fit(*space.transform(X), y) # test warm-starting

    with torch.no_grad():
        py, ps2  = model.predict(*space.transform(X))
        disabled = ((X.param_A == False) & (X.param_A2 != 'a')).values
        assert(py[disabled].var() == approx(0., abs = 1e-4))
