# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from hebo.design_space.design_space import DesignSpace
from hebo.acquisitions.acq import Acquisition
from hebo.acq_optimizers.evolution_optimizer import EvolutionOpt
import hebo.mindspore as hebo_ms
import mindspore as ms
from pytest import approx
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')


class ToyExample(Acquisition):
    def __init__(self, constr_v=1.0):
        super().__init__(None)
        self.constr_v = constr_v
    
    @property
    def num_obj(self):
        return 1
    
    @property
    def num_constr(self):
        return 1
    
    def eval(self, x, xe):
        # minimize L2norm(x) s.t. L2norm(x) > constr_v
        out = hebo_ms.sum(x ** 2, axis=1, keep_dims=True)
        constr = self.constr_v - out
        return ms.ops.Concat(axis=1)([out, constr])


class ToyExampleMO(Acquisition):
    def __init__(self):
        super().__init__(None)
    
    @property
    def num_obj(self):
        return 2
    
    @property
    def num_constr(self):
        return 0
    
    def eval(self, x, xe):
        # minimize L2norm(x) s.t. L2norm(x) > 1.0
        o1 = hebo_ms.sum(x ** 2, axis=1, keep_dims=True)
        o2 = hebo_ms.sum((x - 1) ** 2, axis=1, keep_dims=True)
        return ms.ops.Concat(axis=1)([o1, o2])


@pytest.mark.parametrize('constr_v',
                         [1.0, 100.0], ids=['feasible', 'infeasible'])
@pytest.mark.parametrize('lhs_init', [True, False], ids=['lhs', 'rand'])
def test_opt(constr_v, lhs_init):
    space = DesignSpace().parse([
        {'name': 'x1', 'type': 'num', 'lb': -3.0, 'ub': 3.0}
    ])
    acq = ToyExample(constr_v=constr_v)
    opt = EvolutionOpt(space, acq, pop=10, iters=100, lhs_init=lhs_init)
    rec = opt.optimize(initial_suggest=space.sample(3))
    x, xe = space.transform(rec)
    if constr_v < 100:
        assert (approx(1.0, 1e-2) == acq(x, xe)[:, 0].squeeze().asnumpy())


def test_opt_fix():
    space = DesignSpace().parse([
        {'name': 'x1', 'type': 'num', 'lb': -3.0, 'ub': 3.0},
        {'name': 'x2', 'type': 'num', 'lb': -3.0, 'ub': 3.0}
    ])
    acq = ToyExample()
    opt = EvolutionOpt(space, acq, pop=10, iters=100)
    rec = opt.optimize(fix_input={'x1': 1.0})
    print(rec)
    assert (rec['x1'].values == approx(1.0, 1e-2))


def test_opt_int():
    space = DesignSpace().parse([
        {'name': 'x1', 'type': 'num', 'lb': -3.0, 'ub': 3.0},
        {'name': 'x2', 'type': 'int', 'lb': -3.0, 'ub': 3.0}
    ])
    acq = ToyExample()
    opt = EvolutionOpt(space, acq, pop=10, iters=100)
    rec = opt.optimize()
    assert (approx(1.0, 1e-2) == acq(*space.transform(rec))
    [:, 0].squeeze().asnumpy())


def test_mo():
    space = DesignSpace().parse([
        {'name': 'x1', 'type': 'num', 'lb': -3.0, 'ub': 3.0},
        {'name': 'x2', 'type': 'int', 'lb': -3.0, 'ub': 3.0}
    ])
    acq = ToyExampleMO()
    opt = EvolutionOpt(space, acq, pop=10, iters=100)
    rec = opt.optimize()
    assert (rec.shape[0] == 10)
