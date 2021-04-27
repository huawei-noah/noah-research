# Notes for HEBO migration

## Plan

1. MindSpore installation
2. Learning MindSpore
3. HEBO migration
4. Passing all test
5. Bayesmark testing
6. Doc and report

## Notes

- Operations like `x[0, 0] = 3` not supported?
- Basic operations like `x.sum()`, `x.std()`, `x.var()` not supported (though they can be implemented by `Ops`)
- Lack of random module, like `randn()`, `rand()`, `randint`
- Lack of logical operators like `a | b`,  `a & b`
- `Tensor(1) + Tensor(1)` raises exception
- `Ones()` and `Zeros` do not support create tensor of shpae `(x, 0)`, but we can do that in `mindspore.numpy`
- Best practice for `ms.ops` operators
- Where is `mindspore.bool`?


```python
from mindspore import Tensor
print(Tensor(1) + Tensor(1) # XXX: can not add two integers, should be a bug
```

```python
from mindspore import Tensor
x = Tensor(0) / Tensor(0)
assert x == 0 # 0 / 0 == 0
```


```python
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np

try:
    x = ms.Tensor(np.random.randn(10, 0))
except ValueError:
    print('Can not create tensor with zero dimension from `Tensor`')

x = mnp.zeros((10, 0)) # XXX: but we can create it from `mindspore.numpy`
print(x)
```

```python
from mindspore import Tensor
x = Tensor([1., 2.])
print(x[0]) # slice operations only supported for int32/fp32, but not for fp64
```

```python
import torch
import mindspore as ms
import numpy as np

assert all([])
assert np.all([])
assert torch.tensor([]).bool().all()

try:
    ms.Tensor([]).astype(bool).all()
except RuntimeError:
    print('Error')
```

```python
# XXX: This also raises exception in GPU
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np

x1 = mnp.zeros((10, 0))
x2 = mnp.zeros((10, 1))

x_all = ms.ops.Concat(axis = 1)([x1, x2]) # OK
print(x_all)

x_all = ms.ops.Concat(axis = 1)([x2, x1]) # Error
print(x_all)
```

The below code works on GPU and CPU + Win10, but not for CPU + Linux

Core dump exception is raised

```python
import mindspore as ms
valid = ms.ops.IsFinite()(ms.ops.Zeros()((88, 6), ms.float32)).all()
```
