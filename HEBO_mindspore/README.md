# README

## Installation

```bash
python -m pip install .
```

## Demo

```python
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import Tensor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from hebo.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

ds.config.set_num_parallel_workers(1)

def obj(lr : float, weight_decay : float, hidden_size : int) -> float:
    X, y = load_boston(return_X_y = True)
    y    = y.reshape(-1, 1)
    X_tr, X_tst, y_tr, y_tst = train_test_split(
            X.astype(np.float32),
            y.astype(np.float32),
            test_size    = 0.3,
            shuffle      = True,
            random_state = 42)

    dataset = ds.GeneratorDataset(
            lambda: ((x_, y_) for (x_, y_) in zip(X_tr, y_tr)),
            column_names = ['x_train', 'y_train'],
            shuffle      = True, 
            python_multiprocessing = False
            )
    dataset = dataset.batch(32)

    net = nn.SequentialCell(
            nn.Dense(13, hidden_size),
            nn.ReLU(),
            nn.Dense(hidden_size, 1))

    crit          = nn.MSELoss()
    opt           = nn.Adam(params = net.trainable_params(), learning_rate = lr, weight_decay = weight_decay)
    net_with_crit = nn.WithLossCell(net, crit)
    train_net     = nn.TrainOneStepCell(net_with_crit, opt)
    for _ in range(100):
        for d in dataset.create_dict_iterator():
            train_net(d['x_train'], d['y_train'])

    py_tst = net(Tensor(X_tst)).asnumpy()
    r2     = r2_score(y_tst, py_tst)
    return -1 * np.array(r2).reshape(-1, 1)

if __name__ == '__main__':
    space = DesignSpace().parse([
        {'name' : 'lr' ,  'type' : 'pow', 'lb' : 1e-4, 'ub' : 3e-2}, 
        {'name' : 'weight_decay' ,  'type' : 'pow', 'lb' : 1e-6, 'ub' : 3e-2}, 
        {'name' : 'hidden_size' ,  'type' : 'int', 'lb' : 16, 'ub' : 128}, 
    ])
    opt = HEBO(space)
    for iter in range(50):
        rec          = opt.suggest()
        lr           = float(rec.iloc[0].lr)
        weight_decay = float(rec.iloc[0].weight_decay)
        hidden_size  = int(rec.iloc[0].hidden_size)
        observation  = obj(lr, weight_decay, hidden_size)
        opt.observe(rec, observation)
        print('After %d iterations, best obj is %.3f' % (iter + 1, opt.y.min()))
```
