# Performance results on NPU

## LLAMA(hidden_dim=11008, dim=4096)

### Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 517073 | 517390 | 517527 | 517370 |
| Sparse | 786467 | 408123 | 225251 | 152883 |
| SpeedUP (=Dense/Sparse) | 0.658 | 1.268 | 2.298 | 3.384 |

### Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 5179069.33 | 5175831.041 | 5177646.605 | 5179365.54 |
| Sparse | 6771117.844 | 4211196.954 | 1530418.214 | 168884.047 |
| SpeedUP (=Dense/Sparse) | 3.868E-07 | 9.664E-07 | 1.546E-06 | 1.836E-06 |


## QWEN(hidden_dim=18944, dim=3584)

### Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 998948 | 999671 | 999493 | 999317 |
| Sparse | 1189586.66 | 728912 | 321422 | 168589 |
| SpeedUP (=Dense/Sparse) | 0.84 | 1.371 | 3.11 | 5.928 |

### Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 8083272.5 | 8082262.707 | 8085862.292 | 8087520.261 |
| Sparse | 10674569.23 | 6827700.62 | 2608982.072 | 354016.121 |
| SpeedUP (=Dense/Sparse) | 2.002E-07 | 5.002E-07 | 8.004E-07 | 9.506E-07 |

## MISTRAL(hidden_dim=14336, dim=4096)

### Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 1746690 | 1747529 | 1733415 | 1757405 |
| Sparse | 2015408 | 1242137 | 685084 | 426516.8 |
| SpeedUP (=Dense/Sparse) | 0.867 | 1.407 | 2.530 | 4.120 |

### Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 7165279.02 | 7167873.255 | 7169420.658 | 7172249.696 |
| Sparse | 8877599.342 | 5421312.263 | 2105415.456 | 251974.262 |
| SpeedUP (=Dense/Sparse) | 1.145E-07 | 2.861E-07 | 4.615E-07 | 5.406E-07 |

In each cell, the value is the total time in microseconds.
