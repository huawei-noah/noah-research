# LLAMA(hidden_dim=11008, dim=4096)

## Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 517073 | 517390 | 517527 | 517370 |
| Sparse | 786467 | 408123 | 225251 | 152883 |
| SpeedUP (=Dense/Sparse) | 0.657463059 | 1.267730562 | 2.297556948 | 3.384091102 |

## Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 5179069.33 | 5175831.041 | 5177646.605 | 5179365.54 |
| Sparse | 6771117.844 | 4211196.954 | 1530418.214 | 168884.047 |
| SpeedUP (=Dense/Sparse) | 3.86793E-07 | 9.66389E-07 | 1.54581E-06 | 1.83621E-06 |

---

# QWEN(hidden_dim=18944, dim=3584)

## Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 998948 | 999671 | 999493 | 999317 |
| Sparse | 1189586.66 | 728912 | 321422 | 168589 |
| SpeedUP (=Dense/Sparse) | 0.839743781 | 1.371456362 | 3.109597352 | 5.927533825 |

## Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 8083272.5 | 8082262.707 | 8085862.292 | 8087520.261 |
| Sparse | 10674569.23 | 6827700.62 | 2608982.072 | 354016.121 |
| SpeedUP (=Dense/Sparse) | 2.00211E-07 | 5.00165E-07 | 8.00406E-07 | 9.50649E-07 |

---

# MISTRAL(hidden_dim=14336, dim=4096)

## Performance on Ascend910B

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 1746690 | 1747529 | 1733415 | 1757405 |
| Sparse | 2015408 | 1242137 | 685084 | 426516.8 |
| SpeedUP (=Dense/Sparse) | 0.866668188 | 1.406872994 | 2.530222571 | 4.120365247 |

## Performance on Ascend310P3

| Level of sparsity | 20% | 50% | 80% | 95% |
|---|---|---|---|---|
| Dense | 7165279.02 | 7167873.255 | 7169420.658 | 7172249.696 |
| Sparse | 8877599.342 | 5421312.263 | 2105415.456 | 251974.262 |
| SpeedUP (=Dense/Sparse) | 1.14502E-07 | 2.86118E-07 | 4.61517E-07 | 5.4057E-07 |


In each cell value is a total time in us
