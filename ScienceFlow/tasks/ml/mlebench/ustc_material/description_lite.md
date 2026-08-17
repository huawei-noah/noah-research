# 🚀 太阳能电池性能预测建模任务书

这是一项典型的**多输出回归 (Multi-output Regression)** 建模任务。在光伏研究中，这四个指标共同描述了太阳能电池在反向扫描（Reverse Scan）下的电流-电压（J-V）特性。

---

## 1. 任务定义 (Task Definition)

* **目标变量 (Targets)**：
    * `JV_reverse_scan_Voc`：开路电压 (V) —— 电池在零电流时的电压。
    * `JV_reverse_scan_Jsc`：短路电流密度 (mA/cm²) —— 电池在零电压时的电流密度。
    * `JV_reverse_scan_FF`：填充因子 (%) —— 最大功率点与 $V_{oc} \times J_{sc}$ 乘积的比值。
    * `JV_reverse_scan_PCE`：光电转换效率 (%) —— 太阳能转化为电能的总效率。
* **输入特征 (Features)**：通常涵盖材料配比、吸光层厚度、退火温度、溶剂处理等工艺参数。
**仔细检查数据泄露**：以上标签不允许出现在特征里，因为他们之间相关性很高。
```python
# 目标列
target_cols = ['JV_reverse_scan_Voc', 'JV_reverse_scan_Jsc', 'JV_reverse_scan_FF', 'JV_reverse_scan_PCE']

---

## 2. 核心挑战与物理逻辑

这些变量之间存在极强的**内在相关性**。根据物理定义，$PCE$ 本质上是由前三个参数推导而来的：

$$PCE = \frac{V_{oc} \times J_{sc} \times FF}{P_{in}}$$

其中 $P_{in}$ 通常为标准光强（100 mW/cm²）。这意味着模型不仅要学习输入与输出的关系，还需隐式地学习这四个目标之间的代数关系。

---


## 3. 评估指标
用validation数据（**不要放在训练集里训练**），针对每个目标分别计算：
* **$R^2$ (决定系数)**：评估模型解释数据的能力。

---

## 5.Submission
- **File:** `submission.csv`（与 `prepared/public/sample_submission.csv` 列名一致）。
- **Schema:** `Ref_ID`，随后四个目标列；每个测试样本一行（`Ref_ID` 与公开 `test.csv` 一致）。

```
Ref_ID,JV_reverse_scan_Voc,JV_reverse_scan_Jsc,JV_reverse_scan_FF,JV_reverse_scan_PCE
16962,0.962122699386503,17.962328174603176,0.651771322292091,12.125109990184594
36452,0.962122699386503,17.962328174603176,0.651771322292091,12.125109990184594
```