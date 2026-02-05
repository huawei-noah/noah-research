# Dino-WM Planning & Quantization Experiments

本项目用于 **评估 World Model（Dino-WM）在长程规划任务中的量化推理行为**，本代码基于 Dino-WM 官方实现，并系统集成了多种主流 **后训练量化（Post-Training Quantization, PTQ）方法**，用于复现实验论文中的核心结论。

> 📌 **研究关注点**
> - 不同 PTQ 方法在 Wall / PushT 任务中的表现差异  
> - World Model 在长程规划中的量化误差累积  
> - 编码器（Encoder）与预测器（Predictor）的量化敏感性不对称  


---

## 基础仓库

本项目基于 Dino-WM 官方仓库构建：

👉 https://github.com/gaoyuezhou/dino_wm.git

请确保你已具备运行原始 Dino-WM 规划代码的完整环境与依赖。

---

## 1. 环境与数据准备

**请严格按照 Dino-WM 官方仓库说明完成以下步骤：**

- Python / CUDA 环境配置  
- 依赖安装  
- Wall / PushT 数据集下载与准备  

在继续本 README 之前，请确认你可以 **无修改运行原始浮点（FP）规划推理代码**。

---

## 2. 路径与占位符说明

本文中所有命令均使用占位符，请在运行前自行替换：

| 占位符 | 含义 |
|------|------|
| `<PROJECT_ROOT>` | 项目根目录 |
| `<DATASET_DIR>` | 数据集根目录 |
| `<GPU_ID>` | 使用的 GPU 编号 |

---

## 3. 运行准备

```bash
cd <PROJECT_ROOT>
mkdir -p plan_outputs
export DATASET_DIR=<DATASET_DIR>
```


## 4. 浮点（FP）推理基线

`plan.py`：**不包含任何量化操作的浮点规划推理基线**，用于对比不同量化配置下的性能退化。备注：参考DINO_WM仓库

```bash
# PushT
python plan.py --config-name plan_pusht.yaml model_name=pusht
# Wall
python plan.py --config-name plan_wall.yaml model_name=wall
```

---

## 5. 激活统计（用于 SmoothQuant）

`plan_act.py` 用于 **统计 World Model 在迭代规划过程中的激活分布**，并生成 SmoothQuant 所需的 scale 参数。

```bash
# Wall
CUDA_VISIBLE_DEVICES=<GPU_ID> python plan_act.py   --config-name plan_wall.yaml   model_name=wall_single   tag=fp   sta_scale=True   n_evals=50   planner.max_iter=2   planner.sub_planner.opt_steps=10   scale_tag=iter2_opt10_eval50

# PushT
CUDA_VISIBLE_DEVICES=<GPU_ID> python plan_act.py   --config-name plan_pusht.yaml   model_name=pusht   tag=fp   sta_scale=True   n_evals=50   planner.max_iter=2   planner.sub_planner.opt_steps=30   scale_tag=iter2_opt30_eval50
```

## 6. 量化推理实验（PTQ）

以下脚本用于在 **不同量化方法与 bit-width 配置下** 评估 Dino-WM 的规划性能，以下以wall数据为例

### 通用环境变量设计

```bash
#group size
export W_GROUP_SIZE=-1
#or
export W_GROUP_SIZE=128
```

---

### 6.1 RTN（Round-To-Nearest）

脚本：`plan_quant_omse_rtn.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="layer_wise"  quant_iter=2   tag=RTN_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_RTN.txt 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="token_wise"  quant_iter=2   tag=RTN_quant_Pw8a8_Ew8a8_per_token_iter2   | tee -a plan_outputs/logfile_plan_wall_RTN.txt 2>&1
```

---

### 6.2 OMSE

脚本：`plan_quant_omse_rtn.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="omse"   a_quant_method="minmax"  calib_mode_a="layer_wise"   quant_iter=2   tag=OMSE_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_OMSE.txt 2>&1
```

---

### 6.3 SmoothQuant

脚本：`plan_quant_smooth.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_smooth.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="layer_wise"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=smooth_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_smoothquant.txt 2>&1
```

---

### 6.4 OmniQuant

脚本：`plan_quant_omniquant.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omniquant.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="omniquant"   a_quant_method="omniquant"  calib_mode_a="layer_wise"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=omni_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_omniquant.txt 2>&1
```

---

### 6.5 AWQ

脚本：`plan_quant_awq.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_awq.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=16   encoder_wbit=8   encoder_abit=16   w_quant_method="awq"   a_quant_method="minmax"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=awq_quant_Pw8a16_Ew8a16_iter2   | tee -a plan_outputs/logfile_plan_wall_awq.txt 2>&1
```

---

## 7. 关键参数说明

| 参数 | 说明 |
|----|----|
| `predictor_wbit / encoder_wbit` | 权重量化 bit-width |
| `predictor_abit / encoder_abit` | 激活量化 bit-width |
| `w_quant_method` | 权重量化方法 |
| `a_quant_method` | 激活量化方法 |
| `quant_iter` | 量化校准迭代轮数 |
| `scale_tag` | SmoothQuant 使用的激活 scale |
| `planner.max_iter` | 规划器外层迭代次数 |
| `planner.sub_planner.opt_steps` | 子规划器优化步数 |
| `n_evals` | 评估回合数 |
| `calib_mode_a` | 激活量化粒度: "layer_wise"(default) / "token_wise" |

---

## 8. 脚本功能总览

| 脚本 | 功能 |
|----|----|
| `plan.py` | 浮点推理（FP baseline） |
| `plan_act.py` | 激活统计（SmoothQuant） |
| `plan_quant_omse_rtn.py` | RTN / OMSE |
| `plan_quant_smooth.py` | SmoothQuant |
| `plan_quant_omniquant.py` | OmniQuant |
| `plan_quant_awq.py` | AWQ |

---
