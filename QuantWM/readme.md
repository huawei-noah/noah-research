# An Empirical Study of World Model Quantization
By Zhongqian Fu, Tianyi Zhao, Kai Han, Hang Zhou, Xinghao Chen and Yunhe Wang.  [[arXiv]](https://www.arxiv.org/abs/2602.02110)


This project is designed to **evaluate the quantization inference behavior of World Model (Dino-WM)**. The code is based on the official Dino-WM implementation and integrates several **Post-Training Quantization (PTQ)** methods to replicate the core conclusions from the related research paper.

---

## Base Repository

This project is built upon the official Dino-WM repository:

👉 https://github.com/gaoyuezhou/dino_wm.git

Please ensure you have the complete environment and dependencies to run the original Dino-WM planning code.

---

## 1. Environment and Data Preparation

**Please strictly follow the instructions in the official Dino-WM repository for the following steps:**

- Python / CUDA environment setup  
- Dependency installation  
- Wall / PushT dataset download and preparation  

Before proceeding with this README, please ensure that you can run the original floating-point (FP) planning inference code **without modifications**.

---

## 2. Path and Placeholder Description

All commands in this document use placeholders. Please replace them with actual values before running the scripts:

| Placeholder | Description |
|-------------|-------------|
| `<PROJECT_ROOT>` | Root directory of the project |
| `<DATASET_DIR>` | Root directory of the dataset |
| `<GPU_ID>` | The GPU ID you want to use |

---

## 3. Running Preparation

```bash
cd <PROJECT_ROOT>
mkdir -p plan_outputs
export DATASET_DIR=<DATASET_DIR>
```

---

## 4. Floating-Point (FP) Baseline

`plan.py`: **Floating-point planning inference baseline without any quantization operations**, used to compare performance degradation under different quantization configurations. Reference: DINO_WM repository.

```bash
# PushT
python plan.py --config-name plan_pusht.yaml model_name=pusht
# Wall
python plan.py --config-name plan_wall.yaml model_name=wall
```

---

## 5. Activation Statistics (For SmoothQuant)

`plan_act.py` is used to **statistically analyze the activation distribution during the iterative planning process of World Model**, and generate the scale parameters required for SmoothQuant.

```bash
# Wall
CUDA_VISIBLE_DEVICES=<GPU_ID> python plan_act.py   --config-name plan_wall.yaml   model_name=wall_single   tag=fp   sta_scale=True   n_evals=50   planner.max_iter=2   planner.sub_planner.opt_steps=10   scale_tag=iter2_opt10_eval50

# PushT
CUDA_VISIBLE_DEVICES=<GPU_ID> python plan_act.py   --config-name plan_pusht.yaml   model_name=pusht   tag=fp   sta_scale=True   n_evals=50   planner.max_iter=2   planner.sub_planner.opt_steps=30   scale_tag=iter2_opt30_eval50
```

---

## 6. Quantization Inference Experiments (PTQ)

The following scripts are used to evaluate the planning performance of Dino-WM under **different quantization methods and bit-width configurations**. Below are examples using the Wall dataset.

### General Environment Variables

```bash
# Group size
export W_GROUP_SIZE=-1
# Or
export W_GROUP_SIZE=128
```

---

### 6.1 RTN (Round-To-Nearest)

Script: `plan_quant_omse_rtn.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="layer_wise"  quant_iter=2   tag=RTN_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_RTN.txt 2>&1
```

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="token_wise"  quant_iter=2   tag=RTN_quant_Pw8a8_Ew8a8_per_token_iter2   | tee -a plan_outputs/logfile_plan_wall_RTN.txt 2>&1
```

---

### 6.2 OMSE

Script: `plan_quant_omse_rtn.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omse_rtn.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="omse"   a_quant_method="minmax"  calib_mode_a="layer_wise"   quant_iter=2   tag=OMSE_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_OMSE.txt 2>&1
```

---

### 6.3 SmoothQuant

Script: `plan_quant_smooth.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_smooth.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="minmax"   a_quant_method="minmax"  calib_mode_a="layer_wise"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=smooth_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_smoothquant.txt 2>&1
```

---

### 6.4 OmniQuant

Script: `plan_quant_omniquant.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_omniquant.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=8   encoder_wbit=8   encoder_abit=8   w_quant_method="omniquant"   a_quant_method="omniquant"  calib_mode_a="layer_wise"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=omni_quant_Pw8a8_Ew8a8_per_tensor_iter2   | tee -a plan_outputs/logfile_plan_wall_omniquant.txt 2>&1
```

---

### 6.5 AWQ

Script: `plan_quant_awq.py`

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> python -u plan_quant_awq.py   --config-name plan_wall.yaml   model_name=wall_single   quant=True   quant_encoder=True   predictor_wbit=8   predictor_abit=16   encoder_wbit=8   encoder_abit=16   w_quant_method="awq"   a_quant_method="minmax"   quant_iter=2   scale_tag=iter2_opt10_eval50   tag=awq_quant_Pw8a16_Ew8a16_iter2   | tee -a plan_outputs/logfile_plan_wall_awq.txt 2>&1
```

---

## 7. Key Parameter Description

| Parameter | Description |
|-----------|-------------|
| `predictor_wbit / encoder_wbit` | Weight quantization bit-width |
| `predictor_abit / encoder_abit` | Activation quantization bit-width |
| `w_quant_method` | Weight quantization method |
| `a_quant_method` | Activation quantization method |
| `quant_iter` | Quantization calibration iterations |
| `scale_tag` | Activation scale for SmoothQuant |
| `planner.max_iter` | Outer loop iterations of the planner |
| `planner.sub_planner.opt_steps` | Optimization steps for the sub-planner |
| `n_evals` | Number of evaluation rounds |
| `calib_mode_a` | Activation quantization granularity: "layer_wise"(default) / "token_wise" |

---

## 8. Script Function Overview

| Script | Function |
|--------|----------|
| `plan.py` | Floating-point inference (FP baseline) |
| `plan_act.py` | Activation statistics (for SmoothQuant) |
| `plan_quant_omse_rtn.py` | RTN / OMSE |
| `plan_quant_smooth.py` | SmoothQuant |
| `plan_quant_omniquant.py` | OmniQuant |
| `plan_quant_awq.py` | AWQ |

---

## Citation

```bibtex
@misc{fu2026empiricalstudyworldmodel,
      title={An Empirical Study of World Model Quantization}, 
      author={Zhongqian Fu and Tianyi Zhao and Kai Han and Hang Zhou and Xinghao Chen and Yunhe Wang},
      year={2026},
      eprint={2602.02110},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.02110}, 
}
```
