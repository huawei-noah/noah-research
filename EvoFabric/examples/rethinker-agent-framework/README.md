# ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control

<a href="https://arxiv.org/abs/2602.04496"><img src="https://img.shields.io/badge/Paper-arXiv-red"></a>


## 🔍 Overview

**ReThinker** is a reasoning framework designed to improve **scientific reasoning** in large language models by enabling them to *rethink* their intermediate conclusions through **guided reflection** and **confidence control**.

Instead of relying on a single-pass chain-of-thought, ReThinker introduces an iterative mechanism that:
- Detects low-confidence reasoning steps
- Actively revisits and revises them
- Produces more robust and reliable final answers

This approach significantly improves reasoning accuracy, especially in complex scientific and multi-step inference tasks.

Our paper can be found at [https://arxiv.org/abs/2602.04496](https://arxiv.org/abs/2602.04496)

---

## ✨ Key Contributions

- **Rethinking-Enabled Agent Framework**  
  ReThinker integrates the rethinking mechanism into an **agent-based workflow**, enabling iterative decision-making and refinement. This design yields **stable and consistent improvements in solution quality** across tasks, rather than occasional gains from stochastic reflection.

- **Guided Reflection Mechanism**  
  Instead of letting a critic independently infer weaknesses from a summarized trajectory, ReThinker **explicitly proposes concrete improvement points during trajectory summarization**, which are then carried into a constrained critic role for reflection and revision. This guidance ensures focused, efficient, and stable rethinking.

- **Confidence Control Strategy**  
  In the selector stage, ReThinker performs iterative selection where each round conditions not only on the **previously selected candidate(s)** but also on the **confidence score(s) of those selections**. This explicit confidence feedback guides the model toward more reliable and stable selection decisions across rethinking rounds.

---

## 📊 Experimental Results
**Main Results of Rethinker on Expert-Level Reasoning Benchmarks.**

| Category                    | Model / Framework                                  | HLE  | GAIA | XBench |
|-----------------------------|----------------------------------------------------|------|------|--------|
| Foundation Model w. tools   | Kimi K2 (Kimi et al., 2025)                        | 18.1 | 57.7 | 50.0   |
| Foundation Model w. tools   | Claude-4.5-Sonnet (Anthropic, 2025)                | 24.5 | 71.2 | 66.0   |
| Foundation Model   w. tools | DeepSeek-V3.2 (Liu et al., 2025a)                  | 27.2 | 63.5 | 71.0   |
| Foundation Model  w. tools  | GLM-4.6 (Zhipu, 2025)                              | 30.4 | 71.9 | 70.0   |
| Foundation Model  w. tools  | GPT-5-high (OpenAI, 2025b)                         | 35.2 | 76.4 | 77.8   |
| Foundation Model   w. tools | Gemini-3-Pro (Google, 2025)                        | 38.3 | 79.0 | 87.0   |
| Inference Framework         | WebExplorer (Liu et al., 2025b)                    | 17.3 | 50.0 | 53.7   |
| Inference Framework         | OpenAI DeepResearch (OpenAI, 2025a)                | 26.6 | 67.4 | –      |
| Inference Framework         | Kimi Researcher (Kimi, 2025)                       | 26.9 | –    | 69.0   |
| Inference Framework         | Tongyi DeepResearch (30BA3B) (Tongyi et al., 2025) | 32.9 | 70.9 | 75.0   |
| Inference Framework         | MiroThinker-v1.0 (30B) (MiroMind et al., 2025)     | 33.4 | 73.5 | 70.6   |
| Inference Framework         | **ReThinker (OpenPangu-72B) (Ours)**               | 33.1 | 72.8 | 78.0   |
| Inference Framework         | **ReThinker (Gemini-3-pro) (Ours)**                        | 52.2 | 81.6 | 90.0   |

## 🚀 Quick Start

### 1. Install Dependencies

First, install the required dependencies. In addition to the base package, you need the extra dependencies for the *rethinker* module:

```bash
pip install evofabric[rethinker]
```

### 2. Configure the Project

Next, set up the configuration file. You can start by copying or modifying the default configuration:

```
configs/config.yaml
```

In this file, make sure to configure the following fields:

* **Model configuration**

  * `api_key`: your model API key
  * `base_url`: the base URL of the model API

* **Web search configuration**

  * `serper_api_key`: API key for web search (used by `web_search`)

* **Web parsing configuration**

  * `jina_api_key`: API key for web content parsing (used by `web_parse`)

### 3. Run the System

After configuring the file, run the main script:

```bash
python run.py --config configs/config.yaml
```

### 4. Output Directory Structure
⚠️ **This step is required if you plan to run the evaluation script.**

If `config.exp.output_root` is set, the output files will be organized as follows:

```
output_root/
    qid00001/
        node1.json
        node2.json
        ...
        result.json
    qid00002/
    ...
```

Each `qidXXXXX` directory corresponds to a single query or experiment, containing intermediate node results and the final aggregated result.

### 5. Run Evaluation

Finally, run the evaluation script to validate the results:

```bash
python evaluation.py \
  --api-key=your-api-key \
  --model-name=your-model-name \
  --base-url=your-base-url \
  --save-result=eval.json \
  --benchmark=hle
```

This will run the benchmark evaluation and save the evaluation results to `eval.json`.

## Acknowledgement
The repository benefit from [Eigen-1](https://github.com/tangxiangru/Eigen-1/tree/main).

## Citation
```
@article{tang2026rethinker,
  author       = {Zhentao Tang and Yuqi Cui and Shixiong Kai and Wenqian Zhao and Ke Ye and Xing Li and Anxin Tian and Zehua Pei and Hui‐Ling Zhen and Shoubo Hu and Xiaoguang Li and Yunhe Wang and Mingxuan Yuan},
  title        = {ReThinker: Scientific Reasoning by Rethinking with Guided Reflection and Confidence Control},
  year         = {2026},
  url          = {https://arxiv.org/abs/2602.04496}
}
```