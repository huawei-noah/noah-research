<div align="center">
  <img src="docs/pics/ultraeval_logo_white.jpg" width="500px"/>
  <br />
  <br />
<p align="center">
  <a href="https://arxiv.org/abs/2404.07584">📖论文</a> •
 <a href="https://ultraeval.openbmb.cn/home"> 🌐网站</a> •
 <a href="#总览">📖总览</a> •
 <a href="#快速开始">🔧快速开始</a> •
 <a href="docs/tutorials/zh/ultraeval.md">🛠️详细教程</a> •
  <a href="README.md">English</a> 
</p>
</div>

# Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hNXtaR3V-VgmG59QJMW7YPaAYgwtmyuH?usp=sharing)

我们在 `Google Colab` 上提供了 `UltraEval` 的快速使用教程，您可以在 `Colab` 中直接运行。


# 更新

- \[2024.6.4\] **UltraEval**被ACL 2024 System Demostration Track (SDT)录用。🔥🔥🔥
- \[2024.4.11\] 我们发布了[UltraEval 论文](https://arxiv.org/abs/2404.07584)🔥🔥🔥，欢迎交流讨论。
- \[2024.2.1\] [MiniCPM](https://github.com/OpenBMB/MiniCPM)已经发布啦🔥🔥🔥，使用了UltraEval作为评测框架。
- \[2023.11.23\] 我们开源了UltraEval评测框架，并发布了第一版榜单。🔥🔥🔥

# 总览
UltraEval是一个开源的基础模型能力评测框架，提供了一套轻量级、易于使用的评测体系，支持主流大模型的性能评估。

UltraEval整体流程如下：
<div align="center">
<p align="center">
<img src="docs/pics/ultraeval_pipeline_white.png" width="800px">
</p>
</div>

它的主要特色如下：
- **轻量易用的评测框架**：具备简洁直观的设计，依赖少，易于部署，具有良好的扩展性，适用多种评测场景。
- **灵活多样的评测方法**：提供了统一的prompt模板和丰富的评估指标，同时支持自定义。
- **高效快速的推理部署**：支持包括torch和vLLM在内的多种模型部署方案，并实现了多实例部署以加速评测过程。
- **公开透明的开源榜单**：维护一个公开的、可追溯和可复现的评测榜单，由社区推动更新，确保透明度。
- **官方权威的评测数据**：采用广泛认可的官方评测集，保证评测的公平性和标准化，确保结果具有可比性和复现性。
- **全面广泛的模型支持**：支持Huggingface平台上的开源模型以及个人训练的模型。


# 快速开始
欢迎体验UltraEval——您的大模型能力评测助手，只需几个简单步骤，即可启动评测：

## 1.安装UltraEval

```shell
git clone https://github.com/OpenBMB/UltraEval.git
cd UltraEval
pip install .
```
## 2.模型测评
进入UltraEval根目录，以下所有指令均在根目录下执行。

### 2.1生成评测任务文件

下载数据
```shell
wget -O RawData.zip "https://cloud.tsinghua.edu.cn/f/11d562a53e40411fb385/?dl=1"
```
Google Drive的[链接](https://drive.google.com/file/d/1S0Ze9ToFC9V6CNzsI_e18ILMAk-l0Ckb/view?usp=drive_link)。

解压数据
```shell
unzip RawData.zip
```
对数据进行预处理
```shell
python data_process.py
```
执行下述指令显示支持的数据集及其对应任务：

```shell
python configs/show_datasets.py
```

通过以下指令指定需要测评的任务：

```shell
python configs/make_config.py --datasets ALL
```
以下是具体的参数描述：
* ``datasets``: 选择数据集，默认为All(所有的数据集)；指定多个数据集，用,间隔。例如：--datasets MMLU,Ceval
* ``tasks``: 选择评测任务，默认为空。
* ``method``: 选择生成方式，默认为gen。
* ``save``: 选择生成评测文件的文件名，默认为eval_config.json。

注意⚠️：当tasks有值时，datasets的数量必须为1。表示执行某个数据集下的某些任务；save是一个文件名，且以.json结尾，不需要传入路径，默认在configs下。执行上述指令将在configs目录下生成评测文件eval_config.json。

RawData.zip中保存了从官网收集的数据，其中为了加快解压速度，Math和race做了预处理（压缩包中保存了代码，便于用户复现）。

### 2.2本地部署模型
以部署meta-llama/Llama-2-7b-hf为例，使用vllm部署模型：
```shell
python URLs/vllm_url.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --gpuid 0 \
    --port 5002
```
以下是具体的参数描述：
* ``model_name``: 模型名，使用vLLM时，model_name和hugging face官方名称需保持一致。
* ``gpuid``: 指定部署模型的gpu id，默认0。如果需要多个，可用,隔开
* ``port``: 部署URL的端口号，默认5002。

执行上面这个代码，会产生一个URL。例如，这里URL为 `http://127.0.0.1:5002/infer`，其中 `5002` 是端口号，`/infer` 是URL路径，由 `URLs/vllm_url.py` 文件中的 `@app.route("/infer", methods=["POST"])` 指定。

关于个人训练的模型以及多GPU批量评测方式请参考[Tutorial](./docs/tutorials/zh/ultraeval.md)。

### 2.3进行测评获取测评结果
创建一个bash脚本，执行main.py程序，获取测评结果：
```shell
python main.py \
    --model general \
    --model_args url=$URL,concurrency=1 \
    --config_path configs/eval_config.json \
    --output_base_path logs \
    --batch_size 1 \
    --postprocess general_torch \
    --params models/model_params/vllm_sample.json \
    --write_out
```
以下是具体的参数描述：
* ``model``: 指定模型，目前支持general、gpt-3.5-turbo、gpt-4三种模型。
* ``model_args``: 指定2.2生成的URL，初始化模型参数，以及并发线程数。以逗号分隔，参数名和参数值用=连接。例如：url=$URL,concurrency=1。
* ``config_path``: 指定2.1中的评测文件路径，默认为configs/eval_config.json。
* ``output_base_path``: 指定评测结果保存路径，默认为logs。
* ``batch_size``: 指定批处理数量，默认为1。
* ``num_fewshot``: 指定fewshot样本的数量。
* ``postprocess``: 指定后处理方法，默认为general_torch。
* ``params``: 指定模型推理时的参数，默认为models/model_params/vllm_sample.json。
* ``write_out``: 是否保存每个instance的相关数据，默认为False。
* ``limit``: 评测每个任务的一定数量的instance，默认为None。

测评结果保存在路径下：
```shell
output_base_path    #：输出路径
--timestamp # 时间戳
----task1   # 评测任务
--------config.json # 评测任务的相关参数配置记录
--------final_metrics.json  # 该任务的最终结果
--------instance.jsonl  # 该任务每一条样例的详细结果
----....    # 其他任务目录
----_all_results.json   # 所有评测任务结果的综合
```
### 2.4更多测评功能支持
更多测评方法和功能（自定义评测集、批量测评、多GPU加速）详情请见[Tutorials.md](./docs/tutorials/zh/ultraeval.md)

# 数据集支持

UltraEval支持59个评测数据集，并按能力分类全面衡量大模型能力，数据集支持如下：
<table border="1">
  <tr>
    <th>一级分类</th>
    <th>二级分类</th>
    <th>数据集列表</th>
  </tr>
  <tr>
    <td rowspan="2">知识推理</td>
    <td>学科知识</td>
    <td>CMMLU, MMLU, CEval, AGI-Eval, JEC-QA, MEDMCQA, MEDQA-MCMLE, MEDQA-USMLE, GAOKAO-Bench</td>
  </tr>
  <tr>
    <td>世界知识</td>
    <td>NQ-open, TriviaQA, TruthfulQA</td>
  </tr>
  <tr>
    <td>数学计算</td>
    <td>数学计算</td>
    <td>GSM8K, MATH</td>
  </tr>
  <tr>
    <td>代码生成</td>
    <td>代码生成</td>
    <td>HumanEval, MBPP</td>
  </tr>
  <tr>
    <td rowspan="3">推理</td>
    <td>逻辑推理</td>
    <td>BBH</td>
  </tr>
  <tr>
    <td>蕴含关系</td>
    <td>AX-B, AX-G, CB, CMNLI, OCNLI, OCNLI-FC, RTE</td>
  </tr>
  <tr>
    <td>常识推理</td>
    <td>HellaSwag, OpenBookQA, ARC-c, ARC-e, CommonsenseQA, COPA, PIQA, SIQA, WinoGrande, Story Cloze, StrategyQA, TheoremQA</td>
  </tr>
  <tr>
    <td rowspan="6">语言理解</td>
    <td>阅读理解</td>
    <td>boolq, C3, ChiD, DRCD, LAMBADA, MultiRC, QuAC, RACE, RECORD, SQuAD, TyDi QA, SummEdits</td>
  </tr>
  <tr>
    <td>翻译</td>
    <td>FLORES, wmt20-en-zh, wmt20-en-zh</td>
  </tr>
  <tr>
    <td>语义相似度</td>
    <td>AFQMC, BUSTM</td>
  </tr>
  <tr>
    <td>词义消歧</td>
    <td>CLUEWSC, WIC, Winogender, WSC</td>
  </tr>
  <tr>
    <td>情感分析</td>
    <td>EPRSTMT</td>
  </tr>
  <tr>
    <td>新闻分类</td>
    <td>TNEWS</td>
  </tr>
</table>


# 测评榜单
请访问UltraEval[官方排行榜](https://ultraeval.openbmb.cn/rank)了解最新模型及其在每个维度中的详细结果。

# 致谢
- [HuggingFace](https://huggingface.co)
- [vLLM](https://github.com/vllm-project/vllm/blob/main)
- [Harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/master)
- [OpenCompass](https://github.com/open-compass/opencompass)
# 联系我们
如果有关于 UltraEval的问题或建议或功能请求，请提交GitHub Issues，共同建设开源透明的UltraEval评测社区。
# License
本项目遵循Apache-2.0协议。

# Citation 
如果您使用了UltraEval，请引用我们的论文。

**BibTeX:**
```bibtex
@misc{he2024ultraeval,
      title={UltraEval: A Lightweight Platform for Flexible and Comprehensive Evaluation for LLMs}, 
      author={Chaoqun He and Renjie Luo and Shengding Hu and Yuanqian Zhao and Jie Zhou and Hanghao Wu and Jiajie Zhang and Xu Han and Zhiyuan Liu and Maosong Sun},
      year={2024},
      eprint={2404.07584},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```