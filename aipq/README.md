# Content-diverse comparisons improve IQA

Source code for FR-IQA with agnostic sampling and differentiable correlation regularizers.
[[Paper]](https://bmvc2022.mpi-inf.mpg.de/0244.pdf).

## Code organization

[core](core) -- core modules to be used in other parts  
[data](data) -- csv files with all the filepaths to dataset images  
[mindspore](mindspore) -- mindspore specific code for evaluation
[torch](torch) -- torch specific code for evaluation
[runs](runs) -- bash script with options to evaluate multiple dataset/model combinations
[checkpoints](checkpoints) -- location of the (compressed) model checkpoints

## Installation

Packages used in the development code (Python 3.8) are in [reqs_py38.txt](reqs_py38.txt), and can be installed with `pip install -r reqs_py38.txt`. BMVC'22 paper results are based on this configuration. Newer versions of PyTorch are known to produce slightly lower PLCC (Pearson Linear Correlation Coefficients).

Newer versions of some of the necessary libraries have since became available and can be installed using `pip install -r reqs_max.txt` instead.


## Model checkpoints
Three models - in both Mindspore and Pytorch - are provided to replicate results from the BMVC'22 paper. They are located in this [HuggingFace repository](https://huggingface.co/huawei-noah/aipq). Just copy the HF repository contents to the ``checkpoints`` directory. 
  
## Datasets

Our models are trained **KADID** and validated on **PIPAL**.  
* Evaluation on **LIVE**, **CSIQ**, **TID2013**, **Liu**, **Ma**, **SHRQ**, **DIBR**, **PIPAL** (train split).  
* Other evaluation datasets: **QADS**, **TQD**, **KADID** and **PieAPP** (test split).

For downloading the datasets, see original papers.

All datasets are referenced in [data](data) where there is a .csv file for every dataset to provide filepaths for distorted images and their reference.



## Metrics

Check [core/common/utils.py](core/common/utils.py) to measure the **PLCC** (and its logistic 4-parameter variant), **SRCC** and **KRCC**.

These metrics have their differentiable counterpart to be included in the loss function.



## Experiments

The [runs/eval.sh](runs/eval.sh) script allows to run multiple combinations of models and datasets. Using either pytorch or mindspore as the framework. Check the variables defined inside the script to replicate paper results.

We provide models for `three architectures` with `agnostic pair formation` and all `regularization` enabled, leading to the numbers shown on Table 3. of the paper:

![table-3](https://github.com/huawei-noah/noah-research/blob/master/aipq/assets/paper_table3.jpg)

# Citation
    @inproceedings{thong2022content,
      title={Content-Diverse Comparisons improve {IQA}},
      author={Thong, William and Costa Pereira, Jose and Parisot, Sarah and Leonardis, Ales and McDonagh, Steven},
      booktitle={British Machine Vision Conference},
      year={2022}
    }
【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】

