# XeroAlign: Zero-shot cross-lingual transformer alignment
This is the code repository and instructions for the ACL 2021 paper https://aclanthology.org/2021.findings-acl.32/. If you find it useful, please cite the publication as:

```
@inproceedings{gritta-iacobacci-2021-xeroalign,
    title = "{X}ero{A}lign: Zero-shot cross-lingual transformer alignment",
    author = "Gritta, Milan and Iacobacci, Ignacio",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.32",
    doi = "10.18653/v1/2021.findings-acl.32",
    pages = "371--381",
}
```
### Third Party Open Source Notice
The starting point for this repo was cloned from [JointBERT](https://github.com/monologg/JointBERT). Some unmodified code that does not constitute the key methodology introduced in our paper remains in the codebase.

### Getting started
**`git clone`** the project first, then we set up data, models and runs.

#### Datasets
We can provide the smaller datasets over email at milan.gritta AT huawei.com.
PAWS-X is too big so download it [here](https://github.com/google-research-datasets/paws/tree/master/pawsx). If you don't want to wait, you can also download MTOP [here](https://fb.me/mtop_dataset), MTOD [here](https://fb.me/multilingual_task_oriented_data) and MultiATIS++ [here](https://github.com/amazon-research/multiatis) right away.

The directory structure is as follows. Create a **`data/`** folder **OUTSIDE** the project root. For each task, create the following subdirectories: **`m_atis`**, **`mtop`**, **`paws_x`** and **`mtod`**. Now save the downloaded data into each task's directory, creating a subdirectory for each language. 

You are now ready to run the preprocessing code in **`preprocess.py`** :) That will generate the required files and subdirectories. Now, inside the **`data`** folder, there should be four task folders, each with multiple languages/subfolders with the generated files/folders. For example, the German data for MultiATIS++ should be found inside `data/m_atis/de/data.pkl`. That should be that as far as data preparation is concerned.

#### Pretrained Transformers
You will need to download the XLM-R (or other) pretrained model(s). We recommend **HuggingFace** :) The base XLM-R can be downloaded [here](https://huggingface.co/xlm-roberta-base/tree/main) and the large model [here](https://huggingface.co/xlm-roberta-large/tree/main). Save these models _outside_ the project directory (same place as **`data/`**) as **`xlm-roberta-base/`** and **`xlm-roberta-large/`**. Each one should contain at least the following: **`config.json`**, **`pytorch_model.bin`** and **`sentencepiece.bpe.model`**. That should be that for models! 

#### Python Environment
Everything is written in Python 3.7.9 and PyTorch 1.7.0 so install the following packages:
- Install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) for your OS, probably Linux or MacOSX
- Install [PyTorch](https://pytorch.org/get-started/locally/) with conda using something like `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`, using the appropriate cudatoolkit...
- `transformers` version 3.5.0 from HuggingFace (or later) using pip
- The previous libraries should install all other dependencies, no more extra packages should be required

#### Running Experiments

In the **`config`** folder, we saved most setups as shell files that were reported in the paper (though not all of them because we reported lots of numbers/tables). That should get you to reproduce our runs. Here is an example:

Open the command line and type: **`nohup ./config/mtop.sh mtop_aligned large &`** this command will run the **`mtop_aligned`** experiments with XLM-R Large. The base model can be launched by using **`'... base &'`** instead.

The command **`nohup ./config/paws_x.sh paws_x_english base_paws_x &`**  (need to use *base_pawsx* or *large_pawsx* to select a different classifier) will train the base XLM-R on PAWS-X English, for example.

Once you trained an English model for MultiATIS++, for instance, you can type: **`nohup ./config/m_atis.sh m_atis_zero_shot base &`**. This will give you the baseline zero-shot scores for M-ATIS++ for XLM-R base.

Finally, **`nohup ./config/mtod.sh mtod_target large &`** should train the large XLM-R on the labelled data, referred to as 'Target' in the paper.

That should give a good idea for further runs, if unsure, look inside the shell file for clues :)

