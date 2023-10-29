# CrossAligner & Co: Zero-Shot Transfer Methods for Task-Oriented Cross-lingual Natural Language Understanding
This is the data/code repository and usage instructions for the above [ACL 2022 paper](https://arxiv.org/abs/2203.09982v1). If you find the resources/paper useful, please cite:

```
@article{gritta2022crossaligner,
  title={CrossAligner \& Co: Zero-Shot Transfer Methods for Task-Oriented Cross-lingual Natural Language Understanding},
  author={Gritta, Milan and Hu, Ruoyu and Iacobacci, Ignacio},
  journal={arXiv preprint arXiv:2203.09982},
  year={2022}
}
```

#### Acknowledgements
The starting point for this repo was cloned from one of our previous [papers](https://aclanthology.org/2021.findings-acl.32/) called [XeroAlign](https://github.com/huawei-noah/noah-research/tree/master/xero_align).

### Getting started
**`git clone`** the project first, then you can set up data, models and runs.

#### Python Environment
Everything is written in Python 3.7.9 and PyTorch 1.7.0 so install the following packages:
- Install [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) with Python 3.7.9 for Linux or MacOSX
- Install [PyTorch](https://pytorch.org/get-started/locally/) 1.7.0 with conda using something like `conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch` and select the appropriate **cudatoolkit** for your GPU.
- Use the `requirements.txt` with **conda** to install every single package
- That should be that, no more extra packages required

#### Pretrained Transformers
You need to download the XLM-R pretrained model from **HuggingFace**. The base XLM-R can be downloaded [here](https://huggingface.co/xlm-roberta-base/tree/main) and the large model [here](https://huggingface.co/xlm-roberta-large/tree/main). Save these models **_outside_** the project directory as they will be loaded using these paths: **`../xlm-roberta-base/`** and **`../xlm-roberta-large/`** from the project (root) directory. Each model directory should contain at least the following: **`config.json`**, **`pytorch_model.bin`** and **`sentencepiece.bpe.model`**. That should be that for models! 

#### Datasets
The XNLU datasets can be downloaded from [this Github repository](https://github.com/milangritta/Datasets) (with some very minor corrections for MultiATIS++, see [XeroAlign paper](https://aclanthology.org/2021.findings-acl.32/)). Next, **place the [downloaded](https://github.com/milangritta/Datasets) zip files into the ```data``` folder**. By the way, the original data downloads can be obtained here: [MTOP](https://fb.me/mtop_dataset), [MTOD](https://fb.me/multilingual_task_oriented_data) and [MultiATIS++](https://github.com/amazon-research/multiatis).

**The next command assumes you have saved a tokenizer in ```../xlm-roberta-large/``` folder**. Run the preprocessing script like this: **`python preprocess.py`**. This will generate the required files and subdirectories. Inside the **`data`** folder, there should be four task folders, each with multiple languages/subfolders with the generated files and folders.

#### Running Experiments

In the **`config`** folder, we saved the experiments as shell files that were reported in the paper. That should help you reproduce our results. Here is an example:

Open the command line and type: **`nohup ./config/mtop.sh mtop_aligned large &`** this command will run the **`mtop_aligned`** experiments with XLM-R Large. The base model can be launched by using **`'... base &'`** instead of `... large &`.

Once you trained an English model for MultiATIS++, for instance, you can type: **`./config/m_atis.sh m_atis_zero_shot base`** to obtain the baseline zero-shot scores for MultiATIS++.

Finally, **`nohup ./config/mtod.sh mtod_target large &`** will train the large XLM-R on the labelled data, referred to as 'Target' in the paper.

To specify which alignment methods to use during training, set the flag **` --use_aux_losses `** to any combination of **` CA,XA,CTR,TI `** (comma-separated, no spaces). To use Coefficient of Variation ([Groenendijk et al. 2020](https://arxiv.org/pdf/2009.01717.pdf)) weighting, set the **` --use_weighting `** flag to **` COV `**. Otherwise 1+1 weighting will be used if weighting method is not specified.

TOP TIP: Should some runs degrade to zero accuracy and zero F-Score, just decrease the learning rate a bit. Some languages are more sensitive to higher learning rates than others (usually happens quite rarely).

That should give a good idea for launching further runs, if unsure, look inside the shell file for hints :)

#### Additional Notes

Here are some more notes to get you started with running experiments with auxiliary losses.

#### 1. Selecting Auxiliary Losses
When running alignment tasks (`m_atis_aligned`, `mtop_aligned`, `mtod_aligned`), set the `--use_aux_losses` flag to a list of auxiliary losses you would like to use.

For example the following code in `config/m_atis.sh` would run the experiments with XeroAlign and CrossAligner as auxiliary losses weighted using CoV weighting.
```angular2html
if [$1 == "m_atis_aligned"]
then
    for lang in de es tr zh hi fr ja pt
    do
        python main.py --task m_atis \
        ...
        --max_seq_len 100 \
        --use_aux_losses XA CA \
        --use_weighting COV
    done
fi
```

To use 1+1 weighting instead, comment/delete the line, it will default to 1+1 as loss weighted are initialised to 1 and not specifying a method will leave it unaltered.
```angular2html
# --use_weighting COV
```

For what parameters corresponds to each loss, see the dictionary in `utils.py`

#### 2. Adding Auxiliary Losses
New auxiliary losses can be added in `train.py`. For example, see XeroAlign, CrossAligner, etc.

To enable new aux losses on the command line, add your new loss to the `choices` for the `use_aux_losses` option of the parser in `main.py`, **and also** add the new loss as a (loss, flag) pair in the `set_aux_losses(args)` function in `utils.py`

For example, to implement my new loss `my_new_loss` add the following to `utils.py`

```
def set_aux_losses(args):
    ...
    loss_keys = {
        ...
        "NL":  "use_new_loss",
    }
```

The new loss could then be implemented in `train.py`
```angular2html
...
if use_losses.use_new_loss:
    # Implement new auxiliary alignment loss here
...
```

#### 3. Adding Weighting Methods
New loss weighting methods can also be added in `train.py` in a manner similar to auxiliary losses. As of the time of writing, the weighting method parser does not use `choices`, thus only `set_weighting_method(args)` in `utils.py` needs to be updated.
```
def set_weighting_method(args):
    weighting_methods = {
        ...
        "NW": "use_new_weighting_method",
    }
    ...
```
And the implement the weighting method in `train.py`
```angular2html
if use_weighting.use_new_weighting_method:
    # Implement new weighting scheme here
    loss_weights = ...
    ...
```

We hope you find our resources useful. Get in touch if you need further help :)

【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】