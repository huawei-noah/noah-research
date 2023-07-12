## Curriculum Learning for LMs and XLMs

Code for the EMNLP 2022 paper 
"[Training Dynamics for Curriculum Learning: A Study on Monolingual and Cross-lingual NLU](https://aclanthology.org/2022.emnlp-main.167/)"

If you like this work, please cite the publication as follows:
@inproceedings{christopoulou-etal-2022-training,
    title = "Training Dynamics for Curriculum Learning: A Study on Monolingual and Cross-lingual {NLU}",
    author = "Christopoulou, Fenia  and
      Lampouras, Gerasimos  and
      Iacobacci, Ignacio",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.167",
    pages = "2595--2611"
}


### Environment
All models were training on a single Nvidia V100 16GB GPU.
Pending to provide a docker image the will help replicate experiments.


### Pre-trained Language Models

XLM-R-base and RoBERTa-base language models from the HuggingFace hub ([xlmr-base](https://huggingface.co/xlm-roberta-base),
[roberta-base](https://huggingface.co/roberta-base) will be downloaded from [HuggingFace](https://huggingface.co/models)
automatically during the first time of training.


### Datasets

Download the datasets used following instructions inside the `get_data.sh` script.
```
cd original_data/
bash get_data.sh
```

Before pre-processing, some data need minor modifications:
```bash
cd datasets/

# NLI diagnostics
python edit_diagnostics.py -i ../../original_data/diagnostics-full.tsv \
                           -o ../../data/diagnostics-full-modified.tsv

# COPA test set
python make_copa.py -i ../../original_data/COPA-resources/datasets/copa-test.xml \
                    -o ../../original_data/COPA/test.jsonl

# MLDoc
# clone the MLDoc repo as the generate_documents.py script is required
sh convert_data_mldoc.sh ../original_data/MLDoc

# Adversarial SQuAD
python convert_advsquad.py -i ../original_data/squad_adversarial \
                           -o ../data/squad_adversarial_converted
```

Data can then be processed choosing one of `['xnli', 'pawsx', 'mldoc', 'xcopa', 'siqa', 'paws', 'twitter-ppdb', 'qnli', 'rte', 'mnli']`.
All data will be saved in the `data/` directory in an appropriate format for 
[HuggingFace Datasets](https://huggingface.co/docs/datasets/) to load.
An example is shown below:
```bash
cd datasets/
python data_preprocess.py --dataset_name xnli \
                          --data_dir ../../original_data/xtreme/download/ \
                          --out_data_dir ../../data/
```


### Training

First, fine-tune a model on one of the datasets and collect training dynamics.
```bash
cd exps_bash/
bash run_baseline.sh xnli xlm-roberta-base 7e-6 10
bash run_baseline.sh xnli roberta-base 7e-6 10
```
More info about those can be found in `exps_bash/run_all.sh`.


#### Collecting Training Dynamics & Difficulty Scores

The previous models should save training dynamics for each epoch inside a 
`saved_models/experiment_name/dataset_epochX.json` file.
Run the following to get:
- Training dynamics difficulty scores (confidence, variability, correctness)
- Heuristics difficulty scores (sentence length, word rarity)
- Perplexity difficulty score
- Cross-Review score

```bash
cd helpers/
python calculate_dynamics.py --model_dir ../../trained_models/folder_where_the_model_is_saved/ \
                             --output_dir ../../dynamics/ \
                             --data_dir ../../data/ \
                             --dataset_name xnli \
                             --analysis \
                             --epochs 0-10
                             
python calculate_heuristics.py --model_name roberta-base \
                               --output_dir ../../dynamics/ \
                               --data_dir ../../data/ \
                               --dataset_name xnli
                           
python calculate_ppl.py --model_name roberta-base \
                        --output_dir ../../dynamics/ \
                        --data_dir ../../data/ \
                        --dataset_name xnli    
```

Collect the cross-review difficulty metrics by running the following:
```bash
python sharding.py --input_dataset_dir ../../data/xnli/ \
                   --output_dataset_dir ../../data_shards/xnli/ \
                   --shards_num 10     
                   
cd ../exps_bash/
bash run_shards.sh xnli 'xlm-roberta-base' 7e-6 10  
bash eval_shards.sh xnli 'xlm-roberta-base' 7e-6 10 32 128 10

cd ../helpers/ 
python get_cr_scores.py --model_name roberta-base \
                        --lr 7e-6 \
                        --len 128 \
                        --bs 32 \
                        --eps 10 \
                        --shards 10 \
                        --seed 123 \
                        --dataset_name xnli \
                        --input_dir ../../trained_models_shards/                                             
```

#### Training with Curricula

Using the training dynamics the following curricula can be used to re-train a model: 
`annealing`, `annealing-bias`, `competence`, `competence-bias`, `cross-review` and heuristics including `length`, `word rarity` and `perplexity`.

Examples for running a model with CL:
```bash
cd exps_bash/
bash run_annealing.sh xnli 'xlm-roberta-base' 7e-6 10 32 128
bash run_cross_review.sh xnli 'xlm-roberta-base' 7e-6 10 32 128
bash run_competence.sh xnli 'xlm-roberta-base' 7e-6 10 32 128 "71500 107000 71500"
bash run_heuristics.sh xnli 'xlm-roberta-base' 7e-6 10 32 128 "71500 107000 71500"
```
The `confidence-based` and `heuristics-based` curricula use the competence scheduler. 
Hence, they require to know the number of training steps in advance.
For each seed you need to give an integer corresponding to the numbers of total steps the teacher model has been trained on.
This is printed at the end of the baseline model training.


### Evaluation

Models (with and without curricula) can be evaluated on different test sets, using the evaluation script.
```bash
python evaluate.py --model_dir ../trained_models/xnli_xlm-roberta-base_123_LR7e-5_LEN128_BS32_E10_annealing_xnli_xlm-roberta-base_123_LR7e-5_LEN128_BS32_E10-correctness/ \
                   --model_name roberta-base \
                   --curriculum_steps 71500 107000 71500 \
                   --dataset_name nli-diagnostics
```

## License

Licensed under the Apache License, Version 2.0. Please see the [License](./LICENSE) file for more information.
Disclaimer: This is not an officially supported HUAWEI product.
