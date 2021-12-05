# Revisting Mahalanobis Distance for Transformer-based Out-of-Domain Detection


This repository contains the code to reproduce the results from the paper:
 
["Revisting Mahalanobis Distance for Transformer-based Out-of-Domain Detection"](https://arxiv.org/abs/2101.03778), 
presented on AAAI-21 conference.


## Prerequisites

Clone this repository from github.com.
```shell script
git clone repo URL
```

### Setup environment
1. Create new environment
    ```shell script
    python3 -m venv env
    ```
2. Activate environment
    ```shell script
    source env/bin/activate
    ```
3. Install requirements
    ```shell script
    pip install -r requirements.txt
    ```
   
### Get datasets
1. CLINC150 dataset. Download this dataset from the [repo](https://github.com/clinc/oos-eval)
and place it in the *data/clinc* folder.
   
2. ROSTD dataset. Download this dataset from the 
   [repo](https://github.com/vgtomahawk/LR_GC_OOD/tree/master/code/data/fbreleasecoarse/unsup) 
   and place it in the *data/rostd* folder.
<!--- https://github.com/thuiar/DeepUnkID/tree/master/data/SNIPS -->
3. SNIPS dataset. Download this dataset from the 
   [repo](https://github.com/vgtomahawk/LR_GC_OOD/tree/master/code/data/snips) train.csv, 
   valid.csv and test.csv and place them in the
   *data/rostd* folder. Then run the script 
   *lib/scripts/dataset_preprocess/snips_create_splits.py* to produce splits of the SNIPS datasets.
   The splits will be stored in files 
   *data/snips/snips_<split_name>_<train_size>_<split_version>.csv*, where *split_name* one of
   train, val, test, *split_size* - the proportion of the original data to be used for training,
   *split_version* is the version number, versions differ in the classes considered as in-domain,
   other classes are assumed to be out-of-domain.


## How to run

### MSP (Maximum Softmax Probabiltiy) and Mahalanobis score for BERT-like models

```bash
python scripts/_run_transformer.py
--experiment_name=MLFLOW_EXPERIMENT_NAME --data_name=DATASET_NAME --n_labels=NUM_CLASSES
--hidden_dropout_prob=P_DROPOUT --n_workers=NUM_WORKERS --n_epochs=NUM_EPOCHS
--batch_size=BATCH_SIZE --accumulate_grad_batches=NUM_STEPS_TO_ACCUMULATE
--temperature=SOFTMAX_TEMPERATURE --score_type=SCORE_TYPE --bert_type=BERT_TYPE
--lr=LEARNING_RATE --overfit_pct=OVERFIT_RATIO --train_percent_check=TRAIN_RATIO
--val_percent_check=VAL_RATIO --version=N --balance_classes=IS_BALANCE
--gradient_clip_val=MAGNITUDE --device=CUDA_DEVICE --use_checkpoint=IS_CHECKPOINT
```

#### Arguments
| Argument                  | Default                    | Description                                                                                               |
|---------------------------|----------------------------|-----------------------------------------------------------------------------------------------------------|
| --experiment_name         | debug                      | MLFlow experiment name                                                                                    |
| --data_name               | `None`                     | Dataset name (rostd, rostd-coarse, clinc, snips_75)                                                       |
| --n_labels                | `None`                     | Number of classes in the dataset                                                                          |
| --hidden_dropout_prob     | 0.1                        | Dropout probability in the classifier layer.                                                              |
| --bert_type               | bert-base-uncased          | Name of the BERT-like model  (naming is the same as in transformers)                                      |
| --lr                      | 1e-5                       | Learning rate                                                                                             |
| --batch_size              | 32                         | Batch size                                                                                                |
| --n_workers               | 10                         | Number of workers in the DataLoader                                                                       |
| --n_epochs                | 5                          | Number of epochs to train                                                                                 |
| --overfit_pct             | 0.0                        | How much of training, validation and test data to check                                                   |
| --train_percent_check     | 1.0                        | How much of the training data to check.                                                                   |
| --val_percent_check       | 1.0                        | How much of the validation data to check.                                                                 |
| --temperature             | 1.0                        | Temperature to use in the softmax for evaluation.                                                         |
| --accumulate_grad_batches | 1                          | Number of steps to accumulate gradients.                                                                  |
| --score_type              | max                        | OOD score method: max (msp), mahalanobis, mahalanobis-pca, marginal-mahalanobis, marginal-mahalanobis-pca |
| --data_path               | 'data/clinc/data_full.json | Path to the dataset                                                                                       |
| --device                  | ''                         | CUDA device to use. Empty string - CPU will be used.                                                      |
| --ood_type                | `None`                     | Type of the OOD data to be used in the SST-2 experiments. (wmt16, rte or snli).                           |
| --version                 | `None`                     | Split variant to use in SNIPS experiments.                                                                |
| --balance_classes         | False                      | Whether to balance classes in the training.                                                               |
| --gradient_clip_val       | 0.                         | Gradient clipping value  (if 0 then no clipping is performed)                                             |
| --use_checkpoint          | False                      | Whether to store final model in the filesystem.                                                           |

### MSP (Maximum Softmax Probabiltiy) and Mahalanobis score for CNN, LSTM and CBOW models.

```bash
python scripts/_run_transformer.py hydra.run.dir=.
```
This scirpt runs MSP method training and  a subsequent evaluation with default configuration
*../configs/config_msp_cnn.yaml*. By specifying *classifier=MODEL_NAME*, one can choose the model.
For available models, look in the folder *configs/classifier*.



### Likelihood ratio
```bash
python scripts/_likelihood_ratio.py hydra.run.dir=.
```
This script runs the LLR method training and a subsequent evaluation with default 
configuration given in the *../configs/config_likelihood_clinc.yaml*. This configuration
uses CLINC dataset, the other two configurations are suitable for ROSTD and SNIPS datasets.
This configuration files contains the best hyperparameters for each of the datasets found
by hyperparameter search. To run other config rewrite *config_path* in *_run_likelihood_ratio.py*. 


## Hyperparmeters
We use LSTMs as the main and the background model for the LLR approach. 
The LSTMs hyperparameters are listed in the configuration files provided with the code. 
In our experiments using Transformer-based models did not show any improvements.

For experiments with MSP, we use the similar architecture of CNN provided in 
[Zheng er al.](https://arxiv.org/abs/1909.03862). 
We found that this architecture with the Mahalanobis distance score 
performs poorly, mainly because of the dropout layers. Our experiments show that the 
substitution of dropout layers with batch normalization layers leads to substantial 
improvement in OOD metrics (the accuracy on the ID examples are comparable) for CNN 
with the Mahalanobis OOD score.

For SST-2 setup, we used CNN with filters of sizes [2, 3, 4, 5] 
with 128 channels. Then we use pooling over time and concatenate features obtained from 
the filters of different sizes. This gives the sentence embeddings of size 512. On top of 
these features, a dropout and a linear layer are used, mapping embedding to class logits. 
This architecture follows the one used in [Hendrycks er al.](https://arxiv.org/abs/2004.06100).

Transformer-based (BERT, RoBERTa, etc) OOD detectors consist of the Transformer model 
itself and a dropout and a linear layer on top of the embedding for [CLS] token.

### Hyperparameter search
We performed hyperparameter search using the following grid for Transformer models
  - Learning rate: [1e-5, 2e-5, 3e-5, 5e-5]
  - Batch size: [16, 32]

We selected the best model by $AUPR_{OOD}$ metric. 
For each hyperparameter configuration, we averaged the results over 10 runs.

We searched for the best LLR model over the following grid
   - Learning rate: [1-3, 5e-4, 1e-4, 5e-5, 1e-5]
   - Main model LSTM hidde size: [256, 300, 512]
   - Background model LSTM hidden size: [64, 128, 256, 512]
   - Background model $L_{2}$-regularization coefficient: 
     [1e-4$, 1e-3, 1e-2, 1e-1, 0, 1, 10, 100]
   - Input dropout: [0, 0.3, 0.5]
   - Output dropout: [0, 0.3, 0.5]
   - Batch size: [16, 32, 64, 128]

The procedure for selecting the model is the same as for transformers.

## Computing infrastructure
Each experiment was done using a single NVIDIA GeForce RTX 2080 GPU, requiring no longer than 2 hours.
