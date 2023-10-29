# EntityCS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching

Official implementation for the EMNLP 2022 Findings paper 
["EntityCS: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching"](https://aclanthology.org/2022.findings-emnlp.499/).

If you like this work or plan to use it, please cite the publication as follows:
```html
@inproceedings{whitehouse-etal-2022-entitycs,
    title = "{E}ntity{CS}: Improving Zero-Shot Cross-lingual Transfer with Entity-Centric Code Switching",
    author = "Whitehouse, Chenxi  and
      Christopoulou, Fenia  and
      Iacobacci, Ignacio",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.499",
    pages = "6698--6714"
}
```

## Environment Setup
The best way to get started is to clone the repository, create a new conda environment and install the necessary dependencies:
```bash
conda create --name entity_cs python=3.8
conda activate entity_cs
pip install -r requirements.txt
```

## Corpus Construction

The corpus used in the experiments can be found here: (pending)
Alternatively, you can create the corpus from scratch, making your own modifications via the following process.
First, download the latest English Wikipedia articles from [here](https://dumps.wikimedia.org/enwiki/latest/), and use the modified [wikiextractor](https://github.com/attardi/wikiextractor)  (**version 3.0.6**) to extract text from xml files while keeping the `[[` and `]]` around the entities.

```bash
cd wiki_cs_corpus
git clone https://github.com/attardi/wikiextractor.git
```
To modify the extracted Wikipedia articles as needed, add the code of `wiki_cs_corpus/extract_hyperlinks.py` inside `wiki_cs_corpus/wikiextractor/wikiextractor/extract.py`,
and replace:  
- line 98: `text = replaceInternalLinksCustomised(text)`  
- Line 801: `keepSections = False`  

Follow instructions from the [official wikiextractor](https://github.com/attardi/wikiextractor) to produce text from wiki pages, which will be saved in `wiki_cs_corpus/wikiextractor/text`.

Use [spaCy](https://spacy.io/) to segment sentences and remove those without entities or containing more than 128 tokens.
```bash
python sentenizer.py
python post_process_sents.py
```

Then download Wikidata json file [here](https://dumps.wikimedia.org/wikidatawiki/entities/) and build a local lookup dictionary 
for mapping Wikidata ID to entities in available languages.
```bash
python create_wikidata_db.py
```
This will generate a `shelf` dictionary (takes ~1 day to generate and results in ~55GB size).

Next, define target language candidates for code-switching (CS) and number of code-switched sentences per English sentence.
The following commands use a modified version of [wikimapper](https://github.com/jcklie/wikimapper) to: (1) map entities (including redirects) to their 
corresponding Wikidata ID, (2) get available translations for each entity, (3) select target languages for CS and (4) produce the final code-switched corpus.
Install [wikimapper](https://github.com/jcklie/wikimapper) following the instructions and then run:
```bash
python wiki_cs_db.py
```
This will generate a `en_sentences.json` file with English sentences in the following format: 
```
{
  'id': 19, 
  'en_sentence': 'The subs then enter a <en>coral reef</en> with many bright reflective colors.'
}
```

And a directory containing the EN and code-switched sentence for each target language:
```
{
  'id': 19, 
  'en_sentence': 'The subs then enter a <en>coral reef</en> with many bright reflective colors.', 
  'cs_sentence': 'The subs then enter a <de>Korallenriff</de> with many bright reflective colors.', 
  'language': 'de'
}
```

Finally, we convert the corpus to arrow format and save it locally. 
```bash
python convert_cs_hf.py
```


## Intermediate Training
To continue training [XLM-R-base](https://huggingface.co/xlm-roberta-base) (or another model of your choice)
on various masking strategies/training objectives described in the paper, run the following:
```bash
cd cs_pretrain/ && bash run_lm.sh
```
For different masking strategies, set the parameters in `cs_pretrain/run_lm.sh` accordingly.
Also remove `torch.distributed.launch` in case you do not want to run across multiple gpus.
See the table below for the different configurations reported in the paper:

| Model                  | MASKING | ENTITY_PROBABILITY | PARTIAL_MASKING | KEEP RANDOM | KEEP_SAME |
|:-----------------------|:--------|:-------------------|:----------------|:------------|:----------|
| MLM                    | mlm     | 0.0                | False           | False       | False     |
| WEP                    | ep      | 1.0                | False           | False       | True      |
| PEP<sub>MRS</sub>      | ep      | 1.0                | True            | True        | True      |
| PEP<sub>MS</sub>       | ep      | 1.0                | True            | False       | True      |
| PEP<sub>M</sub>        | ep      | 1.0                | True            | False       | False     |
| PEP<sub>MS</sub> + MLM | ep_mlm  | 0.5                | True            | False       | True      |
| WEP + MLM              | ep_mlm  | 0.5                | False           | False       | True      |


## Evaluation on Downstream Tasks

### NER
Use the following script to download the target task dataset from the HF hub and convert as follows:
```bash
cd ner/ && python convert_wikiann_dataset.py
```

Change the parameters in `ner/run_wikiann.sh` if you want to run another model, and run the following:
```bash
cd ner/
bash run_wikiann.sh
```

### Slot Filling
We evaluate and [MultiATIS++](https://github.com/amazon-research/multiatis) and [MTOP](https://fb.me/mtop_dataset) on slot filling. 
Obtain the original datasets and save them in the `downloaded_datasets/` directory.
Then convert them to the Huggingface datasets format. 
```bash
cd slot_filling/ && python convert_matis_dataset.py
python convert_mtop_dataset.py
```

For slot filling-only training, run:
```bash
cd slot_filling/
bash run_mtop_joint.sh
bash run_mtop.sh
```

For joint slot-filling and intent classification training, run:
```bash
cd slot_filling/
bash run_matis_joint.sh
bash run_mtop_joint.sh
```

### Fact Retrieval
The X-FACTR evaluation framework was cloned from [the original repo](https://github.com/jzbjyb/X-FACTR) 
with a minimal adaptation to the `X-FACTR/scripts/probe.py` script, to work with Transformers version `4.15.0`:
```python
# at the end of the script
from transformers import AutoConfig, AutoModelForMaskedLM
print('load model')
config = AutoConfig.from_pretrained(
    LM,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1
)
model = AutoModelForMaskedLM.from_pretrained(LM, config=config)
```

Follow the instructions in the [X-FACTR repo](https://github.com/jzbjyb/X-FACTR) and install additional libraries required inside a conda environment.
Edit `requirements.txt` file to include newer torch and transformers versions.
```bash
torch==1.8.0
transformers==4.15.0
```
Then:
```bash
conda create -n xfactr -y python=3.7 && conda activate xfactr && bash setup.sh
```

Place everything inside the `X-FACTR/` directory, then, run the following to get both `confidence-based` and `independent` decoding predictions:
```bash
cd X-FACTR/
bash run_xfactr.sh  # Run this to probe the model first
python collect_predictions.py path_to_model path_do_a_prefictions_dir conf
python collect_predictions.py path_to_model path_do_a_prefictions_dir ind
```

### Word Sense Disambiguation
Download the English dataset from [here](https://dl.fbaipublicfiles.com/glue/superglue/data/v2/WiC.zip) and save into the `downloaded_datasets/` directory.
Use the conversion script to convert it to arrow format.
You will also need to download [XL-WiC](https://pilehvar.github.io/xlwic/), the multilingual version of the WiC dataset.
The model used for fine-tuning can be found in `wsd/model.py`.
```bash
cd wsd/
python convert_xlwic_dataset.py
```

Run the model as follows:
```bash
bash run_xl_wic.sh
```

## EntityCS Corpus
We provide the EntityCS corpus in all 93 languages on the HF Hub: 
[https://huggingface.co/datasets/huawei-noah/entity_cs](https://huggingface.co/datasets/huawei-noah/entity_cs)

Code-switched instances in different languages are associated with the original English sentence, containing two fields `en_sentence` and
`cs_sentence`. For the English instances (`en`), only the `en_sentence` field exists.

Example usage:
```python
import datasets

for lang in ['en', 'fr', 'es']:
    d = datasets.load_dataset("huawei-noah/entity_cs", data_dir=f"data/{lang}/", split="train")
    print(d)
```

## Trained Models on the EntityCS corpus
We provide 4 models based on different objectives, trained on the EntityCS corpus with 39 languages (those of WikiANN) 
and achieve the lowest perplexity score on the validation set:
- MLM: [https://huggingface.co/huawei-noah/EntityCS-39-MLM-xlmr-base](https://huggingface.co/huawei-noah/EntityCS-39-MLM-xlmr-base)
- WEP: [https://huggingface.co/huawei-noah/EntityCS-39-WEP-xlmr-base](https://huggingface.co/huawei-noah/EntityCS-39-WEP-xlmr-base)
- PEP<sub>MS</sub>: [https://huggingface.co/huawei-noah/EntityCS-39-PEP_MS-xlmr-base](https://huggingface.co/huawei-noah/EntityCS-39-PEP_MS-xlmr-base)
- PEP<sub>MS</sub> + MLM: [https://huggingface.co/huawei-noah/EntityCS-39-PEP_MS_MLM-xlmr-base](https://huggingface.co/huawei-noah/EntityCS-39-PEP_MS_MLM-xlmr-base)
  
## License
Licensed under the Apache License, Version 2.0. Please see the [License](./LICENSE) file for more information.
Disclaimer: This is not an officially supported HUAWEI product.

【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】