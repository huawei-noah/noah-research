# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# PAWS-X & XNLI
git clone https://github.com/google-research/xtreme
cd xtreme/
bash install_tools.sh
bash scripts/download_data.sh
conda deactivate

# PAWS
wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
tar -xzvf paws_wiki_labeled_final.tar.gz
mv final/ paws/

# Twitter-PPDB: [source](https://languagenet.github.io/)

# SIQA
wget https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip
unzip socialiqa-train-dev.zip

# COPA
wget https://people.ict.usc.edu/~gordon/downloads/COPA-resources.tgz
tar -xzvf COPA-resources.tgz
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/COPA.zip
unzip COPA.zip

# XCOPA:
git clone https://github.com/cambridgeltl/xcopa.git

# MLDoc
# https://github.com/facebookresearch/MLDoc) (you will need appropriate license for this)

# QNLI / RTE / MNLI-M
python -c "import datasets; dataset = datasets.load_dataset('glue', 'rte'); print(dataset); dataset.save_to_disk('rte')"
python -c "import datasets; dataset = datasets.load_dataset('glue', 'qnli'); print(dataset); dataset.save_to_disk('qnli')"
python -c "import datasets; dataset = datasets.load_dataset('glue', 'mnli'); print(dataset); dataset.save_to_disk('mnli')"

# Commonsense QA
python -c "import datasets; dataset = datasets.load_dataset('commonsense_qa'); print(dataset); dataset.save_to_disk('../data/csqa')"

# HANS
python -c "import datasets; dataset = datasets.load_dataset('hans'); print(dataset); dataset.save_to_disk('../data/hans')"

# NLI Diagnostics (https://super.gluebenchmark.com/diagnostics)
wget "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv" -O diagnostics-full.tsv

# Adversarial SQuAD
python -c """
import datasets
dataset_1 = datasets.load_dataset('squad_adversarial', 'AddSent')
print(dataset_1)
dataset_2 = datasets.load_dataset('squad_adversarial', 'AddOneSent')
print(dataset_2)
dataset = datasets.concatenate_datasets([dataset_1['validation'], dataset_2['validation']])
print(dataset)
dataset.save_to_disk('squad_adversarial')
"""
