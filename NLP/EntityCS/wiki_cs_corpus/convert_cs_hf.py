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

import os
import argparse
import json
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cs_sent_folder",
        type=str,
        default="data_by_lang_v7",
        help="folder for en_cs sentences in json",
    )
    parser.add_argument(
        "--en_sents",
        type=str,
        default="en_sentences.json",
        help="EN sentences in json",
    )
    parser.add_argument(
        "--output_en_folder",
        type=str,
        default="huggingface_datasets_en_v7",
        help="folder for EN sentences",
    )
    parser.add_argument(
        "--output_non_en_folder",
        type=str,
        default="huggingface_datasets_no_en_v7",
        help="folder for non-EN sentences",
    )
    args = parser.parse_args()
    os.makedirs(args.output_non_en_folder, exist_ok=False)
    os.makedirs(args.output_en_folder, exist_ok=False)

    for file in os.listdir(args.cs_sent_folder):
        lang = file.split('_')[-1].split('.')[0]

        dataset = load_dataset('json', data_files=os.path.join(args.cs_sent_folder, file))
        dataset.save_to_disk(os.path.join(args.output_non_en_folder, lang))

    en_dataset = load_dataset('json', data_files=args.en_sents)
    en_dataset.save_to_disk(os.path.join(args.output_en_folder, 'en'))


if __name__ == "__main__":
    main()
