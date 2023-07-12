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
import pandas as pd
from datasets import load_dataset


def main():
    tsv_file_dir = "../downloaded_datasets/MultiATIS++.v0.1/data/train_dev_test"
    csv_file_dir = "../downloaded_datasets/MultiATIS++.v0.1/data/csv_data_multiatis"
    os.makedirs(csv_file_dir, exist_ok=False)
    files = os.listdir(tsv_file_dir)
    for file in files:
        cvs_file = pd.read_table(os.path.join(tsv_file_dir, file), sep="\t")
        cvs_file.to_csv(
            os.path.join(csv_file_dir, file.split(".")[0] + ".csv"), index=False
        )

    LANGS = ["EN", "ES", "DE", "HI", "FR", "PT", "ZH", "JA", "TR"]
    for l in LANGS:
        data_files = {
            "train": os.path.join(csv_file_dir, "train_" + l + ".csv"),
            "validation": os.path.join(csv_file_dir, "train_" + l + ".csv"),
            "test": os.path.join(csv_file_dir, "test_" + l + ".csv"),
        }
        data = load_dataset("csv", data_files=data_files)
        print(l)
        print(data)
        for k in data_files.keys():
            data[k] = data[k].filter(
                lambda ex: len(ex["utterance"].split(" "))
                == len(ex["slot_labels"].split(" "))
            )
        print("After removing un-matching examples")
        print(data)

        for k in data_files.keys():
            data[k] = data[k].rename_column("utterance", "tokens")
            data[k] = data[k].rename_column("slot_labels", "slot_tags")

        def modify_dataset(example):
            example["langs"] = [l.lower()] * len(example["tokens"].split(" "))
            example["tokens"] = example["tokens"].split(" ")
            example["slot_tags"] = example["slot_tags"].split(" ")
            assert len(example["tokens"]) == len(example["slot_tags"])
            return example

        data = data.map(modify_dataset)
        print(data)

        data.save_to_disk(f"../data/multiatis/{l.lower()}")
        print(f"done for {l}")


if __name__ == "__main__":
    main()
