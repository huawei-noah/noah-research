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
import datasets
from glob import glob
import pandas as pd
import ast


def main():

    final_dataset = {}
    unique_labels = {}
    for language_folder in glob(os.path.join("../downloaded_datasets/mtop", "*")):
        if language_folder.endswith(".txt"):
            continue
        print(language_folder)
        lang_name = language_folder.split("/")[-1]

        final_dataset[lang_name] = {}
        for split in ["train", "eval", "test"]:
            tmp = []
            with open(os.path.join(language_folder, split + ".txt"), "r") as infile:
                for line in infile:
                    line = line.rstrip().split("\t")
                    # print(line)
                    id_ = line[0]
                    intent = line[1].split(":")[-1]
                    slots = line[2]
                    dict_ = ast.literal_eval(line[7])

                    tokens = dict_["tokens"]
                    token_spans = dict_["tokenSpans"]
                    slot_tags = []

                    spans = {}
                    if slots:
                        if slots.endswith(","):
                            slots = slots[:-1]
                        if "，" in slots:
                            slots = slots.replace("，", ",")

                        for s in slots.split(","):
                            start, end, special, tag = s.split(":")
                            spans[(start, end, tag)] = []

                    for token, span in zip(tokens, token_spans):

                        added = False

                        if slots:
                            for s in spans:
                                start, end, tag = s

                                if int(start) == span["start"]:
                                    slot_tags.append(f"B-{tag}")
                                    unique_labels[f"B-{tag}"] = 1
                                    added = True
                                    spans[(start, end, tag)] += [token]

                                elif (
                                    int(start)
                                    < span["start"] + span["length"]
                                    <= int(end)
                                ):
                                    slot_tags.append(f"I-{tag}")
                                    unique_labels[f"I-{tag}"] = 1
                                    added = True
                                    spans[(start, end, tag)] += [token]

                            if not added:
                                slot_tags.append("O")
                                unique_labels["O"] = 1
                        else:
                            slot_tags.append("O")

                    spans = [f"{k[2]}: {' '.join(v)}" for k, v in spans.items()]

                    assert len(tokens) == len(
                        slot_tags
                    ), f"ERROR: {tokens} <> {slot_tags} <> {slots}"
                    tmp.append(
                        {
                            "tokens": tokens,
                            "slot_tags": slot_tags,
                            "langs": [lang_name] * len(tokens),
                            "spans": ",".join(spans),
                            "intent": intent,
                        }
                    )

            if split == "eval":
                split = "validation"

            final_dataset[lang_name][split] = datasets.Dataset.from_pandas(
                pd.DataFrame(tmp)
            )
            print(final_dataset[lang_name][split])
        final_dataset[lang_name] = datasets.DatasetDict(final_dataset[lang_name])

    for lang in final_dataset:
        print(final_dataset[lang])
        final_dataset[lang].save_to_disk(f"../data/mtop/{lang}")


if __name__ == "__main__":
    main()
