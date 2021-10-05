# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import torch
from torch.utils import data
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import csv

ROOT = "data/"


class Dataset(data.Dataset):
    def __init__(self, tokenizer, df, max_length=512):
        pass

    @classmethod
    def load(cls):
        pass

    @classmethod
    def get_n_labels(cls):
        pass

    def collate_fn(self, batch):
        batch = list(zip(*batch))

        out = []

        for part in batch:
            xs = list(zip(*part))

            xs = [
                pad_sequence(
                    x, batch_first=True, padding_value=self.tokenizer.pad_token_id
                )
                if len(x[0].shape) > 0
                else torch.stack(x, 0)
                for x in xs
            ]
            out.append(xs)

        return out


class QNLIDataset(Dataset):
    @classmethod
    def load(cls):
        train_data = pd.read_csv(
            f"{ROOT}QNLI/train.tsv", sep="\t", quoting=csv.QUOTE_NONE
        )
        val_data = pd.read_csv(f"{ROOT}QNLI/dev.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        test_data = pd.read_csv(
            f"{ROOT}QNLI/test.tsv", sep="\t", quoting=csv.QUOTE_NONE
        )

        for df in [train_data, val_data]:
            df["label"] = df["label"] == "entailment"

        return train_data, val_data, test_data

    def __init__(self, tokenizer, df, max_length=512):
        self.tokenizer = tokenizer
        self.data = []

        for i, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row["question"],
                row["sentence"],
                max_length=max_length,
                truncation="longest_first",
            )

            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": row["label"] if "label" in row else None,
                }
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_n_labels(self):
        return 1

    def __getitem__(self, idx):
        output = [
            torch.tensor(self.data[idx]["input_ids"]),
            torch.tensor(self.data[idx]["attention_mask"]),
        ]

        if self.data[idx]["label"] is not None:
            label = torch.tensor(self.data[idx]["label"]).long()
            output.append(label)

            return (output, [label])

        return (output,)


class RTEDataset(Dataset):
    @classmethod
    def load(cls):
        train_data = pd.read_json(
            f"{ROOT}RTE/train.jsonl", lines=True, orient="records",
        )
        val_data = pd.read_json(f"{ROOT}RTE/val.jsonl", lines=True, orient="records",)
        test_data = pd.read_json(f"{ROOT}RTE/test.jsonl", lines=True, orient="records",)

        for df in [train_data, val_data]:
            df["label"] = df["label"] == "entailment"

        return train_data, val_data, test_data

    def __init__(self, tokenizer, df, max_length=512):
        self.tokenizer = tokenizer
        self.data = []

        for i, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row["premise"],
                row["hypothesis"],
                max_length=max_length,
                truncation="longest_first",
            )

            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": row["label"] if "label" in row else None,
                }
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_n_labels(self):
        return 1

    def __getitem__(self, idx):
        output = [
            torch.tensor(self.data[idx]["input_ids"]),
            torch.tensor(self.data[idx]["attention_mask"]),
        ]

        if self.data[idx]["label"] is not None:
            label = torch.tensor(self.data[idx]["label"]).long()
            output.append(label)

            return (output, [label])

        return (output,)


class CBDataset(Dataset):
    @classmethod
    def load(cls):
        train_data = pd.read_json(
            f"{ROOT}CB/train.jsonl", lines=True, orient="records",
        )
        val_data = pd.read_json(f"{ROOT}CB/val.jsonl", lines=True, orient="records",)
        test_data = pd.read_json(f"{ROOT}CB/test.jsonl", lines=True, orient="records",)

        for df in [train_data, val_data]:
            df["label"] = df["label"].apply(
                lambda x: {"contradiction": 0, "entailment": 1, "neutral": 2}[x]
            )

        return train_data, val_data, test_data

    def __init__(self, tokenizer, df, max_length=512):
        self.tokenizer = tokenizer
        self.data = []

        for i, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row["premise"],
                row["hypothesis"],
                max_length=max_length,
                truncation="longest_first",
            )

            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": row["label"] if "label" in row else None,
                }
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_n_labels(self):
        return 3

    def __getitem__(self, idx):
        output = [
            torch.tensor(self.data[idx]["input_ids"]),
            torch.tensor(self.data[idx]["attention_mask"]),
        ]

        if self.data[idx]["label"] is not None:
            label = torch.tensor(self.data[idx]["label"]).long()
            output.append(label)

            return (output, [label])

        return (output,)


class FullANLIDataset(Dataset):
    @classmethod
    def load(cls):
        train_data = pd.read_json(
            f"{ROOT}anli_v1.0/train.jsonl", lines=True, orient="records",
        )
        val_data = pd.read_json(
            f"{ROOT}anli_v1.0/dev.jsonl", lines=True, orient="records",
        )
        test_data = pd.read_json(
            f"{ROOT}anli_v1.0/test.jsonl", lines=True, orient="records",
        )

        for df in [train_data, val_data, test_data]:
            df["label"] = df["label"].apply(lambda x: {"c": 0, "e": 1, "n": 2}[x])

        test_data.drop("label", axis=1, inplace=True)

        return train_data, val_data, test_data

    def __init__(self, tokenizer, df, max_length=512):
        self.tokenizer = tokenizer
        self.data = []

        for i, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row["context"],
                row["hypothesis"],
                max_length=max_length,
                truncation="longest_first",
            )

            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": row["label"] if "label" in row else None,
                }
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_n_labels(self):
        return 3

    def __getitem__(self, idx):
        output = [
            torch.tensor(self.data[idx]["input_ids"]),
            torch.tensor(self.data[idx]["attention_mask"]),
        ]

        if self.data[idx]["label"] is not None:
            label = torch.tensor(self.data[idx]["label"]).long()
            output.append(label)

            return (output, [label])

        return (output,)


class CNLIDataset(Dataset):
    @classmethod
    def load(cls):
        train_data = pd.read_csv(
            f"{ROOT}CNLI/train.tsv", sep="\t", quoting=csv.QUOTE_NONE
        )
        val_data = pd.read_csv(f"{ROOT}CNLI/test.tsv", sep="\t", quoting=csv.QUOTE_NONE)
        test_data = None

        for df in [train_data, val_data]:
            df["label"] = df["gold_label"].apply(
                lambda x: {"contradiction": 0, "entailment": 1, "neutral": 2}[x]
            )

        return train_data, val_data, test_data

    def __init__(self, tokenizer, df, max_length=512):
        self.tokenizer = tokenizer
        self.data = []

        for i, row in df.iterrows():
            encoded = tokenizer.encode_plus(
                row["sentence1"],
                row["sentence2"],
                max_length=max_length,
                truncation="longest_first",
            )

            self.data.append(
                {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "label": row["label"] if "label" in row else None,
                }
            )

    def __len__(self):
        return len(self.data)

    @classmethod
    def get_n_labels(self):
        return 3

    def __getitem__(self, idx):
        output = [
            torch.tensor(self.data[idx]["input_ids"]),
            torch.tensor(self.data[idx]["attention_mask"]),
        ]

        if self.data[idx]["label"] is not None:
            label = torch.tensor(self.data[idx]["label"]).long()
            output.append(label)

            return (output, [label])

        return (output,)


DATASETS = {
    "CB": CBDataset,
    "RTE": RTEDataset,
    "CNLI": CNLIDataset,
    "ANLI": FullANLIDataset,
    "QNLI": QNLIDataset,
}
