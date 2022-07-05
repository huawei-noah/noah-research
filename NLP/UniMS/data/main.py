# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from os.path import join
import json

from preprocess import (
    make_article_image_caption_link,
    check_file,
    make_extract_pseudo_label,
    make_image_pseudo_label,
)


def make_file_data(source_path, split):
    file_data = make_article_image_caption_link(source_path, split)
    checked_file_data = check_file(source_path, file_data)
    return checked_file_data


def make_pseudo_label(source_path, checked_file_data):
    extract_data = make_extract_pseudo_label(source_path, checked_file_data)
    data = make_image_pseudo_label(source_path, extract_data)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="data path")
    parser.add_argument("--split", type=str, required=True, help="data path")
    args = parser.parse_args()
    source_path = args.path
    split = args.split
    assert split in ["train", "valid", "test"]

    output_path = join(source_path, "preprocess")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print(f"Collect Data for {split} split.")
    checked_file_data = make_file_data(source_path, split)
    data = make_pseudo_label(source_path, checked_file_data)

    save_path = join(output_path, f"{split}_data.json")
    print(f"Save Data to {save_path}.")
    with open(save_path, 'w') as file:
        json.dump(data, file)
    print()
