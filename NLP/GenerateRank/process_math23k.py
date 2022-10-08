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

import os
import json
from exp_tree import from_postfix_to_infix

def map_symbol(text):
    for i in range(26):
        sym = chr(ord('a')+i)
        sym = f"temp_{sym}"
        text = text.replace(sym, f"#{i}")
    return text

def prepare_data(data_type="train"):
    data_path = "data"
    data_file = os.path.join(data_path, f"{data_type}23k_processed.json")
    data = json.load(open(data_file))
    with open(f"data/math23k/infix_math23k_processed.{data_type}", "w") as f:
        for d in data:
            idx = d['id']
            text, target = d["text"], d["target_norm_post_template"][2:]
            text = text.replace(" ", "")
            text = map_symbol(text)

            try:
                infix_target = from_postfix_to_infix(target)  # id 8883 and 17520 are deleted
            except:
                print(idx)
                continue
            infix_target = map_symbol(infix_target)
            f.write(text + '\t' + infix_target + '\n')

prepare_data("train")
prepare_data("valid")
prepare_data("test")
