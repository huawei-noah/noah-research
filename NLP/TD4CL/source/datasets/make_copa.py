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

import xml.etree.ElementTree as ET
import sys
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str)
parser.add_argument("--output", "-o", type=str)
args = parser.parse_args()

os.system(f"mkdir -p {'/'.join(args.output.split('/')[:-1])}")

tree = ET.parse(args.input)
root = tree.getroot()

with open(args.output, 'w') as outfile:
    for child in root:
        item = {'idx': child.attrib['id'], 'question': child.attrib['asks-for'],
                'label': int(child.attrib['most-plausible-alternative'])-1}

        assert len(child) == 3
        for i, gchild in enumerate(child):
            if i == 0:
                item['premise'] = gchild.text
            elif i == 1:
                item['choice1'] = gchild.text
            else:
                item['choice2'] = gchild.text

        outfile.write(json.dumps(item) + '\n')
