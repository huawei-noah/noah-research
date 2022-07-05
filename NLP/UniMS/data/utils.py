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

from os.path import join
import hashlib
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode


def get_tokenizer(backbone):
    from transformers import GPT2Tokenizer, T5Tokenizer, BartTokenizer
    if backbone == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained(
            "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/transformers/gpt2-xl/")
    elif backbone == 'GPTNEO':
        tokenizer = GPT2Tokenizer.from_pretrained(
            "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/transformers/gpt-neo-2.7B/")
    elif backbone == 'GPTJ':
        tokenizer = GPT2Tokenizer.from_pretrained(
            "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/transformers/gpt-j-6B/")
    elif backbone == 'T5':
        tokenizer = T5Tokenizer.from_pretrained(
            "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/transformers/t5-3b/")
    elif backbone == 'BART':
        tokenizer = BartTokenizer.from_pretrained(
            "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/transformers/bart-base/")
    else:
        raise ValueError
    return tokenizer


def transform(n_px):
    return Compose([
        Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def parse_article(dataset_path, filename, lower=True):
    with open(join(dataset_path, filename), 'r', encoding='utf8') as file:
        content = file.readlines()
    iftitle, ifbody = False, False
    title = []
    body = []
    references = []

    for line in content:
        line = line.strip()
        if not has_meaning(line):
            continue
        elif line == '@title':
            iftitle = True
        elif line == '@body':
            ifbody = True
            iftitle = False
        elif line == '@summary':
            ifbody = False
        else:
            line = fix_missing_period(line)
            if iftitle:
                title.append(line.lower() if lower else line)
            elif ifbody:
                body.append(line.lower() if lower else line)
            else:
                references.append(line.lower() if lower else line)
    return title, body, references


def read_text_file(text_file):
    lines = []
    with open(text_file, "rb") as file:
        for line in file:
            lines.append(line.strip())
    return lines


def hashhex(s):
    h = hashlib.sha1()
    h.update(s)
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def get_msmo_filelist(source_path):
    # you can get all the text files in the `article` directory
    url_list = read_text_file(join(source_path, "url", "url_list"))
    hash_list = get_url_hashes(url_list)
    file_list = [join("article", f"{hash}.txt") for hash in hash_list]
    return file_list


def has_meaning(s):
    h = False
    for c in s:
        h = h or c.isalnum()
    return h
