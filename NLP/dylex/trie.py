# coding=utf-8
# Copyright (c) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

from pygtrie import CharTrie, StringTrie

from collections import namedtuple
from pathlib import Path
import torch
import logging
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)
DicInstance = namedtuple("DicInstance", ["name", "match_str", "start", "end", "length"])


class CustomTrieTree:
    """ 定制TrieTree模块，输入文本，返回所有匹配的结果 """

    def __init__(self):
        self.trietree = CharTrie()

    def load_dict(self, dict_path, name=True):
        lines = file2list(dict_path)
        for item in lines:
            if item:
                self._add_trie(item, name)
        logger.info(f"load {dict_path} successful!")

    def _add_trie(self, item, name):
        self.trietree[item] = name

    def has_key(self, key):
        return self.trietree.has_key(key)

    def query(self, text):
        """ 查询文本中词典出现的情况 """
        qrlt = []
        i, j = 0, 0
        while i < len(text):
            for j in range(i, len(text)):
                substr = text[i:j + 1]
                if self.trietree.has_key(substr):
                    name = self.trietree[substr]
                    instance = DicInstance(
                        name=name,
                        match_str=substr,
                        start=i,
                        end=j + 1,
                        length=j - i + 1)
                    qrlt.append(instance)
            i += 1
        return qrlt

    @classmethod
    def allvec(cls, text, qrlt, sort=False):
        """ 
        将trie树的查询结果转换为匹配向量 
        sort=true，将按照起始位置和长度进行排序
        """
        if sort:
            sorted(qrlt, key=lambda x: (x.start, x.length))
        vectors = []
        for instance in qrlt:
            vectors.append(cls.onevec(text, instance))
        return vectors

    @classmethod
    def onevec(cls, text, instance):
        vector = [0] * len(text)
        start = instance.start
        end = instance.end
        for i in range(start, end + 1):
            vector[i] = 1
        return vector


class SpaceTrieTree:
    """ 定制TrieTree模块，输入文本，返回所有匹配的结果 """

    def __init__(self, args):
        self.trietree = StringTrie(separator=" ")
        self.match_num = args.match_num

    def load_dict(self, dict_path, name=True):
        lines = file2list(dict_path)
        for item in lines:
            if item:
                self._add_trie(item, name)
        logger.info(f"load {dict_path} successful!")

    def load_dict_from_tsv(self, dict_path, delimiter="\t"):
        df_dic = open(dict_path, "r", encoding="utf-8")
        for line in tqdm(df_dic.readlines()):
            row = line.strip().split("\t")
            self._add_trie(row[1], row[0])
        logger.info(f"load {dict_path} successful!")

    def _add_trie(self, item, name):
        if not self.has_key(item):
            self.trietree[item] = set()
        self.trietree[item].add(name)

    def has_key(self, key):
        return self.trietree.has_key(key)

    def query(self, text):
        """ 查询文本中词典出现的情况 """
        qrlt = []
        i, j = 0, 0
        while i < len(text):
            # biggest_j = -1
            max_js = []
            for j in range(i, len(text)):
                substr = "".join(text[i:j + 1])
                if self.trietree.has_key(substr):
                    max_js = [j] + max_js
                    # print(max_js)
                    # biggest_j = j
            if max_js == []:
                i += 1
                continue
            max_js = max_js[0:min(len(max_js),self.match_num)]
            # print(self.match_num)
            # print(max_js)
            # if biggest_j != -1:

            for max_j in max_js:
                longest = "".join(text[i:max_j + 1])
                name = self.trietree[longest]
                for x in name:
                    # print(x)
                    instance = DicInstance(
                        name=x,
                        match_str=longest,
                        start=i,
                        end=max_j,
                        length=max_j - i + 1)
                    qrlt.append(instance)
            i += 1
        return qrlt, text

    @classmethod
    def allvec(cls, text, qrlt, sort=False):
        """
        将trie树的查询结果转换为匹配向量
        sort=true，将按照起始位置和长度进行排序
        """
        if sort:
            sorted(qrlt, key=lambda x: (x.start, x.length))
        vectors = []
        for instance in qrlt:
            vectors.append(cls.onevec(text, instance))
        return vectors

    @classmethod
    def onevec(cls, text, instance, max_length=None):
        vector = None
        if max_length:
            text = text[:max_length]
            vector = [0] * len(text)
        else:
            vector = [0] * len(text)
        start = instance.start
        end = instance.end
        for i in range(start, end + 1):
            vector[i] = 1
        return vector


def file2list(file_path, blank=True):
    """ 把文件读入列表中 """
    l_line = []
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if blank:
                l_line.append(s)
            else:
                if s: l_line.append(s)
    logger.info(f"sucess reading {file_path} into list !")
    return l_line


def list2file(lines, save_path, blank=True):
    """ 把列表存入文件 """
    with open(save_path, "w", encoding="utf-8") as f_w:
        for line in lines:
            s = line.strip()
            if blank:
                f_w.write(s)
            else:
                if s: f_w.write(s)
            f_w.write('\n')
    logger.info(f"sucess save list to {save_path} !")


def load_state_keywise_torch(model, model_state_path):
    """ 载入pytorch模型,支持dataparallel生成的模型 """
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_state_path, map_location='cpu')

    key = list(pretrained_dict.keys())[0]
    # filter out unnessary keys, multigpu -> cpu
    if (str(key).startswith('module.')):
        new_key = lambda x: x.replace('module.', "")
        pretrained_dict = {
            new_key(k): v if (new_key(k) in model_dict and v.size() == model_dict[new_key(k)].size())
            else logger.warn(f"key({k}) not in model or its size may not match!")
            for k, v in pretrained_dict.items()
        }
    else:
        pretrained_dict = {
            k: v if (k in model_dict and v.size() == model_dict[k].size())
            else logger.warn(f"key({k}) not in model or its size may not match!")
            for k, v in pretrained_dict.items()
        }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    logger.info(f'load state dict from {model_state_path}')
    return model
