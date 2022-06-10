#!/usr/bin/env python3
# -*- coding: utf-8

# Copyright 2021 Huawei Technologies Co., Ltd.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib import tzip

class PinyinCharDataProcesser(object):
    def __init__(self,
                vocabFile, 
                pinyinFile,
                seq_length = 16,
                pyDataFile = '', 
                charDataFile = ''):
        self.vocabFile = vocabFile
        self.vocab, self.word2id, self.id2word = self.LoadLexicon(self.vocabFile)
        self.pinyinFile = pinyinFile
        self.pinyins, self.py2id, self.id2py = self.LoadLexicon(self.pinyinFile)
        self.seq_length = seq_length
        self.pyDataFile = pyDataFile
        self.charDataFile = charDataFile
        self.num_samples = 0
        pass

    def LoadLexicon(self, fileIn):
        print(f'loading {fileIn}......')
        lexicon = []
        lex2id = {}
        id2lex = {}
        with open(fileIn, 'r', encoding='utf-8') as fFileIn:
            for line in fFileIn.readlines():
                item = line.strip()
                lexicon.append(item)
            for idx, item in enumerate(lexicon):
                lex2id[item] = idx
                id2lex[idx] = item
        return lexicon, lex2id, id2lex

    def SplitListBySeqLength(self, dataListIn):
        seqLists = list()
        seqNum = len(dataListIn) // self.seq_length
        for i in range(seqNum):
            seqList = dataListIn[i*self.seq_length:(i+1)*self.seq_length]
            seqLists.append(seqList)
        return seqLists

    def LoadData(self):
        print('loading data......')
        print(f'the pinyin file is: {self.pyDataFile}')
        print(f'the chardata file is: {self.charDataFile}')
        print(f'the max sequence length file is: {self.seq_length}')
        pinyindata = list()
        chardata = list()
        with open(self.pyDataFile, 'r', encoding='utf-8') as fPYDataIn:
            with open(self.charDataFile, 'r', encoding='utf-8') as fCharDataIn:
                print('getting into reading pinyin and char files......')
                pinyinset = set(self.pinyins)
                vocabset = set(self.vocab)
                for pyLine, charLine in tzip(fPYDataIn.readlines(), fCharDataIn.readlines()):
                    pyItems = pyLine.strip().split()
                    charItems = charLine.strip().split()
                    if len(pyItems) == len(charItems):
                        idsTup = [(self.py2id[pyItem], self.word2id[charItem]) for (pyItem, charItem) in zip(pyItems, charItems) if (pyItem in pinyinset) and (charItem in vocabset)]
                        pyIds, charIds = zip(*idsTup)
                        pinyindata.extend(pyIds)
                        chardata.extend(charIds)
        print('finishing reading the files, then splitting the data ......')
        pinyinLists = self.SplitListBySeqLength(pinyindata)
        chardataLists = self.SplitListBySeqLength(chardata)
        assert(len(pinyinLists) == len(chardataLists))
        self.num_samples = len(pinyinLists)
        print(f'the number of samples is: {self.num_samples}')
        dataset = TensorDataset(
            torch.as_tensor(np.array(pinyinLists, dtype=np.int64, copy=False)), 
            torch.as_tensor(np.array(chardataLists, dtype=np.int64, copy=False))
            )
        print('finishing dataset preparison......')
        return dataset


if __name__ == "__main__":
    print("hello")
    theDataProcesser = PinyinCharDataProcesser(
        vocabFile = "Corpus\\CharListFrmC4P.txt", 
        pinyinFile = "Corpus\\pinyinList.txt",
        seq_length = 16,
        pyDataFile = 'Corpus\\train_texts_pinyin_1k.txt', 
        charDataFile = 'Corpus\\train_texts_CharSeg_1k.txt'
    )
    theDataset = theDataProcesser.LoadData()
    train_dataloader = DataLoader(
        dataset=theDataset, 
        batch_size = 32,
        num_workers = 2,
        shuffle = False,
        drop_last = True)
    for step, batch in enumerate(train_dataloader):
        pinyindata = batch[0]
        chardata = batch[1]
        print(pinyindata.shape)
        print(pinyindata[0][0])
    print("finishing")

