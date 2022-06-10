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

# Take PERT as the emission probability and ngram (bigram) as transition 
# probability, and unify them into Markov framework.


import os
import argparse
import time
import numpy as np
import math
import json
import logging
from tqdm import tqdm
from tqdm.contrib import tzip

import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn

from PinyinCharDataProcesser import PinyinCharDataProcesser
from NEZHA.modeling_nezha import BertConfig, BertForTokenClassification

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

class Py2CharIndex(object):
    def __init__(self, py2CharPath):
        self.py2CharPath = py2CharPath
        self.py2char = dict()
    # the line is: ai, 埃, 挨, 哎, 唉, 哀, 皑, 蔼, 矮, 碍, 爱, 隘, 癌, 艾,
    def LoadPy2CharIndexFrmCSV(self):
        with open(self.py2CharPath, 'r', encoding='utf-8') as fIn:
            reader = csv.reader(fIn)
            char_num = 0
            for item in reader:
                length = len(item)
                if length != 0:
                    pinyin = item[0]
                    charList = [item[i] for i in range(1, length) if len(item[i])]
                    self.py2char[pinyin] = charList
                    char_num += len(charList)
    # the line is: 麽 me ma yao mo
    # this is acutually used one
    def LoadPy2CharIndexFrmTxt(self):
        with open(self.py2CharPath, 'r', encoding='utf-8') as fIn:
            #char_num = 0
            for line in fIn.readlines():
                items = line.strip().split()
                if len(items) < 2:
                    continue
                else:
                    char = items[0]
                    pinyinList = items[1:]
                    for pinyin in pinyinList:
                        if pinyin in self.py2char.keys():
                            self.py2char[pinyin].append(char)
                        else:
                            charList = list(char)
                            self.py2char[pinyin] = charList
                        #char_num += 1
        pass
    def GetCharListFrmPinyin(self, pinyin):
        if pinyin in self.py2char.keys():
            return self.py2char[pinyin]
        else:
            return None

class BigramModel(object):
    def __init__(self, bigramModelPath):
        self.unigram, self.bigram, self.totalCount = self.LoadBigramFrmJson(bigramModelPath)
    # load bigram from json
    def LoadBigramFrmJson(self, bigramModelPath):
        with open(bigramModelPath, 'r') as fBigram:
            dict_list = json.load(fBigram)
            Unigram = dict_list[0]
            Bigram = dict_list[1]
        # calculate the total word frequence
        totalCount = 0
        for word in Unigram.keys():
            totalCount += Unigram[word]
        return Unigram, Bigram, totalCount
    # calculate the unigram probability
    def GetUnigramProb(self, word):
        if word not in self.unigram.keys():
            return 1e-10 / self.totalCount
        else:
            unigramProb = (self.unigram[word]+1e-10) / self.totalCount
            return unigramProb
            # unigramProbLog = math.log(unigramProb)
            # return unigramProbLog
    # calculate the bigram probability
    def GetBigramProb(self, word1, word2):
        if word1 not in self.bigram.keys():
            return 0.0
        elif word2 not in self.bigram[word1].keys():
            return 0.0
        else:
            unigramCount = self.unigram[word1] 
            bigramCount = self.bigram[word1][word2]
            bigramProb = float(bigramCount) / unigramCount
            return bigramProb
            # bigramProbLog = math.log(bigramProb)
            # return bigramProbLog

class ViterbiNode(object):
     def __init__(self):
        self.charStr = str("")
        self.maxPathId = -1
        self.maxPathProbLog = -1e10
        self.emitProbLog = -1e10

class InputPinyinCharNode(object):
     def __init__(self):
        self.pinyin = str("")
        self.ViterbiNodeList = list()

class Py2WordPERT(object):
    def __init__(self, 
        charLexPath, 
        pyLexPath, 
        pinyin2PhrasePath, 
        phrase2CharPath, 
        bigramModelPath, 
        modelPath
        ):
        # load the char lexicon
        _, self.char2id, _ = self.LoadLexicon(charLexPath) 
        # load the pinyin lexicon
        self.pyLex, self.pinyin2id, _ = self.LoadLexicon(pyLexPath)
        self.pyLexSet = set(self.pyLex)
        # load the pinyin to char or word mapping, a dict of [pinyin] = word1, word2 ...
        self.Py2Phrase = self.LoadMapOfPhrase2PYOrChar(pinyin2PhrasePath)
        # load the bigram
        self.BigramModel = BigramModel(bigramModelPath)
        # now it's the char-based bert model
        self.bert_config = BertConfig.from_json_file(os.path.join(modelPath,'model_config.json'))
        self.model = BertForTokenClassification(config=self.bert_config, num_labels=self.bert_config.num_labels)
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(os.path.join(modelPath,'model.pt'), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
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

    def LoadMapOfPhrase2PYOrChar(self, theMapFileIn):
        MapDict = dict()
        with open(theMapFileIn, 'r', encoding='utf-8') as fMapFile:
            for line in fMapFile.readlines():
                items = line.strip().split()
                MapDict[items[0]] = items[1:]
        return MapDict

    def ConvertPyList2CharProbMat(self, pinyinList):
        pyIdList = [self.pinyin2id[pinyin] for pinyin in pinyinList if pinyin in self.pyLexSet]
        assert(self.bert_config.seq_length > 0)
        def partition(theList, size):
            return [theList[i:i+size] for i in range(0, len(theList), size)]
        pyIdList_partition = partition(pyIdList, self.bert_config.seq_length)
        num_parts = len(pyIdList_partition)
        if num_parts > 1:
            pyIdList_partition_1stPart = pyIdList_partition[0:num_parts-1]
            input_ids = Variable(torch.tensor(pyIdList_partition_1stPart,dtype=torch.int64).contiguous()).to(self.device)
            output_1stPart = self.model(input_ids = input_ids)
            output_1stPart = torch.reshape(output_1stPart, (-1, output_1stPart.shape[-1])) # from [bs, len, vocab_size] to [bs * len, vocab_size]
        pyIdList_partition_2ndPart = pyIdList_partition[-1]
        len_2ndPart = len(pyIdList_partition_2ndPart)
        pyIdList_partition_2ndPart = pyIdList_partition_2ndPart + [0] * (self.bert_config.seq_length - len_2ndPart)
        input_ids = Variable(torch.tensor(pyIdList_partition_2ndPart,dtype=torch.int64).contiguous()).to(self.device)
        input_ids = input_ids.unsqueeze(dim=0)
        attention_mask = [1] * len_2ndPart + [0] * (self.bert_config.seq_length - len_2ndPart)
        attention_mask = Variable(torch.tensor(attention_mask,dtype=torch.int64).contiguous()).to(self.device)
        attention_mask = attention_mask.unsqueeze(dim=0)
        output_2ndPart = self.model(input_ids = input_ids, attention_mask = attention_mask)
        output_2ndPart = output_2ndPart[:, :len_2ndPart, :]
        output_2ndPart = torch.reshape(output_2ndPart, (-1, output_2ndPart.shape[-1])) # from [1, len_2ndPart, vocab_size] to [len_2ndPart, vocab_size]
        if num_parts > 1:
            output = torch.cat((output_1stPart, output_2ndPart), dim=0)
        else:
            output = output_2ndPart
        output_prob_log = torch.nn.functional.log_softmax(output, dim=1)
        return output_prob_log.cpu()

    def GetEmitProbOfWord(self, charEmitProbMat, word, iBegin, iEnd):
        wordEmitProbLog = -1e10
        charList = [i for i in word]
        # assert (len(charList)==iEnd-iBegin)
        if len(charList)!=iEnd-iBegin:
            # something error, such as: the input is 'tu an', but we inquiry the candidate word of 'tuan'
            return wordEmitProbLog
        sum_prob_log = 0.0
        for k, theChar in enumerate(charList):
            iCharId = self.char2id[theChar]
            sum_prob_log += charEmitProbMat[iBegin+k][iCharId].item() 
        wordEmitProbLog = sum_prob_log / len(charList)
        return wordEmitProbLog

    def ConstructViterbiLattice(self, pinyinList):
        charEmitProbMat = self.ConvertPyList2CharProbMat(pinyinList)
        ViterbiLattice = list()
        length = len(pinyinList)
        for i in range(length):
            theInputPinyinCharNode = InputPinyinCharNode()
            theInputPinyinCharNode.pinyin = pinyinList[i]
            for j in range(i+1):
                theCombinedPys = ''.join(pinyinList[j:i+1])
                if theCombinedPys in self.Py2Phrase.keys():
                    wordList = self.Py2Phrase[theCombinedPys]
                else:
                    continue
                for word in wordList:
                    theViterbiNode = ViterbiNode()
                    theViterbiNode.wordStr = word
                    theViterbiNode.wordLength = (i + 1) - j
                    theViterbiNode.emitProbLog = self.GetEmitProbOfWord(charEmitProbMat, word, j, i+1) 
                    theInputPinyinCharNode.ViterbiNodeList.append(theViterbiNode)
            ViterbiLattice.append(theInputPinyinCharNode)
        return ViterbiLattice

    def ViterbiSearch(self, ViterbiLattice):
        length = len(ViterbiLattice)
        if length <= 0:
            return
        # for the first pinyin, set the unigram
        the1stPinyinCharNode = ViterbiLattice[0]
        viterbiNodeLength = len(the1stPinyinCharNode.ViterbiNodeList)
        for i in range(viterbiNodeLength):
            unigramProb = self.BigramModel.GetUnigramProb(the1stPinyinCharNode.ViterbiNodeList[i].wordStr)
            unigramProbLog = math.log10(unigramProb)
            emitProbLog = the1stPinyinCharNode.ViterbiNodeList[i].emitProbLog
            the1stPinyinCharNode.ViterbiNodeList[i].maxPathProbLog = unigramProbLog + emitProbLog
        if length == 1:
            return
        else:
            # the rest pinyins
            for i in range(1, length):
                viterbiNodeLength = len(ViterbiLattice[i].ViterbiNodeList)
                for j in range(viterbiNodeLength):
                    word2 = ViterbiLattice[i].ViterbiNodeList[j].wordStr
                    unigram = self.BigramModel.GetUnigramProb(word2)
                    word2_len = ViterbiLattice[i].ViterbiNodeList[j].wordLength
                    # iterate the previous pinyin and get the 1st word, to obtain the max path prob
                    iPreViterbiNodeId = i - word2_len
                    if iPreViterbiNodeId == -1: # the first node
                        theProbLog = math.log10(unigram) + ViterbiLattice[i].ViterbiNodeList[j].emitProbLog
                        ViterbiLattice[i].ViterbiNodeList[j].maxPathProbLog = theProbLog
                    else:
                        for k, thePreViterbiNode in enumerate(ViterbiLattice[iPreViterbiNodeId].ViterbiNodeList):
                            word1 = thePreViterbiNode.wordStr
                            bigram = self.BigramModel.GetBigramProb(word1, word2)
                            theProb = 0.1 * unigram + 0.9 * bigram
                            theProbLog = math.log10(theProb) + thePreViterbiNode.maxPathProbLog + ViterbiLattice[i].ViterbiNodeList[j].emitProbLog
                            if theProbLog >= ViterbiLattice[i].ViterbiNodeList[j].maxPathProbLog:
                                ViterbiLattice[i].ViterbiNodeList[j].maxPathProbLog = theProbLog
                                ViterbiLattice[i].ViterbiNodeList[j].maxPathId = k
        return

    def BackTrace(self, ViterbiLattice):
        wordListReturn = list()
        # get the max node of the last time step
        viterbiNodeLength = len(ViterbiLattice[-1].ViterbiNodeList)
        theMaxPathProbLog = -1e10
        theMaxPathId = -1
        theLastMaxId = -1
        for j in range(viterbiNodeLength):
            if ViterbiLattice[-1].ViterbiNodeList[j].maxPathProbLog > theMaxPathProbLog:
                theMaxPathProbLog = ViterbiLattice[-1].ViterbiNodeList[j].maxPathProbLog
                theMaxPathId = ViterbiLattice[-1].ViterbiNodeList[j].maxPathId
                theLastMaxId = j
        word = ViterbiLattice[-1].ViterbiNodeList[theLastMaxId].wordStr
        wordListReturn.append(word)
        word_length = ViterbiLattice[-1].ViterbiNodeList[theLastMaxId].wordLength
        i = -1
        while (theMaxPathId != -1):
            theViterbiNode = ViterbiLattice[i-word_length].ViterbiNodeList[theMaxPathId]
            word = theViterbiNode.wordStr
            wordListReturn.append(word)
            theMaxPathId = theViterbiNode.maxPathId
            i -= word_length
            word_length = theViterbiNode.wordLength
        # revert the word list
        wordListReturn.reverse()
        return wordListReturn

    def ConvertPinyinListToCharList(self, pinyinList):
        if len(pinyinList) <= 0:
            return 
        # construct the viterbi lattice
        ViterbiLattice = self.ConstructViterbiLattice(pinyinList)
        # the viterbi search
        self.ViterbiSearch(ViterbiLattice)
        # back trace
        wordlist = self.BackTrace(ViterbiLattice)
        return wordlist

    def ConvertPinyinListToCharListOnFile(self, pinyinFile, rsltCharFile):
        with open(pinyinFile, 'r', encoding='utf-8') as fIn:
            with open(rsltCharFile, 'w', encoding='utf-8') as fOut:
                for line in tqdm(fIn.readlines()):
                    pinyinList = line.strip().split()
                    wordlist = self.ConvertPinyinListToCharList(pinyinList)
                    tmpLine = ''.join(wordlist)
                    # resegment it by single char
                    charlist = [i for i in tmpLine]
                    outline = ' '.join(charlist) + '\n'
                    fOut.write(outline)
        pass

def EvaluteOnLine(goldenCharLine, rsltCharLine):
    totalCharCount = 0
    correctCharCount = 0
    goldenCharList = goldenCharLine.split()
    rsltCharList = rsltCharLine.split()
    for goldenChar, rsltChar in zip(goldenCharList, rsltCharList):
        if goldenChar == rsltChar:
            correctCharCount += 1
        totalCharCount += 1
    return correctCharCount, totalCharCount

def EvaluateOnFiles(goldenCharFileIn, rsltCharFileIn, logOut):
    totalCount = 0
    correctCount = 0
    with open(logOut, 'w', encoding='utf-8') as fOut:
        with open(goldenCharFileIn, 'r', encoding='utf-8') as fGoldenCharFileIn:
            with open(rsltCharFileIn, 'r', encoding='utf-8') as frsltCharFileIn:
                for goldenLine, rsltLine in tzip(fGoldenCharFileIn.readlines(), frsltCharFileIn.readlines()):
                    fOut.write("G: " + goldenLine)
                    fOut.write("R: " + rsltLine)
                    correctCharCount, totalCharCount = EvaluteOnLine(goldenLine, rsltLine)
                    fOut.write("Correct: " + str(correctCharCount) + "  Total: " + str(totalCharCount) + \
                        " precision: " + str(correctCharCount/totalCharCount) + " error rate: " + \
                        str((totalCharCount-correctCharCount)/totalCharCount) + "\n")
                    correctCount += correctCharCount
                    totalCount += totalCharCount
                fOut.write("Correct in the whole file: " + str(correctCount) + \
                    "  Total in the whole file: " + str(totalCount) + \
                     " precision: " + str(correctCount/totalCount) + " error rate: " + \
                        str((totalCount-correctCount)/totalCount) + "\n")
                fOut.flush()
    pass

def Evaluation(thePy2Char, pinyinFile, rsltCharFileIn, goldenCharFileIn, logFileOut):
    t_begin = time.time()
    thePy2Char.ConvertPinyinListToCharListOnFile(pinyinFile, rsltCharFileIn)
    t_end = time.time()
    total_time = (t_end - t_begin)
    EvaluateOnFiles(goldenCharFileIn, rsltCharFileIn, logFileOut)
    with open(logFileOut, 'a+') as fLog:
        fLog.write("Total time consuming: " + str(total_time) + "\n")
    pass

def PrintModelParaNum(model):
    totalParaNum = 0
    for name, para in model.named_parameters():
        print(name + " : " + str(para.shape) + " : " + str(para.numel()))
        totalParaNum += para.numel()
    print('total parameter number :' + str(totalParaNum))

def main():
    parser = argparse.ArgumentParser(description='Bert model for pinyin2char in PyTorch')
    parser.add_argument('--charLex', type=str, default='',
                        help='location of the char lexicon file')
    parser.add_argument('--pyLex', type=str, default='',
                        help='location of the pinyin list')
    parser.add_argument('--pinyin2PhrasePath', type=str, default='',
                        help='location of the mapping file from pinyin to Chinese word')
    parser.add_argument('--phrase2CharPath', type=str, default='',
                        help='location of the mapping file from Chinese word to single Chinese character')
    parser.add_argument('--bigramModelPath', type=str, default='',
                        help='location of char bigram file')
    parser.add_argument('--modelPath', type=str, default='',
                        help='this is the path to bert model which contains the model file and the config file')
    parser.add_argument('--charFile', type=str, default='',
                        help='location of the char corpus')
    parser.add_argument('--pinyinFile', type=str, default='',
                        help='location of the pinyin corpus')
    parser.add_argument('--conversionRsltFile', type=str, default='',
                        help='this is the file to contain the conversion result')
    parser.add_argument('--logFile', type=str, default='',
                        help='this is the file to contain the evaluation result')
    args, _ = parser.parse_known_args()
    thePy2WordPERT = Py2WordPERT(
        charLexPath = args.charLex, 
        pyLexPath = args.pyLex, 
        pinyin2PhrasePath = args.pinyin2PhrasePath,
        phrase2CharPath = args.phrase2CharPath, 
        bigramModelPath = args.bigramModelPath, 
        modelPath = args.modelPath)
    pinyinList = ['zhong', 'guo', 'gong', 'cheng', 'yuan', 'yuan', 'shi', 'zhong', 'nan', 'shan', 'jiu', 'jin', 'qi', 'de', 'xiang', 'gang', 'yi', 'qing', 'fa', 'zhan', 'ji', 'fang', 'kong', 'cuo', 'shi', 'deng', 'wen', 'ti', 'hui', 'da', 'ji', 'zhe', 'ti', 'wen', 'shi']
    charList = thePy2WordPERT.ConvertPinyinListToCharList(pinyinList)
    print(charList)
    Evaluation(thePy2Char = thePy2WordPERT, 
        pinyinFile = args.pinyinFile, 
        rsltCharFileIn = args.conversionRsltFile,
        goldenCharFileIn = args.charFile,
        logFileOut = args.logFile)
    # PrintModelParaNum(thePy2CharBert.model)
    pass

if __name__ == "__main__":   
    print ('hello')
    main()
    print('olleh')
