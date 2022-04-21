# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import pickle
from zipfile import ZipFile
from utils import load_tokenizer


def extract_zip(file_name: str, directory: str):
    with ZipFile(directory + os.path.sep + file_name, 'r') as zip_ref:
        zip_ref.extractall(directory)


if __name__ == "__main__":

    preprocess_mtod = True
    preprocess_mtop = True
    preprocess_atis = True
    model_path = '../xlm-roberta-large/'

    if preprocess_mtod:
        # ---------- MTOD DATA PREPARATION -----------
        task, data_dir = 'mtod', 'data'
        extract_zip(task + ".zip", data_dir)
        tokenizer = load_tokenizer(model_path)
        unique_id = -1
        for lang in ['en', 'es', 'th']:
            for split in ['train', 'dev', 'test']:
                print("Processing %s-%s split, language: %s." % (task, split, lang))
                if not os.path.exists(os.path.join(data_dir, task, lang, split)):
                    os.mkdir(os.path.join(data_dir, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels, all_ids = [], [], [], []
                lines = open(os.path.join(data_dir, task, lang, "%s-%s.conllu" % (split, lang)), "r", encoding="utf-8").readlines()
                i = 0
                while i < len(lines):
                    inputs, slot_labels = [], []
                    assert lines[i].startswith("# text: ")
                    assert lines[i + 1].startswith("# intent: ")
                    intent = lines[i + 1].replace("# intent: ", "").rstrip()
                    intent_label = intent
                    assert lines[i + 2].startswith("# slots: ")
                    i += 3
                    while len(lines[i].strip()) != 0:
                        line = lines[i].split("\t")
                        assert len(line) == 4
                        slot = line[3].rstrip()
                        slot = "O" if slot == 'NoLabel' else slot
                        slot = slot.replace("B-", "").replace("I-", "")
                        tokenized = [t for t in tokenizer.tokenize(line[1]) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                        i += 1
                    i += 1
                    unique_id += 1
                    all_inputs.append(inputs)
                    all_ids.append(unique_id)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels,
                        'intent_labels': all_intent_labels, 'all_ids': all_ids}
                pickle.dump(dump, open(os.path.join(data_dir, task, lang, split, "data.pkl"), "wb"))
        print("Finished %s preprocessing." % task)

    if preprocess_mtop:
        # ---------- MTOP DATA PREPARATION -----------
        task, data_dir = 'mtop', 'data'
        extract_zip(task + ".zip", data_dir)
        tokenizer = load_tokenizer(model_path)
        unique_id = -1
        for lang in ['en', 'de', 'es', 'th', 'hi', 'fr']:
            for split in ['train', 'dev', 'test']:
                print("Processing %s-%s split, language: %s." % (task, split, lang))
                if not os.path.exists(os.path.join(data_dir, task, lang, split)):
                    os.mkdir(os.path.join(data_dir, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels, all_ids = [], [], [], []
                lines = open(os.path.join(data_dir, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                i = 0
                while i < len(lines):
                    inputs, slot_labels = [], []
                    assert lines[i].startswith("IN:")
                    line = lines[i].split("\t")
                    assert len(line) == 7
                    intent = line[0].strip().replace("IN:", "")
                    intent_label = intent
                    slots = [] if len(line[1]) == 0 else [s.split(":") for s in line[1].split(",")]
                    all_tokens = eval(line[6])
                    tokens, tokenSpans = all_tokens['tokens'], all_tokens['tokenSpans']
                    assert len(tokens) == len(tokenSpans)
                    for token, tokenSpan in zip(tokens, tokenSpans):
                        slot = "O"
                        for s in slots:
                            if tokenSpan['start'] == int(s[0]):
                                slot = s[3]
                                break
                            elif tokenSpan['start'] > int(s[0]) and tokenSpan['start'] + tokenSpan['length'] <= int(s[1]):
                                slot = s[3]
                                break
                        tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                    i += 1
                    unique_id += 1
                    all_inputs.append(inputs)
                    all_ids.append(unique_id)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels,
                        'intent_labels': all_intent_labels, 'all_ids': all_ids}
                pickle.dump(dump, open(os.path.join(data_dir, task, lang, split, "data.pkl"), "wb"))
        print("Finished %s preprocessing." % task)

    if preprocess_atis:
        # ---------- MULTI ATIS++ DATA PREPARATION -----------
        task, data_dir = 'm_atis', 'data'
        extract_zip(task + ".zip", data_dir)
        tokenizer = load_tokenizer(model_path)
        unique_id = -1
        for lang in ['en', 'de', 'es', 'tr', 'fr', 'zh', 'hi', 'ja', 'pt']:
            for split in ['train', 'dev', 'test']:
                print("Processing %s-%s split, language: %s." % (task, split, lang))
                if not os.path.exists(os.path.join(data_dir, task, lang, split)):
                    os.mkdir(os.path.join(data_dir, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels, all_ids = [], [], [], []
                lines = open(os.path.join(data_dir, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                i = 0
                if lines[0].startswith('id'):
                    lines = lines[1:]
                while i < len(lines):
                    inputs, slot_labels = [], []
                    line = lines[i].split("\t")
                    assert len(line) == 4
                    intent = line[3].strip()
                    intent_label = intent
                    slots = line[2].split(" ")
                    tokens = line[1].split(" ")
                    assert len(tokens) == len(slots)
                    for token, slot in zip(tokens, slots):
                        slot = slot.replace("B-", "").replace("I-", "")
                        tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                    i += 1
                    unique_id += 1
                    all_inputs.append(inputs)
                    all_ids.append(unique_id)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels,
                        'intent_labels': all_intent_labels, 'all_ids': all_ids}
                pickle.dump(dump, open(os.path.join(data_dir, task, lang, split, "data.pkl"), "wb"))
        print("Finished %s preprocessing." % task)
