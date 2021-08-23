# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import pickle
from utils import load_tokenizer

if __name__ == "__main__":

    do_preprocess_mtod = False
    do_preprocess_mtop = False
    do_preprocess_atis = False
    do_preprocess_paws = False

    if do_preprocess_mtod:
        # ---------- MTOD DATA PREPARATION -----------
        unique_intents, unique_slots = set(), set()
        task, data = 'mtod', 'data'
        tokenizer = load_tokenizer('../xlm-roberta-base/')
        for lang in ['en', 'es', 'th']:
            for split in ['train', 'dev', 'test']:
                print("Processing the %s split of language: %s." % (split, lang))
                if not os.path.exists(os.path.join("..", data, task, lang, split)):
                    os.mkdir(os.path.join("..", data, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels = [], [], []
                lines = open(os.path.join("..", data, task, lang, "%s-%s.conllu" % (split, lang)), "r", encoding="utf-8").readlines()
                i = 0
                while i < len(lines):
                    inputs, slot_labels = [], []
                    assert lines[i].startswith("# text: ")
                    assert lines[i + 1].startswith("# intent: ")
                    intent = lines[i + 1].replace("# intent: ", "").rstrip()
                    intent_label = intent
                    unique_intents.add(intent)
                    assert lines[i + 2].startswith("# slots: ")
                    i += 3
                    while len(lines[i].strip()) != 0:
                        line = lines[i].split("\t")
                        assert len(line) == 4
                        slot = line[3].rstrip()
                        slot = "O" if slot == 'NoLabel' else slot
                        slot = slot.replace("B-", "").replace("I-", "")
                        unique_slots.add(slot)
                        tokenized = [t for t in tokenizer.tokenize(line[1]) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                        i += 1
                    i += 1
                    all_inputs.append(inputs)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels, 'intent_labels': all_intent_labels}
                pickle.dump(dump, open(os.path.join("..", data, task, lang, split, "data.pkl"), "wb"))
        pickle.dump(list(unique_slots), open(os.path.join('..', data, task, 'slots.pkl'), 'wb'))
        pickle.dump(list(unique_intents), open(os.path.join('..', data, task, 'intents.pkl'), 'wb'))
        print("Finished.")

    if do_preprocess_mtop:
        # ---------- MTOP DATA PREPARATION -----------
        unique_intents, unique_slots = set(), set()
        task, data = 'mtop', 'data'
        tokenizer = load_tokenizer('../xlm-roberta-base/')
        for lang in ['en', 'de', 'es', 'th', 'hi', 'fr']:
            for split in ['train', 'dev', 'test']:
                print("Processing the %s split of language: %s." % (split, lang))
                if not os.path.exists(os.path.join("..", data, task, lang, split)):
                    os.mkdir(os.path.join("..", data, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels = [], [], []
                lines = open(os.path.join("..", data, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                i = 0
                while i < len(lines):
                    inputs, slot_labels = [], []
                    assert lines[i].startswith("IN:")
                    line = lines[i].split("\t")
                    assert len(line) == 7
                    intent = line[0].strip().replace("IN:", "")
                    intent_label = intent
                    unique_intents.add(intent)
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
                        unique_slots.add(slot)
                        tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                    i += 1
                    all_inputs.append(inputs)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels, 'intent_labels': all_intent_labels}
                pickle.dump(dump, open(os.path.join("..", data, task, lang, split, "data.pkl"), "wb"))
        pickle.dump(list(unique_slots), open(os.path.join('..', data, task, 'slots.pkl'), 'wb'))
        pickle.dump(list(unique_intents), open(os.path.join('..', data, task, 'intents.pkl'), 'wb'))
        print("Finished.")

    if do_preprocess_atis:
        # ---------- MULTI ATIS++ DATA PREPARATION -----------
        unique_intents, unique_slots = set(), set()
        task, data = 'm_atis', 'data'
        tokenizer = load_tokenizer('../xlm-roberta-base/')
        for lang in ['en', 'de', 'es', 'tr', 'fr', 'zh', 'hi', 'ja', 'pt']:
            for split in ['train', 'dev', 'test']:
                print("Processing the %s split of language: %s." % (split, lang))
                if not os.path.exists(os.path.join("..", data, task, lang, split)):
                    os.mkdir(os.path.join("..", data, task, lang, split))
                all_inputs, all_slot_labels, all_intent_labels = [], [], []
                lines = open(os.path.join("..", data, task, lang, "%s.txt" % split), "r", encoding="utf-8").readlines()
                i = 0
                if lines[0].startswith('id'):
                    lines = lines[1:]
                while i < len(lines):
                    inputs, slot_labels = [], []
                    line = lines[i].split("\t")
                    assert len(line) == 4
                    intent = line[3].strip()
                    intent_label = intent
                    unique_intents.add(intent)
                    slots = line[2].split(" ")
                    tokens = line[1].split(" ")
                    assert len(tokens) == len(slots)
                    for token, slot in zip(tokens, slots):
                        slot = slot.replace("B-", "").replace("I-", "")
                        unique_slots.add(slot)
                        tokenized = [t for t in tokenizer.tokenize(token) if t != '▁']
                        for idx, t in enumerate(tokenized):
                            if idx == 0 or slot == "O":
                                slot_labels.append(slot)
                            else:
                                slot_labels.append("PAD")
                            inputs.append(t)
                    i += 1
                    all_inputs.append(inputs)
                    all_slot_labels.append(slot_labels)
                    all_intent_labels.append(intent_label)
                dump = {'inputs': all_inputs, 'slot_labels': all_slot_labels, 'intent_labels': all_intent_labels}
                pickle.dump(dump, open(os.path.join("..", data, task, lang, split, "data.pkl"), "wb"))
        pickle.dump(list(unique_slots), open(os.path.join('..', data, task, 'slots.pkl'), 'wb'))
        pickle.dump(list(unique_intents), open(os.path.join('..', data, task, 'intents.pkl'), 'wb'))
        print("Finished.")

    if do_preprocess_paws:
        # ---------- MULTI ATIS++ DATA PREPARATION -----------
        unique_labels = set()
        task, data = 'paws_x', 'data'
        tokenizer = load_tokenizer('../xlm-roberta-base/')
        for lang in ['en', 'de', 'es', 'fr', 'zh', 'ja', 'ko']:
            for split in ['train', 'dev', 'test']:
                print("Processing the %s split of language: %s." % (split, lang))
                if not os.path.exists(os.path.join("..", data, task, lang, split)):
                    os.mkdir(os.path.join("..", data, task, lang, split))
                all_inputs, all_labels = [], []
                lines = open(os.path.join("..", data, task, lang, "%s.tsv" % split), "r", encoding="utf-8").readlines()
                i = 0
                if lines[0].startswith('id'):
                    lines = lines[1:]
                while i < len(lines):
                    inputs = []
                    line = lines[i].split("\t")
                    assert len(line) == 4
                    label = line[3].strip()
                    unique_labels.add(label)
                    if len(line[1].strip()) == 0 or len(line[2].strip()) == 0:
                        i += 1
                        continue
                    inp_one = tokenizer.tokenize(line[1])
                    inp_two = tokenizer.tokenize(line[2])
                    inputs.append((inp_one, inp_two))
                    i += 1
                    all_inputs.append(inputs)
                    all_labels.append(label)
                dump = {'inputs': all_inputs, 'intent_labels': all_labels}
                pickle.dump(dump, open(os.path.join("..", data, task, lang, split, "data.pkl"), "wb"))
