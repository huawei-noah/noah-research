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

# coding=utf-8

import json
from os import listdir
from os.path import isfile, join
import networkx as nx
from sklearn.metrics import f1_score
from typing import Dict, List, Tuple
import numpy


class MultiWozConvGraph:

    def __init__(self, dir_name: str, file_names: List[str], seq_length: int = 5):
        self.seq_length = seq_length
        self.dialogue_id = 0
        self.dir_name = dir_name
        self.file_names = file_names
        self.graph = nx.DiGraph()
        self.augmented_conversations = dict()
        self.augmented_paths = set()
        g_vectors = self._init_graph_vectors()
        self.belief_state_to_idx = g_vectors[0]
        self.dialog_act_to_idx = g_vectors[1]
        self.final_state = str([1] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
        self._initiate_graph()

    def _initiate_graph(self) -> None:
        no_of_dialogues = 0
        unique_turns, repetitive_turns = 0.0, 0.0
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    dialogue['goal'] = {}
                    no_of_dialogues += 1
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    previous_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            start, end = str(previous_state), str(current_state)
                            if self.graph.has_edge(start, end):
                                self.graph[start][end]['turn'].append(turn)
                                self.graph[start][end]['probability'] += 1.0
                                repetitive_turns += 1
                            else:
                                self.graph.add_edge(start, end, turn=[turn], probability=1.0)
                                unique_turns += 1
                            previous_state = current_state
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            start, end = str(previous_state), str(current_state)
                            if self.graph.has_edge(start, end):
                                self.graph[start][end]['turn'].append(turn)
                                self.graph[start][end]['probability'] += 1.0
                                repetitive_turns += 1
                            else:
                                self.graph.add_edge(start, end, turn=[turn], probability=1.0)
                                unique_turns += 1
                            previous_state = current_state
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    previous_state = str(previous_state)
                    if self.graph.has_edge(previous_state, self.final_state):
                        self.graph[previous_state][self.final_state]['probability'] += 1
                    else:
                        self.graph.add_edge(previous_state, self.final_state, turn=[], probability=1.0)
        # nx.write_edgelist(self.graph, "graph.csv", delimiter=';', encoding="utf-8")
        degree = []
        # noinspection PyCallingNonCallable
        for node, value in self.graph.out_degree():
            if value < 21:  # Removing outliers that can really skew the average
                degree.append(value)
        # n, bins, patches = plt.hist(degree, 20, density=True, facecolor='b')
        # plt.xlabel('# of Outgoing Edges (capped at 20 for clearer comparison)')
        # plt.ylabel('% of Outgoing Edges')
        # plt.title('Histogram of Outgoing Edges - MultiWOZ')
        # plt.grid(True)
        # plt.show()
        print("-----------------------------------------------")
        print("Stats for ConvGraph for %s%s" % (self.dir_name, " and ".join(self.file_names)))
        print("Average degree: %2.3f (excluding outliers)" % numpy.mean(degree))
        print("Number of nodes: %d" % len(self.graph.nodes()))
        print("Number of edges: %d" % len(self.graph.edges()))
        print("Number of conversations: %d" % no_of_dialogues)
        print("Unique turns: %d" % unique_turns)
        print("Total turns: %d" % int(unique_turns + repetitive_turns))
        print("As a percentage: %2.3f" % (100 * unique_turns / (unique_turns + repetitive_turns)))
        print("-----------------------------------------------")
        # ------------------------- Calculate Probabilities --------------------------
        for start in self.graph:
            probabilities = []
            for end in self.graph[start]:
                probabilities.append(self.graph[start][end]['probability'])
            for end in self.graph[start]:
                self.graph[start][end]['probability'] = self.graph[start][end]['probability'] / sum(probabilities)

    def _init_graph_vectors(self) -> List[Dict[str, int]]:
        state_vector = set()
        dialog_act_vector = set()
        files = [f for f in listdir(self.dir_name) if isfile(join(self.dir_name, f))
                 and f.endswith(self.file_names[0].split(".")[-1])]
        for input_file in files:
            with open(self.dir_name + input_file, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        state_vector.add(domain + "_" + slot.lower())
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                state_vector.add(domain + "_" + slot + "_" + key.lower())
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            state_vector.add(domain + "_" + slot.lower())
                        if len(turn['dialog_act']) == 0:
                            dialog_act_vector.add("empty_dialogue_act")
                        else:
                            for act in turn['dialog_act']:
                                for slot in turn['dialog_act'][act]:
                                    dialog_act_vector.add(act.lower() + "_" + slot[0].lower())
                                dialog_act_vector.add(act.lower())
        state_vector = list(state_vector)
        state_vector.sort()
        dialog_act_vector = list(dialog_act_vector)
        dialog_act_vector.sort()
        # print("State Vector:", state_vector)
        # print("Dialog_act Vector:", dialog_act_vector)
        return [dict([(s, i) for i, s in enumerate(state_vector)]), dict([(s, i) for i, s in enumerate(dialog_act_vector)])]

    def generate_augmented_data(self, to_json: bool = False) -> numpy.array:
        self.augmented_paths.clear()
        train_x, train_y = [], []
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    visited_nodes = [[0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))]
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    visited_nodes.append(eval(self.final_state))
                    assert len(visited_nodes) % 2 == 0
                    if not to_json:
                        for i in range(2, len(visited_nodes) - 1, 2):
                            x = visited_nodes[max(0, i - self.seq_length): i]
                            while len(x) < self.seq_length:
                                x.insert(0, [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
                            dialog_acts = [(self.graph[str(visited_nodes[i - 1])][t], t) for t in self.graph[str(visited_nodes[i - 1])]]
                            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
                            for dialog_act in dialog_acts[:1]:
                                if dialog_act[1] == self.final_state:
                                    continue
                                y = eval(dialog_act[1])[len(self.belief_state_to_idx):]
                                if str(x) + str(y) not in self.augmented_paths:
                                    train_x.append(x)
                                    train_y.append(y)
                                    self.augmented_paths.add(str(x) + str(y))
                    else:
                        log, last_index = [], 0
                        for i in range(2, len(visited_nodes) - 1, 2):
                            log.extend(dialogue['log'][last_index: i - 1])
                            dialog_acts = [(self.graph[str(visited_nodes[i - 1])][t], t) for t in self.graph[str(visited_nodes[i - 1])]]
                            dialog_acts.sort(key=lambda t: t[0]['probability'], reverse=True)
                            log.append(dialog_acts[0][0]['turn'][0])
                            last_index = i
                        self.dialogue_id += 1
                        self.augmented_conversations[str(self.dialogue_id)] = {'goal': dict(), 'log': log}
        if to_json:
            with open(self.dir_name + self.file_names[0], "r") as inp:
                original_data = json.load(inp)
                self.augmented_conversations.update(original_data)
                with open(self.dir_name + "output/" + self.file_names[0], "w") as fp:
                    json.dump(self.augmented_conversations, fp, indent=2, sort_keys=True)
        return numpy.array(train_x, dtype='float32'), numpy.array(train_y, dtype='float32')

    def generate_standard_data(self, unique: bool) -> numpy.array:
        self.augmented_paths.clear()
        train_x, train_y = [], []
        for f_name in self.file_names:
            with open(self.dir_name + f_name, 'r') as f:
                data = json.load(f)
                for dialogue in data:
                    dialogue = data[dialogue]
                    visited_nodes = [[0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))]
                    last_belief_state = [0] * len(self.belief_state_to_idx)
                    for turn in dialogue['log']:
                        if len(turn['metadata']) > 0:
                            current_state = [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx))
                            for domain in turn['metadata']:
                                for slot in turn['metadata'][domain]['semi']:
                                    if turn['metadata'][domain]['semi'][slot] not in ["", "not mentioned", "none"]:
                                        index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                        current_state[index] = 1
                                        last_belief_state[index] = 1
                                for slot in turn['metadata'][domain]['book']:
                                    if slot == "booked":
                                        for item in turn['metadata'][domain]['book'][slot]:
                                            for key in item:
                                                index = self.belief_state_to_idx[domain + "_" + slot + "_" + key.lower()]
                                                current_state[index] = 1
                                                last_belief_state[index] = 1
                                    else:
                                        if turn['metadata'][domain]['book'][slot] not in ["", "not mentioned", "none"]:
                                            index = self.belief_state_to_idx[domain + "_" + slot.lower()]
                                            current_state[index] = 1
                                            last_belief_state[index] = 1
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                        else:
                            current_state = last_belief_state + ([0] * len(self.dialog_act_to_idx))
                            if len(turn['dialog_act']) == 0:
                                index = self.dialog_act_to_idx["empty_dialogue_act"]
                                current_state[index + len(self.belief_state_to_idx)] = 1
                            else:
                                for act in turn['dialog_act']:
                                    for slot in turn['dialog_act'][act]:
                                        index = self.dialog_act_to_idx[act.lower() + "_" + slot[0].lower()]
                                        current_state[index + len(self.belief_state_to_idx)] = 1
                                    index = self.dialog_act_to_idx[act.lower()]
                                    current_state[index + len(self.belief_state_to_idx)] = 1
                            visited_nodes.append(current_state)
                            last_belief_state = [0] * len(self.belief_state_to_idx)
                    visited_nodes.append(eval(self.final_state))
                    assert len(visited_nodes) % 2 == 0
                    for i in range(2, len(visited_nodes) - 1, 2):
                        x = visited_nodes[max(0, i - self.seq_length): i]
                        y = visited_nodes[i][len(self.belief_state_to_idx):]
                        while len(x) < self.seq_length:
                            x.insert(0, [0] * (len(self.belief_state_to_idx) + len(self.dialog_act_to_idx)))
                        if unique:
                            if str(x) + str(y) not in self.augmented_paths:
                                train_x.append(x)
                                train_y.append(y)
                                self.augmented_paths.add(str(x) + str(y))
                        else:
                            train_x.append(x)
                            train_y.append(y)
        return numpy.array(train_x, dtype='float32'), numpy.array(train_y, dtype='float32')

    def get_valid_dialog_acts(self, state: List[List[float]]):
        current_state = str([int(value) for value in state[-1]])
        dialog_acts = [eval(node)[-len(self.dialog_act_to_idx):] for node in self.graph[current_state]]
        return dialog_acts

    def get_best_f1_score(self, state: List[List[float]], y_pred: List[float]) -> Tuple[List[int], float]:
        dialog_acts = self.get_valid_dialog_acts(state)
        f1_scores = []
        for y_true in dialog_acts:
            f1_scores.append(f1_score(y_pred=y_pred, y_true=y_true))
        return dialog_acts[numpy.argmax(f1_scores)], max(f1_scores)


# train_graph = MultiWozConvGraph(dir_name="./", file_names=['train.json'])
# train_graph.generate_standard_data(unique=False)
# train_graph.generate_augmented_data(to_json=True)
