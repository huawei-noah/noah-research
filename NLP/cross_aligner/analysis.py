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
import random
from seqeval.metrics import classification_report as seq_report
from sklearn.metrics import classification_report as sk_report

# Generate debug files by running e.g. './config/m_atis.sh m_atis_eval large' with your chosen language(s)
task = "m_atis"
target = "debug_" + task + "_target_es"
aligned = "debug_" + task + "_aligned_es"
random.seed(123456789)
print("Processing task: %s, target file: %s, aligned file: %s" % (task, target, aligned))
target_dump = pickle.load(open(os.path.join(target, "debug_dump.pkl"), "rb"))
aligned_dump = pickle.load(open(os.path.join(aligned, "debug_dump.pkl"), "rb"))
intent_map = pickle.load(open(os.path.join("data", task, "intents.pkl"), 'rb'))

assert len(target_dump) == len(aligned_dump)
total_slots_disagreed = 0
intent_prediction_agreement = 0
aligned_slots_preds, aligned_slot_labels = [], []
intent_predictions_target, intent_predictions_aligned = [], []
keys = list(target_dump.keys())
random.shuffle(keys)

for i in keys:
    t_example, a_example = target_dump[i], aligned_dump[i]
    s_preds_target, s_labels_target, i_pred_target, i_label_target = t_example[2], t_example[3], t_example[4], t_example[5]
    s_preds_aligned, s_labels_aligned, i_pred_aligned, i_label_aligned = a_example[2], a_example[3], a_example[4], a_example[5]
    assert len(s_preds_target) == len(s_preds_aligned)
    assert s_labels_target == s_labels_aligned

    if not s_preds_aligned == s_preds_target:
        print("-" * 50)
        total_slots_disagreed += 1
        print("INTENT:  " + intent_map[i_label_target])
        print("SENTENCE: " + "".join([w.replace("▁", " ") for w in t_example[0]]))
        s_preds_iter, s_labels_iter = iter(s_preds_aligned), iter(s_labels_target)
        print("WORD".ljust(25) + "PREDICTION".ljust(25) + "LABEL".ljust(25))
        for word, slot_type in zip(a_example[0], a_example[1]):
            if slot_type != -100:
                p, l = next(s_preds_iter), next(s_labels_iter)
                print("".join([word.ljust(25), p.ljust(25), l.ljust(25), str(p == l)]))
            else:
                print(word.ljust(25) + "PAD".ljust(25) + "PAD".ljust(25))
        aligned_slots_preds.append(s_preds_aligned)
        aligned_slot_labels.append(s_labels_aligned)

    if i_pred_aligned == i_pred_target:
        intent_prediction_agreement += 1
    intent_predictions_target.append(intent_map[i_pred_target])
    intent_predictions_aligned.append(intent_map[i_pred_aligned])

print(sk_report(intent_predictions_target, intent_predictions_aligned, zero_division=0))
print(seq_report(aligned_slot_labels, aligned_slots_preds, zero_division=0))
print("-" * 80)
print("%.1f percent slots agreement (%d out of %d) for aligned & target." % (100 * (1 - (total_slots_disagreed / float(len(keys)))), len(keys) - total_slots_disagreed, len(keys)))
print("%.1f percent intent agreement (%d out of %d) for aligned & target." % (100 * (intent_prediction_agreement / float(len(keys))), intent_prediction_agreement, len(keys)))
print("-" * 80)
