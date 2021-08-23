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

# Third Party Open Source Notice
# The starting point for this repo was cloned from [JointBERT](https://github.com/monologg/JointBERT).
# Some unmodified code that does not constitute the key methodology introduced in our paper remains in the codebase.

import os
import random
import logging
from enum import Enum
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from xlm_ra import XNLUModel, XPairModel, get_intent_labels
from sklearn.metrics import classification_report as sk_report
from seqeval.metrics import classification_report as seq_report, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'large': (XLMRobertaModel, XNLUModel),
    'base': (XLMRobertaModel, XNLUModel),
    'large_paws_x': (XLMRobertaModel, XPairModel),
    'base_paws_x': (XLMRobertaModel, XPairModel)
}

MODEL_PATH_MAP = {
    'large': '../xlm-roberta-large/',
    'base': '../xlm-roberta-base/',
    'large_paws_x': '../xlm-roberta-large/',
    'base_paws_x': '../xlm-roberta-base/'
}


class Tasks(Enum):
    MTOD = 'mtod'
    MTOP = 'mtop'
    PAWS_X = 'paws_x'
    M_ATIS = 'm_atis'


def load_tokenizer(model_name_or_path):
    return XLMRobertaTokenizer.from_pretrained(model_name_or_path)


def init_logger(args):
    if not os.path.exists(os.path.join(args.model_dir)):
        os.mkdir(os.path.join(args.model_dir))
    # noinspection PyArgumentList
    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.model_dir, "log.log"), "a+", "utf-8"), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda_device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_predictions, intent_labels, slot_predictions, slot_labels, examples, guids, args):
    assert len(intent_predictions) == len(intent_labels) == len(slot_predictions) == len(slot_labels)
    results = {}
    results.update({"Intent_Acc": (intent_predictions == intent_labels).mean()})
    results.update(get_slot_metrics(slot_predictions, slot_labels))
    results.update(get_joint_acc_and_errors(intent_predictions, intent_labels, slot_predictions, slot_labels, examples, guids, args))
    return results


def get_slot_metrics(predictions, labels):
    assert len(predictions) == len(labels)
    return {
        "Slot_Precision": precision_score(labels, predictions),
        "Slot_Recall": recall_score(labels, predictions),
        "Slot_F1": f1_score(labels, predictions)
    }

def get_joint_acc_and_errors(intent_predictions, intent_labels, slot_predictions, slot_labels, examples, guids, args):
    """For the cases that intent and all the slots are correct (in one sentence)"""

    joint_result = []
    intent_map = get_intent_labels(args)
    for s_preds, s_labels, i_pred, i_label, i in zip(slot_predictions, slot_labels, intent_predictions, intent_labels, guids):
        assert len(s_preds) == len(s_labels)
        one_sent_result = i_pred == i_label
        for p, l in zip(s_preds, s_labels):
            if p != l:
                one_sent_result = False

        joint_result.append(one_sent_result)
        if not one_sent_result and args.debug:
            logger.info("-" * 50)
            logger.info("SENTENCE: " + "".join([w.replace("▁", " ") for w in examples[i].words]))
            s_preds, s_labels = iter(s_preds), iter(s_labels)
            for word, slot_type in zip(examples[i].words, examples[i].slot_labels):
                if slot_type != -100:
                    p, l = next(s_preds), next(s_labels)
                    logger.info("".join([word.ljust(25), p.ljust(25), l.ljust(25), str(p == l)]))
                else:
                    logger.info(word.ljust(25) + "PAD")
            logger.info("Intent (pred, label): " + ", ".join([intent_map[i_pred], intent_map[i_label], str(i_pred == i_label)]))

    if args.debug:
        logger.info("\n" + seq_report(slot_labels, slot_predictions, zero_division=0))
        logger.info("\n" + sk_report([intent_map[i] for i in intent_labels],
                                     [intent_map[i] for i in intent_predictions], zero_division=0))

    return {"Joint_Accuracy": np.array(joint_result).mean()}
