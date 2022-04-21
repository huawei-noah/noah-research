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
import logging
import argparse
from enum import Enum
import torch
import numpy as np
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from xlm_ra import XNLUModel, get_intent_labels
from sklearn.metrics import classification_report as sk_report
from seqeval.metrics import classification_report as seq_report, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'large': (XLMRobertaModel, XNLUModel),
    'base': (XLMRobertaModel, XNLUModel)
}

MODEL_PATH_MAP = {
    'large': '../xlm-roberta-large/',
    'base': '../xlm-roberta-base/'
}


class Tasks(Enum):
    MTOD = 'mtod'
    MTOP = 'mtop'
    M_ATIS = 'm_atis'


def load_tokenizer(model_name_or_path):
    return XLMRobertaTokenizer.from_pretrained(model_name_or_path)


# noinspection PyArgumentList
def init_logger(args):
    if not os.path.exists(os.path.join(args.model_dir)):
        os.mkdir(os.path.join(args.model_dir))
    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.model_dir, "log.log"), "a+", "utf-8"), logging.StreamHandler()],
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


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

    joint_result, debug_dump = [], {}
    intent_map = get_intent_labels(args) if args.debug else None
    for s_preds, s_labels, i_pred, i_label, i in zip(slot_predictions, slot_labels, intent_predictions, intent_labels, guids):
        assert len(s_preds) == len(s_labels)
        joint_correct = i_pred == i_label and s_preds == s_labels
        joint_result.append(joint_correct)

        if args.debug:
            debug_dump[i] = (examples[i].words, examples[i].slot_labels, s_preds, s_labels, i_pred, i_label)
            if not joint_correct:
                logger.info("-" * 50)
                logger.info("SENTENCE: " + "".join([w.replace("▁", " ") for w in examples[i].words]))
                s_preds_iter, s_labels_iter = iter(s_preds), iter(s_labels)
                logger.info("WORD".ljust(25) + "PREDICTION".ljust(25) + "LABEL".ljust(25))
                for word, slot_type in zip(examples[i].words, examples[i].slot_labels):
                    if slot_type != -100:
                        p, l = next(s_preds_iter), next(s_labels_iter)
                        logger.info("".join([word.ljust(25), p.ljust(25), l.ljust(25), str(p == l)]))
                    else:
                        logger.info(word.ljust(25) + "PAD".ljust(25) + "PAD".ljust(25))
                logger.info("Intent (pred, label, correct?): " + ", ".join([intent_map[i_pred], intent_map[i_label], str(i_pred == i_label)]))

    if args.debug:
        logger.info("\n" + seq_report(slot_labels, slot_predictions, zero_division=0))
        logger.info("\n" + sk_report([intent_map[i] for i in intent_labels],
                                     [intent_map[i] for i in intent_predictions], zero_division=0))
        pickle.dump(debug_dump, open(os.path.join(args.model_dir, "debug_dump.pkl"), "wb"))

    return {"Joint_Accuracy": np.array(joint_result).mean()}


def set_aux_losses(args):
    """
        Parse the auxiliary losses to use in alignment
    """
    num_aux_losses = 0

    loss_keys = {
        "XA": "use_xeroalign",
        "CA": "use_crossaligner",
        "CTR": "use_contrastive",
        "TI": "use_translate_intent",
    }

    losses_to_use = {}
    arguments = []
    if args.use_aux_losses:
        arguments = args.use_aux_losses[0]
    for loss in loss_keys.keys():
        loss_selected = loss in arguments
        losses_to_use.update({loss_keys[loss]: loss_selected})
        num_aux_losses += int(loss_selected)

    return num_aux_losses, argparse.Namespace(**losses_to_use)


def set_weighting_method(args):
    """
        Parse weighting method to use when there are multiple auxiliary losses
    """
    weighting_methods = {
        "COV": "use_cov",
    }
    weighting_to_use = {}
    complete = False
    arguments = []
    if args.use_weighting:
        arguments = [arg.strip() for arg in args.use_weighting.split(',')]
    for method in weighting_methods.keys():
        if not complete and method in arguments:
            weighting_to_use.update({weighting_methods[method]: True})
            complete = True
        else:
            weighting_to_use.update({weighting_methods[method]: False})

    return argparse.Namespace(**weighting_to_use)
