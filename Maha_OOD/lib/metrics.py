# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from bisect import bisect_right
from logging import warning
from typing import Union

import torch
from numpy import asarray, where
from sklearn.metrics import accuracy_score, roc_curve, auc, average_precision_score, f1_score


def _maybe_cast_torch_objects_to_numpy(logits, labels):
    """
    Casts objects to Numpy array
    :param logits: ood or classification logits
    :param labels: ood or classification labels
    :return: casted logits and labels
    """
    if isinstance(logits, torch.Tensor):
        warning("Better not to pass torch tensors for logits. Too much copyting from GPU")
        logits = logits.detach().cpu().numpy()
    return asarray(logits), asarray(labels)


def _validate_ood_labels(labels):
    """Ensures that labels are either 0 or 1. Accepts lists and numpy arrays"""
    labels = asarray(labels)
    if not ((labels == 0) | (labels == 1)).all():
        raise RuntimeError("OOD labels can only be 0 or 1")


def _validate_sizes(logits, labels, only_batch_size=False):
    """Checks if sizes are same, if `only_batch_size` is True checks only first dimension"""
    if not logits.size or not labels.size:
        raise RuntimeError("Passed empty array to metric")
    if not only_batch_size:
        if logits.shape != labels.shape:
            raise RuntimeError("Predictions and labels should have same shape")
    else:
        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError("Predictions and labels should have same batch size")


def classification_accuracy(predictions, labels):
    """
    Classification accuracy metric
    :param logits: classification predictions: batch_size X 1
    :param labels: classification labels: batch_size X 1
    :return: accuracy score
    """
    predictions, labels = _maybe_cast_torch_objects_to_numpy(predictions, labels)
    _validate_sizes(predictions, labels)
    return accuracy_score(predictions.flatten(), labels.flatten())


def classification_f1_macro_score(predictions, labels):
    predictions, labels = _maybe_cast_torch_objects_to_numpy(predictions, labels)
    _validate_sizes(predictions, labels)
    return f1_score(labels, predictions, average='macro')


def classification_f1_micro_score(predictions, labels):
    predictions, labels = _maybe_cast_torch_objects_to_numpy(predictions, labels)
    _validate_sizes(predictions, labels)
    return f1_score(labels, predictions, average='micro')


def _cast_and_validate_ood(ood_scores, labels):
    """Combine validation helpers for OOD metrics"""
    ood_scores, labels = _maybe_cast_torch_objects_to_numpy(ood_scores, labels)
    _validate_ood_labels(labels)
    _validate_sizes(ood_scores, labels)
    return ood_scores, labels


def ood_classification_accuracy(ood_scores, labels, threshold):
    """
    Classification accuracy metric for OOD task
    :param ood_scores: OOD certainty scores: batch_size X 1
    :param labels: OOD labels, 1 for OOD, 0 for in-domain: batch_size X 1
    :param threshold: decision rule for `ood_scores`
    :return: OOD classification accuracy
    """
    ood_scores, labels = _cast_and_validate_ood(ood_scores, labels)
    ood_predictions = ood_scores >= threshold
    return accuracy_score(ood_predictions, labels)


def roc_auc(ood_scores, labels, swap_labels: bool = False):
    """
    Area under ROC curve for OOD task
    :param ood_scores: OOD certainty scores: batch_size X 1
    :param labels: OOD labels, 1 for OOD, 0 for in-domain: batch_size X 1
    :param swap_labels: whether to swap labels, i.e. positive class would be a negative and vice versa.
    :return: AUROC
    """
    ood_scores, labels = _cast_and_validate_ood(ood_scores, labels)
    if swap_labels:
        ood_scores, labels = swap_labels_scores(ood_scores, labels)
    fpr, tpr, _ = roc_curve(labels, ood_scores)
    return auc(fpr, tpr)


def roc_aupr(ood_scores, labels, swap_labels: bool = False):
    """
    Area under PR curve for OOD task
    :param ood_scores: OOD certainty scores: batch_size X 1
    :param labels: OOD labels, 1 for OOD, 0 for in-domain: batch_size X 1
    :param swap_labels: whether to swap labels, i.e. positive class would be a negative and vice versa.
    :return: AUPR
    """
    ood_scores, labels = _cast_and_validate_ood(ood_scores, labels)
    if swap_labels:
        ood_scores, labels = swap_labels_scores(ood_scores, labels)
    return average_precision_score(labels, ood_scores)


def _custom_bisect(tpr, tpr_level):
    idx = bisect_right(tpr, tpr_level)
    while idx > -1 and tpr[idx - 1] >= tpr_level:
        idx -= 1
    return idx


def fpr_at_x_tpr(ood_scores, labels, tpr_level: Union[int, float], swap_labels: bool = False):
    """
    Computer False Positive rate (1 - in-domain recall) at fixed True Positive rate (OOD recall)
    :param ood_scores: OOD certainty scores: batch_size X 1
    :param labels: OOD labels, 1 for OOD, 0 for in-domain: batch_size X 1
    :param tpr_level: OOD recall, 0-100 for int arg, 0.0-1.0 for float arg
    :param swap_labels: whether to swap labels, i.e. positive class would be a negative and vice versa.
    :return: FPR@{trp_level}TPR
    """
    assert isinstance(tpr_level, (int, float))
    if isinstance(tpr_level, int):
        assert 0 <= tpr_level <= 100
        tpr_level /= 100
    assert 0 <= tpr_level <= 1
    ood_scores, labels = _cast_and_validate_ood(ood_scores, labels)
    if swap_labels:
        ood_scores, labels = swap_labels_scores(ood_scores, labels)
    fpr, tpr, _ = roc_curve(labels, ood_scores, drop_intermediate=False)
    closest_index = _custom_bisect(tpr, tpr_level)
    idx = max(closest_index, 0)
    idx = min(idx, len(fpr) - 1)
    return fpr[idx]


def swap_labels_scores(scores, labels):
    """
    Swaps positive class with negative one, revert scores order.
    :param scores: certainty scores
    :param labels: binary labels, 1 for positive class, 0 for negative class
    :return:
    """
    swapped_labels = where(labels, 0, 1)
    reverted_scores = -scores
    return reverted_scores, swapped_labels
