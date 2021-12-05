# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
import torch

from lib.metrics imp    ort _validate_ood_labels, _maybe_cast_torch_objects_to_numpy, _validate_sizes, \
    classification_accuracy, ood_classification_accuracy, roc_auc, roc_aupr, fpr_at_x_tpr, \
    classification_f1_macro_score, classification_f1_micro_score, swap_labels_scores
import pytest


def test_converting_from_torch_to_numpy():
    data = np.array([1, 3, 1.2])
    labels = np.array([0, 4, 4])
    assert np.array_equal(_maybe_cast_torch_objects_to_numpy(data, labels), (data, labels))
    assert np.array_equal(_maybe_cast_torch_objects_to_numpy(torch.tensor(data), labels), (data, labels))
    assert np.array_equal(_maybe_cast_torch_objects_to_numpy(data, torch.tensor(labels)), (data, labels))


def test_ood_labels_validation():
    correct_labels_1 = [0, 0, 1, 0, 1]
    _validate_ood_labels(correct_labels_1)
    _validate_ood_labels(np.array(correct_labels_1))

    correct_labels_2 = [0, 0.0, 1.0]
    _validate_ood_labels(correct_labels_2)

    incorrect_labels_1 = [0, 0, 0, -1]
    with pytest.raises(RuntimeError):
        _validate_ood_labels(incorrect_labels_1)


def test_sizes_validation():
    logits = np.random.rand(12)
    labels = np.random.rand(12)

    _validate_sizes(logits, labels)
    with pytest.raises(RuntimeError):
        _validate_sizes(logits[:-1], labels)
    with pytest.raises(RuntimeError):
        _validate_sizes(logits.reshape(12, 1), labels)

    logits = np.random.rand(12, 4)
    _validate_sizes(logits, labels, only_batch_size=True)
    with pytest.raises(RuntimeError):
        _validate_sizes(logits[:-1], labels, only_batch_size=True)

    with pytest.raises(RuntimeError):
        _validate_sizes(np.array([]), np.array([]), only_batch_size=True)


def test_classification_accuracy():
    predictions = [1, 0, 1, 0]
    assert classification_accuracy(predictions, [0, 0, 0, 0]) == 0.5
    assert classification_accuracy(predictions, np.array([0, 0, 0, 0])) == 0.5
    assert classification_accuracy(np.array(predictions), [1, 0, 1, 0]) == 1.0
    assert classification_accuracy(predictions, [1, 0, 0, 0]) == 3 / 4


def test_f1_macro():
    predictions = [1, 1, 1, 2, 2, 3]
    assert classification_f1_macro_score(predictions, [1, 1, 1, 2, 2, 3]) == 1.0
    assert classification_f1_macro_score(predictions, [1, 1, 1, 2, 2, 1]) == (1.5 / 1.75 + 1 + 0) / 3
    assert classification_f1_macro_score(predictions, [3, 3, 3, 2, 2, 1]) == (0 + 1 + 0) / 3


def test_f1_micro():
    predictions = [1, 1, 1, 2, 2, 3]
    assert classification_f1_micro_score(predictions, [1, 1, 1, 2, 2, 3]) == 1.0
    assert classification_f1_micro_score(predictions, [1, 1, 1, 2, 2, 1]) == 2 * (5 / 6 * 5 / 6) / (5 / 6 + 5 / 6)
    assert classification_f1_micro_score(predictions, [3, 3, 3, 2, 2, 1]) == 2 * (2 / 6 * 2 / 6) / (2 / 6 + 2 / 6)


# Perfect classifier
OOD_LOGITS_1 = [i / 10 for i in range(10)]
OOD_LABELS_1 = [0] * 5 + [1] * 5

# Random classifier
OOD_LOGITS_2 = [0.1, 0.5] * 6
OOD_LABELS_2 = [0] * 6 + [1] * 6

# Classifier with outlier
OOD_LOGITS_3 = [i / 10 for i in range(9)] + [-1]
OOD_LABELS_3_1 = [0] * 5 + [1] * 5
OOD_LABELS_3_2 = [0] * 9 + [1]


def test_ood_classification_accuracy():
    assert ood_classification_accuracy(OOD_LOGITS_1, OOD_LABELS_1, 0.5) == 1.0
    assert ood_classification_accuracy(OOD_LOGITS_1, OOD_LABELS_1, 0.0) == 0.5
    assert ood_classification_accuracy(OOD_LOGITS_1, OOD_LABELS_1, 0.7) == 0.8
    assert ood_classification_accuracy(OOD_LOGITS_1, OOD_LABELS_1, 1.0) == 0.5
    assert ood_classification_accuracy(OOD_LOGITS_2, OOD_LABELS_2, 0.3) == 0.5


def test_auc_roc():
    assert roc_auc(OOD_LOGITS_1, OOD_LABELS_1) == 1.0
    assert roc_auc(OOD_LOGITS_2, OOD_LABELS_2) == 0.5
    assert roc_auc(OOD_LOGITS_3, OOD_LABELS_3_1) == 0.8
    assert roc_auc(OOD_LOGITS_3, OOD_LABELS_3_2) == 0.0


def test_aupr():
    assert roc_aupr(OOD_LOGITS_1, OOD_LABELS_1) == 1.0
    assert roc_aupr(OOD_LOGITS_2, OOD_LABELS_2) == 0.5
    assert np.allclose(roc_aupr(OOD_LOGITS_3, OOD_LABELS_3_1), 0.9)
    assert roc_aupr(OOD_LOGITS_3, OOD_LABELS_3_2) == 0.1


def test_fpr_tpr():
    assert fpr_at_x_tpr(OOD_LOGITS_1, OOD_LABELS_1, 0.5) == 0.0
    assert fpr_at_x_tpr(OOD_LOGITS_1, OOD_LABELS_1, 0.0) == 0.0
    assert fpr_at_x_tpr(OOD_LOGITS_1, OOD_LABELS_1, 1.0) == 0.0

    assert fpr_at_x_tpr(OOD_LOGITS_1, 1 - np.array(OOD_LABELS_1), 1.0) == 1.0
    assert fpr_at_x_tpr(OOD_LOGITS_1, 1 - np.array(OOD_LABELS_1), 0.4) == 1.0
    assert fpr_at_x_tpr(OOD_LOGITS_1, [0, 0, 0, 0, 1, 0, 1, 1, 1, 1], 0.4) == 0.0

    assert fpr_at_x_tpr(OOD_LOGITS_1, [1, 0, 0, 0, 1, 0, 0, 1, 1, 1], 0.8) == 0.4
    assert fpr_at_x_tpr(OOD_LOGITS_1, [1, 0, 0, 0, 1, 0, 0, 1, 1, 1], 0.6) == 0.0

    assert fpr_at_x_tpr([0.1, 0.1, 0.2, 0.2, 0.3], [0, 0, 1, 1, 1], 95) == 0.0
    assert fpr_at_x_tpr(OOD_LOGITS_1, 1 - np.array(OOD_LABELS_1), 100) == 1.0

    with pytest.raises(AssertionError):
        fpr_at_x_tpr([0.1, 0.1, 0.2, 0.2, 0.3], [0, 0, 1, 1, 1], 95.0)


def test_swap_labels_scores():
    labels = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    scores, swapped_labels = swap_labels_scores(np.array(OOD_LOGITS_1), labels)
    assert all([x == -y for x, y in zip(OOD_LOGITS_1, scores)])
    assert all([(x == 1) or (x == 0) for x in swapped_labels])
    assert all([x != y for x, y in zip(swapped_labels, labels)])


@pytest.mark.cuda
def test_validation_cuda():
    clf_logits = np.array([[0.1, 0.4], [0.9, 0.1], [0.4, 0.8], [0.9, 0.8]])
    cuda_clf_logits = torch.tensor(clf_logits).cuda()
    clf_labels = np.array([1, 0, 0, 0])
    cuda_clf_labels = torch.tensor(clf_labels).cuda()
    casted_results = _maybe_cast_torch_objects_to_numpy(cuda_clf_logits, cuda_clf_labels)
    assert np.array_equal(casted_results[0], clf_logits)
    assert np.array_equal(casted_results[1], clf_labels)
    assert classification_accuracy(cuda_clf_logits.argmax(-1), clf_labels) == 3 / 4
