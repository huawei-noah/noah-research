# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch.nn import Module

from lib.score_functions import register_score


class AbstractMahalanobisScore(Module):
    def __init__(self, dim):
        super(AbstractMahalanobisScore, self).__init__()
        self.dim = dim
        self.register_buffer(
            'covariance_matrix',
            torch.eye(dim, dtype=torch.float)
        )

    def __call__(self, features):
        raise NotImplementedError

    def update(self, train_feats, train_labels):
        raise NotImplementedError

    def _check_scores(self, scores):
        if scores.dim() == 0:
            return scores.view(-1)
        return scores

    def update_inv_convmat(self, centered_feats):
        self.covariance_matrix.zero_()
        for feat in centered_feats:
            self.covariance_matrix += feat.view(-1, 1) @ feat.view(-1, 1).transpose(0, 1)
        self.covariance_matrix = self.covariance_matrix / centered_feats.shape[0]
        self.covariance_matrix = self.covariance_matrix.inverse()


@register_score('mahalanobis')
class MahalanobisScore(AbstractMahalanobisScore):
    def __init__(self, dim, num_labels):
        super(MahalanobisScore, self).__init__(dim)
        self.num_labels = num_labels
        self.register_buffer(
            'means',
            torch.zeros(self.num_labels, dim, dtype=torch.float)
        )

    def _get_min_dist(self, r):
        dist = r @ self.covariance_matrix @ r.transpose(2, 3)
        dist = dist.squeeze()
        min_dist, min_idx = dist.min(-1)
        return min_dist, min_idx

    def _get_centered_vectors(self, features):
        r = features.unsqueeze(1) - self.means.unsqueeze(0)
        r = r.unsqueeze(2)
        return r

    def __call__(self, features):
        r = self._get_centered_vectors(features)
        min_dist, _ = self._get_min_dist(r)
        return self._check_scores(min_dist)

    def update_means(self, train_feats, train_labels):
        self.means.zero_()
        label_cnt = torch.zeros(self.num_labels, 1, device=train_feats.device)
        for label, feat in zip(train_labels, train_feats):
            self.means[label] += feat
            label_cnt[label] += 1
        self.means.div_(label_cnt)

    def center_feats(self, train_feats, train_labels):
        centered_feats = torch.zeros_like(train_feats)
        for idx, (label, feat) in enumerate(zip(train_labels, train_feats)):
            centered_feats[idx] = feat - self.means[label]
        return centered_feats

    def update(self, train_feats, train_labels):
        self.update_means(train_feats, train_labels)
        centered_feats = self.center_feats(train_feats, train_labels)
        self.update_inv_convmat(centered_feats)


@register_score('euclidean')
class EuclideanDistanceScore(MahalanobisScore):
    def update(self, train_feats, train_labels):
        self.update_means(train_feats, train_labels)


@register_score('mahalanobis-pca')
class MahalanobisPCAScore(MahalanobisScore):
    def __init__(self, dim, num_labels, start_elem):
        super(MahalanobisPCAScore, self).__init__(dim, num_labels)
        self.start_elem = start_elem
        self.pca = PCA(n_components=dim).fit(np.random.randn(dim, dim))

    def update_pca(self, centered_feats):
        centered_feats = centered_feats.cpu().numpy()
        self.pca = PCA(n_components=self.dim).fit(centered_feats)

    def update(self, train_feats, train_labels):
        self.update_means(train_feats, train_labels)
        centered_feats = self.center_feats(train_feats, train_labels)
        self.update_pca(centered_feats)

    def __call__(self, features):
        r = self._get_centered_vectors(features)
        min_dist, min_idx = self._get_min_dist(r)
        r_centered = r.squeeze(2)[torch.arange(len(min_idx)), min_idx]
        r_components = self.pca.transform(r_centered.cpu().numpy())
        scores = np.power(r_components[:, self.start_elem:], 2) / \
                 self.pca.explained_variance_[self.start_elem:].reshape(1, -1)
        scores = torch.from_numpy(scores).sum(-1)
        return self._check_scores(scores)


@register_score('marginal-mahalanobis')
class MarginalMahalanobisScore(AbstractMahalanobisScore):
    def __init__(self, dim):
        super(MarginalMahalanobisScore, self).__init__(dim)
        self.register_buffer(
            'mean',
            torch.zeros(dim, dtype=torch.float)
        )

    def __call__(self, features):
        r = features - self.mean
        r = r.unsqueeze(1)
        dist = r @ self.covariance_matrix @ r.transpose(1, 2)
        return self._check_scores(dist.squeeze())

    def center_feats(self, train_feats):
        centered_feats = torch.zeros_like(train_feats)
        for idx, feat in enumerate(train_feats):
            centered_feats[idx] = feat - self.mean
        return centered_feats

    def update(self, train_feats, train_labels):
        self.mean = train_feats.mean(dim=0)
        centered_feats = self.center_feats(train_feats)
        self.update_inv_convmat(centered_feats)


@register_score('marginal-mahalanobis-pca')
class MarginalMahalanobisPCAScore(MarginalMahalanobisScore):
    def __init__(self, dim, start_elem):
        super(MarginalMahalanobisPCAScore, self).__init__(dim)
        self.start_elem = start_elem
        self.pca = PCA(n_components=dim).fit(np.random.randn(dim, dim))

    def __call__(self, features):
        r = features - self.mean
        r_components = self.pca.transform(r.cpu().numpy())
        scores = np.power(r_components[:, self.start_elem:], 2) / \
                 self.pca.explained_variance_[self.start_elem:].reshape(1, -1)
        ood_scores = torch.from_numpy(scores).sum(-1)
        return self._check_scores(ood_scores)

    def update_pca(self, centered_feats):
        centered_feats = centered_feats.cpu().numpy()
        self.pca = PCA(n_components=self.dim).fit(centered_feats)

    def update(self, train_feats, train_labels):
        self.mean = train_feats.mean(dim=0)
        centered_feats = self.center_feats(train_feats)
        self.update_pca(centered_feats)
