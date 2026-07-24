# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from collections import defaultdict


class ContrastiveClusterSampler:
    def __init__(self, n_total_samples=30):
        self.n_total_samples = n_total_samples

    def sample(self, all_cases, all_cases_labels, all_cases_embeddings, all_cases_centers, failures, corrects, out_txt):
        """ Get anchor, negative, boundary paris samples."""
        failure_set = set(failures)
        cluster_info = defaultdict(lambda: {'all_idx': [], 'fail_idx': [], 'correct_idx': []})

        for idx, label in enumerate(all_cases_labels):
            cluster_info[label]['all_idx'].append(idx)
            if all_cases[idx] in failure_set:
                cluster_info[label]['fail_idx'].append(idx)
            else:
                cluster_info[label]['correct_idx'].append(idx)

        cluster_ids = list(cluster_info.keys())
        error_rates = np.array([
            len(cluster_info[cid]['fail_idx']) / len(cluster_info[cid]['all_idx'])
            if len(cluster_info[cid]['all_idx']) > 0 else 0
            for cid in cluster_ids
        ])

        weights = error_rates / (error_rates.sum() + 1e-9)
        cluster_quotas = np.round(weights * self.n_total_samples).astype(int)
        final_samples = []

        with out_txt.open('a', encoding='utf-8') as outf:
            outf.write(f"---- cluster_quotas: {cluster_quotas}\n")
            outf.write(f"---- weights: {weights}\n")
            outf.write(f"---- error_rates: {error_rates}\n")

        for i, cid in enumerate(cluster_ids):
            quota = cluster_quotas[i]
            if quota <= 0: continue

            c_fail_idx = cluster_info[cid]['fail_idx']
            c_correct_idx = cluster_info[cid]['correct_idx']
            center_vec = all_cases_centers[cid].reshape(1, -1)

            n_anchor = max(1, int(quota * 0.2)) if c_correct_idx else 0

            if n_anchor > 0:
                dist_correct = euclidean_distances(all_cases_embeddings[c_correct_idx], center_vec).flatten()
                closest_correct = np.argsort(dist_correct)[:n_anchor]
                for idx in closest_correct:
                    global_idx = c_correct_idx[idx]
                    final_samples.append((all_cases[global_idx], "anchor", cid))

            n_hard_neg = max(1, int(quota * 0.2)) if c_fail_idx else 0
            if n_hard_neg > 0:
                dist_fail = euclidean_distances(all_cases_embeddings[c_fail_idx], center_vec).flatten()
                closest_fail = np.argsort(dist_fail)[:n_hard_neg]
                for idx in closest_fail:
                    global_idx = c_fail_idx[idx]
                    final_samples.append((all_cases[global_idx], "hard_negative", cid))

            n_boundary_pairs = max(0, (quota - n_anchor - n_hard_neg) // 2)
            if n_boundary_pairs > 0 and c_fail_idx and c_correct_idx:
                pair_dist = euclidean_distances(all_cases_embeddings[c_correct_idx],
                                                all_cases_embeddings[c_fail_idx])
                flat_indices = np.argsort(pair_dist.flatten())[:n_boundary_pairs]

                for f_idx in flat_indices:
                    row = f_idx // pair_dist.shape[1]
                    col = f_idx % pair_dist.shape[1]

                    final_samples.append((all_cases[c_correct_idx[row]], "boundary_correct", cid))
                    final_samples.append((all_cases[c_fail_idx[col]], "boundary_fail", cid))

        return final_samples
