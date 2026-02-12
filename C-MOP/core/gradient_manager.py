# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from collections import defaultdict
from sklearn.cluster import KMeans
import math
import numpy as np


class GradientBatcher:
    def __init__(self, samples_per_gradient=1):
        self.samples_per_gradient = samples_per_gradient

    def create_mini_batches(self, final_samples):
        cluster_buckets = defaultdict(lambda: {"anchors": [], "negatives": [], "boundary_pairs": []})
        bp_temp = defaultdict(list)

        for item in final_samples:
            sample_obj, role, cid = item
            if role == "anchor":
                cluster_buckets[cid]["anchors"].append(sample_obj)
            elif role == "hard_negative":
                cluster_buckets[cid]["negatives"].append(sample_obj)
            elif "boundary" in role:
                bp_temp[cid].append(item)

        for cid, items in bp_temp.items():
            for i in range(0, len(items), 2):
                if i + 1 < len(items):
                    cluster_buckets[cid]["boundary_pairs"].append((items[i][0], items[i + 1][0]))

        all_mini_batches = []

        for cid, data in cluster_buckets.items():
            bps = data["boundary_pairs"]
            negs = data["negatives"]
            ancs = data["anchors"]

            num_batches = max(1, len(bps) // 2)

            for i in range(num_batches):
                mini_batch = {
                    "cluster_id": cid,
                    "boundary_pairs": bps[i * 2: (i + 1) * 2],
                    "hard_negatives": negs[i * 2: (i + 1) * 2 + 1],
                    "anchors": ancs[i: i + 1]
                }
                if mini_batch["boundary_pairs"] or mini_batch["hard_negatives"]:
                    all_mini_batches.append(mini_batch)

        return all_mini_batches

    def format_meta_prompt(self, mini_batch, current_prompt):
        sections = [
            "### ROLE & OBJECTIVE",
            "You are a sophisticated Prompt Optimization Expert. Your task is to analyze model performance on a "
            "specific 'logical cluster' and extract a high-precision 'Textual Gradient' to improve the current "
            "prompt.",
            f"### CURRENT PROMPT\n```\n{current_prompt}\n```",
            f"### DIAGNOSTIC SAMPLES (Cluster ID: {mini_batch['cluster_id']})",
            "These samples represent a specific failure pattern. Analyze the contrast between correct and incorrect "
            "responses carefully. "
        ]

        if mini_batch["boundary_pairs"]:
            sections.append("\n#### [TYPE A] Critical Boundary Pairs")
            sections.append(
                "The following pairs are semantically or structurally similar, yet the model succeeded on one and "
                "failed on the other. This indicates a 'fuzzy zone' in the current prompt's logic.")
            for i, (corr, fail) in enumerate(mini_batch["boundary_pairs"]):
                sections.append(f"Pair {i + 1}:")
                sections.append(f" - [SIMILAR & CORRECT]: {corr}")
                sections.append(f" - [SIMILAR & INCORRECT]: {fail}")
                sections.append(
                    f" - Analysis Point: Identify the exact missing constraint that caused the failure in the second "
                    f"case.")

        if mini_batch["hard_negatives"]:
            sections.append("\n#### [TYPE B] Representative Failures (Hard Negatives)")
            sections.append(
                "These cases represent the 'centroid' of errors in this cluster. They share a systematic logical flaw.")
            for s in mini_batch["hard_negatives"]:
                sections.append(f" - Failure Case: {s}")

        if mini_batch["anchors"]:
            sections.append("\n#### [TYPE C] Success Anchors")
            sections.append(
                "The following cases are handled correctly. Your optimization MUST ensure these remain correct (avoid "
                "catastrophic forgetting).")
            for s in mini_batch["anchors"]:
                sections.append(f" - Anchor Case: {s}")

        sections.extend([
            "\n### FINAL ANALYSIS TASK",
            "1. Identify the common reasoning failure in this specific cluster.",
            "2. Focus on the 'Boundary Pairs': What subtle nuance did the model miss compared to the correct "
            "counterpart?",
            "3. Propose **three** precise, atomic 'Textual Gradients' (prompt modifications or additional "
            "constraints) that fix these errors while preserving the Anchors.",

            "\n### OUTPUT FORMAT",
            "Please give **three** reasons ('Textual Gradients') why the prompt could have gotten these examples "
            "wrong and wrap each reason ('Textual Gradient') with <START> and <END> tags !!",
            "Please provide your insight in the following format:",
            "<START>Your concise, technical insight into the prompt's missing logic 1<END><START>Your concise, "
            "technical insight into the prompt's missing logic 2<END><START>Your concise, technical insight into the "
            "prompt's missing logic 3<END> "
        ])
        return "\n".join(sections)


class GradientMomentumManager:
    def __init__(self, gamma=0.9, min_weight=0.2):
        self.gamma = gamma
        self.min_weight = min_weight
        self.buffer = []
        self.buffer_albation = {}

    def add_new_gradients(self, new_gradient_texts,
                          gradient_embeddings, current_step):

        if current_step in self.buffer_albation.keys():
            self.buffer_albation[current_step].extend(new_gradient_texts)
        else:
            self.buffer_albation[current_step] = new_gradient_texts
        for g in self.buffer:
            g["weight"] *= math.pow(self.gamma, current_step)

        self.buffer = [g for g in self.buffer if g["weight"] > self.min_weight]
        for text, vec in zip(new_gradient_texts, gradient_embeddings):
            self.buffer.append({
                "text": text,
                "vector": vec,
                "weight": 1.0
            })

    def get_momentum_gradients(self, n_clusters, out_txt):
        if not self.buffer:
            return []

        vecs = np.array([g["vector"] for g in self.buffer])
        weights = np.array([g["weight"] for g in self.buffer])
        texts = [g["text"] for g in self.buffer]

        n_samples = len(vecs)
        k = n_clusters if n_clusters else min(5, max(1, n_samples // 3))

        if n_samples < 2:
            return [{"texts": [texts[0]], "intensity": weights[0]}]
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(vecs)

        cluster_data = defaultdict(lambda: {"indices": [], "total_weight": 0.0})
        for i, label in enumerate(labels):
            cluster_data[label]["indices"].append(i)
            cluster_data[label]["total_weight"] += weights[i]
        final_momentum_results = []

        final_gradients = []
        sorted_clusters = sorted(cluster_data.items(), key=lambda x: x[1]["total_weight"], reverse=True)

        for cluster_id, info in sorted_clusters:
            indices = np.array(info["indices"])
            cluster_weights = weights[indices]
            sorted_idx_within_cluster = np.argsort(cluster_weights)[::-1]
            top_k = min(2, len(sorted_idx_within_cluster))
            top_indices = indices[sorted_idx_within_cluster[:top_k]]

            top_texts = [texts[idx] for idx in top_indices]

            final_momentum_results.append({
                "cluster_id": cluster_id,
                "texts": top_texts,
                "intensity": info["total_weight"]
            })
            final_gradients.extend(top_texts)

        with out_txt.open('a', encoding='utf-8') as outf:
            outf.write(f"++++++ final_momentum_results :{final_momentum_results}\n")

        return final_gradients
