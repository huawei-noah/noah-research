# evaluation/metrics/utils.py
"""Evaluation metrics utilities"""

from __future__ import annotations

from typing import Dict, List, Any
from collections import Counter
import numpy as np
import re


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenization."""
    return normalize_text(text).split()


def calculate_f1(prediction: str, reference: str) -> float:
    """Calculate token-level F1 score."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    common = pred_counter & ref_counter
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_bleu(prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
    """Calculate BLEU scores for n-grams up to max_n."""
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    
    scores = {}
    
    for n in range(1, max_n + 1):
        if len(pred_tokens) < n or len(ref_tokens) < n:
            scores[f"bleu_{n}"] = 0.0
            continue
        
        # Generate n-grams
        pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
        
        if not pred_ngrams or not ref_ngrams:
            scores[f"bleu_{n}"] = 0.0
            continue
        
        # Count matches
        pred_counter = Counter(pred_ngrams)
        ref_counter = Counter(ref_ngrams)
        
        common = pred_counter & ref_counter
        num_common = sum(common.values())
        
        precision = num_common / len(pred_ngrams) if pred_ngrams else 0.0
        scores[f"bleu_{n}"] = precision
    
    return scores


def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate all metrics for a prediction-reference pair.
    
    Returns:
        Dict with keys: f1, bleu_1, bleu_2, bleu_3, bleu_4
    """
    metrics = {}
    
    # F1 Score
    metrics["f1"] = calculate_f1(prediction, reference)
    
    # BLEU Scores
    bleu_scores = calculate_bleu(prediction, reference, max_n=4)
    metrics.update(bleu_scores)
    
    return metrics


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    all_categories: List[int],
) -> Dict[str, Any]:
    """Aggregate metrics across all examples.
    
    Args:
        all_metrics: List of metric dicts
        all_categories: List of category labels
        
    Returns:
        Dict with aggregated statistics per metric and per category
    """
    if not all_metrics:
        return {}
    
    aggregated = {"overall": {}, "by_category": {}}
    
    # Get all metric names
    metric_names = list(all_metrics[0].keys())
    
    # Overall statistics
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        aggregated["overall"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
    
    # Per-category statistics
    unique_categories = sorted(set(all_categories))
    for category in unique_categories:
        category_key = f"category_{category}"
        aggregated[category_key] = {}
        
        # Get metrics for this category
        category_metrics = [
            all_metrics[i] 
            for i, cat in enumerate(all_categories) 
            if cat == category
        ]
        
        if not category_metrics:
            continue
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in category_metrics]
            aggregated[category_key][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }
    
    return aggregated