# evaluation/evals.py
"""Evaluate search results using F1, BLEU, and LLM Judge"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict
import sys
import concurrent.futures
import threading

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from tqdm import tqdm

from evaluation.metrics.utils import calculate_metrics, aggregate_metrics
from evaluation.metrics.llm_judge import evaluate_llm_judge


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_results(
    results: Dict[str, List[Dict[str, Any]]],
    use_llm_judge: bool = True,
    verbose: bool = False,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Evaluate all results with F1, BLEU, and LLM Judge metrics."""
    
    all_metrics = []
    all_categories = []
    llm_judge_by_category = defaultdict(list)
    detailed_results = defaultdict(list)
    
    # Also track latency statistics
    latency_stats = {
        "search_latencies": [],
        "llm_latencies": [],
        "e2e_latencies": [],
    }
    
    # Thread-safe locks
    metrics_lock = threading.Lock()
    
    def evaluate_single_qa(user_id: str, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single QA item."""
        question = qa_item.get("question", "")
        gold_answer = qa_item.get("answer", "")
        generated_answer = qa_item.get("response", "")
        category = int(qa_item.get("category", 0))
        
        # Skip category 5
        if category == 5:
            return {"skip": True}
        
        # Calculate F1 and BLEU metrics
        metrics = calculate_metrics(generated_answer, gold_answer)
        
        # LLM judge evaluation
        llm_label = 0
        llm_evaluated = False 
        if use_llm_judge and gold_answer and generated_answer:
            try:
                llm_label = evaluate_llm_judge(question, gold_answer, generated_answer)
                llm_evaluated = True
            except Exception as e:
                if verbose:
                    print(f"\n⚠️  LLM judge error for {user_id}: {e}")
                llm_evaluated = False
        
        return {
            "skip": False,
            "user_id": user_id,
            "question": question,
            "gt_answer": gold_answer,
            "response": generated_answer,
            "category": category,
            "metrics": metrics,
            "llm_label": llm_label,
            "llm_evaluated": llm_evaluated, 
            "search_latency_ms": qa_item.get("search_latency_ms", 0),
            "llm_latency_ms": qa_item.get("llm_latency_ms", 0),
            "e2e_latency_ms": qa_item.get("e2e_latency_ms", 0),
            "success": qa_item.get("success", False),
        }
    
    # Prepare all tasks
    total_questions = sum(len(qa_list) for qa_list in results.values())
    
    print(f"\n📊 Evaluating {total_questions} answers...")
    print("=" * 80)
    
    # Process in parallel
    with tqdm(total=total_questions, desc="Evaluating", ncols=80) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = []
            for user_id, qa_list in results.items():
                for qa_item in qa_list:
                    future = executor.submit(evaluate_single_qa, user_id, qa_item)
                    futures.append(future)
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    
                    if not result.get("skip", False):
                        with metrics_lock:
                            user_id = result["user_id"]
                            
                            # Collect metrics
                            all_metrics.append(result["metrics"])
                            all_categories.append(result["category"])
                            
                            # LLM judge 
                            if result.get("llm_evaluated", False):
                                llm_judge_by_category[result["category"]].append(result["llm_label"])
                            
                            # Latency stats
                            if result.get("success", False):
                                latency_stats["search_latencies"].append(result["search_latency_ms"])
                                latency_stats["llm_latencies"].append(result["llm_latency_ms"])
                                latency_stats["e2e_latencies"].append(result["e2e_latency_ms"])
                            
                            # Store detailed results
                            result_copy = {
                                "question": result["question"],
                                "gt_answer": result["gt_answer"],
                                "response": result["response"],
                                "category": result["category"],
                                "metrics": result["metrics"],
                                "llm_label": result["llm_label"],
                                "search_latency_ms": result["search_latency_ms"],
                                "llm_latency_ms": result["llm_latency_ms"],
                                "e2e_latency_ms": result["e2e_latency_ms"],
                            }
                            detailed_results[user_id].append(result_copy)
                    
                except Exception as e:
                    if verbose:
                        print(f"\n⚠️  Error processing result: {e}")
                
                pbar.update(1)
    
    # Convert defaultdict to regular dict
    detailed_results = dict(detailed_results)
    
    # Aggregate metrics
    print("\n" + "=" * 80)
    print("📈 Computing Aggregated Metrics")
    print("=" * 80)
    
    aggregated = aggregate_metrics(all_metrics, all_categories)
    
    # Add LLM judge results
    if use_llm_judge:
        print("\n🤖 LLM Judge Results:")
        print("-" * 80)
        
        all_llm_scores = []
        for category in sorted(llm_judge_by_category.keys()):
            scores = llm_judge_by_category[category]
            if scores:
                accuracy = np.mean(scores)
                all_llm_scores.extend(scores)
                print(f"  Category {category}: {accuracy:.4f} ({sum(scores)}/{len(scores)})")
                
                # Add to aggregated results
                if f"category_{category}" not in aggregated:
                    aggregated[f"category_{category}"] = {}
                
                aggregated[f"category_{category}"]["llm_judge_accuracy"] = {
                    "mean": accuracy,
                    "count": len(scores),
                    "correct": sum(scores),
                }
        
        if all_llm_scores:
            overall_accuracy = np.mean(all_llm_scores)
            print(f"\n  Overall: {overall_accuracy:.4f} ({sum(all_llm_scores)}/{len(all_llm_scores)})")
            
            aggregated["overall"]["llm_judge_accuracy"] = {
                "mean": overall_accuracy,
                "count": len(all_llm_scores),
                "correct": sum(all_llm_scores),
            }
    
    # Add latency statistics
    print("\n⏱️  Latency Statistics:")
    print("-" * 80)
    if latency_stats["search_latencies"]:
        search_avg = np.mean(latency_stats["search_latencies"])
        llm_avg = np.mean(latency_stats["llm_latencies"])
        e2e_avg = np.mean(latency_stats["e2e_latencies"])
        
        print(f"  Search Latency:    {search_avg:.2f}ms (±{np.std(latency_stats['search_latencies']):.2f})")
        print(f"  LLM Latency:       {llm_avg:.2f}ms (±{np.std(latency_stats['llm_latencies']):.2f})")
        print(f"  End-to-End:        {e2e_avg:.2f}ms (±{np.std(latency_stats['e2e_latencies']):.2f})")
        
        aggregated["latency"] = {
            "search_ms": {
                "mean": float(search_avg),
                "std": float(np.std(latency_stats["search_latencies"])),
                "min": float(np.min(latency_stats["search_latencies"])),
                "max": float(np.max(latency_stats["search_latencies"])),
            },
            "llm_ms": {
                "mean": float(llm_avg),
                "std": float(np.std(latency_stats["llm_latencies"])),
                "min": float(np.min(latency_stats["llm_latencies"])),
                "max": float(np.max(latency_stats["llm_latencies"])),
            },
            "e2e_ms": {
                "mean": float(e2e_avg),
                "std": float(np.std(latency_stats["e2e_latencies"])),
                "min": float(np.min(latency_stats["e2e_latencies"])),
                "max": float(np.max(latency_stats["e2e_latencies"])),
            },
        }
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 Overall Metrics Summary")
    print("=" * 80)
    
    if "overall" in aggregated:
        print("\nAccuracy Metrics:")
        if "llm_judge_accuracy" in aggregated["overall"]:
            llm_acc = aggregated["overall"]["llm_judge_accuracy"]
            print(f"  LLM Judge: {llm_acc['mean']:.4f}")
        
        print("\nOverlap Metrics:")
        if "f1" in aggregated["overall"]:
            print(f"  F1 Score:  {aggregated['overall']['f1']['mean']:.4f} (±{aggregated['overall']['f1']['std']:.4f})")
        
        for n in range(1, 5):
            key = f"bleu_{n}"
            if key in aggregated["overall"]:
                print(f"  BLEU-{n}:    {aggregated['overall'][key]['mean']:.4f} (±{aggregated['overall'][key]['std']:.4f})")
    
    print("=" * 80)
    
    return {
        "aggregated_metrics": aggregated,
        "detailed_results": detailed_results,
        "summary": {
            "total_questions": len(all_metrics),
            "categories_evaluated": sorted(set(all_categories)),
            "metrics_computed": ["f1", "bleu_1", "bleu_2", "bleu_3", "bleu_4", "llm_judge_accuracy"],
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate search results")
    parser.add_argument("--input", default="results/search_results.json")
    parser.add_argument("--output", default="results/evaluation_results.json")
    parser.add_argument("--config", default="evaluation/config.json")
    parser.add_argument("--no-llm-judge", action="store_true", help="Disable LLM judge evaluation")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel workers")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    config = load_config(config_path)
    
    # Load search results
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Evaluate
    evaluation_results = evaluate_results(
        results,
        use_llm_judge=not args.no_llm_judge,
        verbose=args.verbose or config.get("verbose", False),
        max_workers=args.max_workers,
    )
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()