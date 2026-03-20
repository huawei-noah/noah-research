# evaluation/generate_scores.py
"""Generate final scores and summary report"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def load_config(path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def generate_markdown_report(
    evaluation_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a markdown report of evaluation results."""
    
    aggregated = evaluation_results.get("aggregated_metrics", {})
    summary = evaluation_results.get("summary", {})
    
    # Read model configuration from environment
    llm_model = os.getenv("LLM_MODEL", "N/A")
    embedding_model = os.getenv("EMBEDDING_MODEL", "N/A")
    embedding_dim = os.getenv("EMBEDDING_DIM", "N/A")
    
    lines = [
        "# TridentMem LoCoMo Evaluation Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Configuration",
        "",
        f"- **LLM Model**: {llm_model}",
        f"- **Embedding Model**: {embedding_model}",
        f"- **Embedding Dimension**: {embedding_dim}",
        "",
        "## Summary",
        "",
        f"- **Total Questions Evaluated**: {summary.get('total_questions', 0)}",
        f"- **Categories**: {', '.join(map(str, summary.get('categories_evaluated', [])))}",
        "",
        "## Overall Results",
        "",
    ]
    
    if "overall" in aggregated:
        lines.append("| Metric | Mean | Std | Min | Max |")
        lines.append("|--------|------|-----|-----|-----|")
        
        for metric_name, stats in sorted(aggregated["overall"].items()):
            if isinstance(stats, dict) and "mean" in stats:
                lines.append(
                    f"| {metric_name} | "
                    f"{stats['mean']:.4f} | "
                    f"{stats.get('std', 0):.4f} | "
                    f"{stats.get('min', 0):.4f} | "
                    f"{stats.get('max', 0):.4f} |"
                )
        
        lines.append("")
    
    # Category breakdown
    lines.extend([
        "## Results by Category",
        "",
    ])
    
    categories = sorted([k for k in aggregated.keys() if k.startswith("category_")])
    
    for category_key in categories:
        category_num = category_key.replace("category_", "")
        lines.append(f"### Category {category_num}")
        lines.append("")
        
        category_data = aggregated[category_key]
        
        # Check for LLM judge results
        if "llm_judge_accuracy" in category_data:
            llm_stats = category_data["llm_judge_accuracy"]
            lines.append(
                f"**LLM Judge Accuracy**: {llm_stats['mean']:.4f} "
                f"({llm_stats.get('correct', 0)}/{llm_stats.get('count', 0)})"
            )
            lines.append("")
        
        lines.append("| Metric | Mean | Std |")
        lines.append("|--------|------|-----|")
        
        for metric_name, stats in sorted(category_data.items()):
            if isinstance(stats, dict) and "mean" in stats and metric_name != "llm_judge_accuracy":
                lines.append(
                    f"| {metric_name} | "
                    f"{stats['mean']:.4f} | "
                    f"{stats.get('std', 0):.4f} |"
                )
        
        lines.append("")
    
    # Write report
    report_text = "\n".join(lines)
    output_path.write_text(report_text, encoding="utf-8")


def generate_json_scores(
    evaluation_results: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate a simplified JSON with just the key scores."""
    
    aggregated = evaluation_results.get("aggregated_metrics", {})
    
    scores = {
        "overall": {},
        "by_category": {},
    }
    
    # Extract overall scores
    if "overall" in aggregated:
        for metric_name, stats in aggregated["overall"].items():
            if isinstance(stats, dict) and "mean" in stats:
                scores["overall"][metric_name] = {
                    "mean": round(stats["mean"], 4),
                    "std": round(stats.get("std", 0), 4),
                }
    
    # Extract category scores
    categories = sorted([k for k in aggregated.keys() if k.startswith("category_")])
    for category_key in categories:
        category_num = category_key.replace("category_", "")
        scores["by_category"][category_num] = {}
        
        category_data = aggregated[category_key]
        for metric_name, stats in category_data.items():
            if isinstance(stats, dict) and "mean" in stats:
                scores["by_category"][category_num][metric_name] = {
                    "mean": round(stats["mean"], 4),
                    "std": round(stats.get("std", 0), 4),
                }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final scores and reports")
    parser.add_argument("--input", default="results/evaluation_results.json")
    parser.add_argument("--config", default="evaluation/config.json")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    
    # Load config (but we won't use model configs from it)
    config_path = Path(args.config)
    config = load_config(config_path)
    
    # Load evaluation results
    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        evaluation_results = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate markdown report
    md_path = output_dir / "evaluation_report.md"
    generate_markdown_report(evaluation_results, md_path)
    print(f"📄 Markdown report saved to: {md_path}")
    
    # Generate JSON scores
    scores_path = output_dir / "final_scores.json"
    generate_json_scores(evaluation_results, scores_path)
    print(f"📊 Final scores saved to: {scores_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print("🎯 Final Results Summary")
    print("=" * 80)
    
    aggregated = evaluation_results.get("aggregated_metrics", {})
    if "overall" in aggregated:
        if "llm_judge_accuracy" in aggregated["overall"]:
            llm_acc = aggregated["overall"]["llm_judge_accuracy"]
            print(f"\n🤖 LLM Judge Accuracy: {llm_acc['mean']:.4f}")
        
        if "f1" in aggregated["overall"]:
            f1 = aggregated["overall"]["f1"]
            print(f"📊 F1 Score: {f1['mean']:.4f} (±{f1.get('std', 0):.4f})")
    
    print("=" * 80)


if __name__ == "__main__":
    main()