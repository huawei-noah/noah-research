#!/usr/bin/env python3
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""
Regenerate ML description_lite.md files with improved quality:
1. Simplified task description (remove irrelevant info)
2. Clear variable definitions and objectives
3. Clear submission data structure
4. Clear dataset structure
5. Simplified background
"""

from pathlib import Path
import re

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks" / "ml" / "mlebench"
MLEBENCH_DATA = Path("./data/mlebench_all_data")


def extract_sections(text: str) -> dict:
    """Extract key sections from original description."""
    sections = {}
    
    # Try to find Overview/Description
    desc_match = re.search(r'# Overview.*?## Description\s*\n(.*?)(?=##|# |\Z)', text, re.DOTALL)
    if desc_match:
        sections['description'] = desc_match.group(1).strip()
    else:
        # Try alternative patterns
        desc_match = re.search(r'## Description\s*\n(.*?)(?=##|# |\Z)', text, re.DOTALL)
        if desc_match:
            sections['description'] = desc_match.group(1).strip()
    
    # Extract evaluation metric
    eval_match = re.search(r'## Evaluation\s*\n(.*?)(?=##|# |\Z)', text, re.DOTALL)
    if eval_match:
        sections['evaluation'] = eval_match.group(1).strip()
    
    # Extract submission format
    sub_match = re.search(r'## Submission File\s*\n(.*?)(?=##|# |\Z)', text, re.DOTALL)
    if sub_match:
        sections['submission'] = sub_match.group(1).strip()
    
    # Extract dataset description
    data_match = re.search(r'(?:# Dataset Description|## Dataset Description)\s*\n(.*?)(?=##|# |\Z)', text, re.DOTALL)
    if data_match:
        sections['dataset'] = data_match.group(1).strip()
    
    return sections


def simplify_text(text: str, max_chars: int = 200) -> str:
    """Simplify text to essential information."""
    if not text:
        return ""
    
    # Remove markdown links
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Remove images
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove excessive whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Truncate if too long
    if len(text) > max_chars:
        # Try to break at sentence end
        truncated = text[:max_chars]
        last_period = truncated.rfind('. ')
        if last_period > max_chars * 0.6:
            text = truncated[:last_period + 1]
        else:
            text = truncated.rstrip() + "…"
    
    return text


def extract_metric_summary(text: str) -> str:
    """Extract clear metric description."""
    if not text:
        return "See full description for metric details."
    
    # Remove LaTeX formulas
    text = re.sub(r'\$+[^$]+\$+', '[formula]', text)
    
    # Get first paragraph which usually contains the main metric
    paragraphs = text.split('\n\n')
    if paragraphs:
        metric = simplify_text(paragraphs[0], 300)
        return metric
    
    return simplify_text(text, 300)


def extract_submission_format(text: str) -> tuple:
    """Extract submission format details."""
    if not text:
        return "CSV with header", "id,prediction"
    
    # Look for format examples
    code_blocks = re.findall(r'```\s*\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        example = code_blocks[0].strip()
        lines = example.split('\n')
        if lines:
            header = lines[0]
            return f"CSV with header `{header}`", example
    
    # Look for format description
    format_match = re.search(r'header[^:]*:\s*`?([^`\n]+)`?', text)
    if format_match:
        header = format_match.group(1).strip()
        return f"CSV with header `{header}`", ""
    
    return "CSV format", ""


def extract_dataset_summary(text: str) -> list:
    """Extract key dataset information."""
    if not text:
        return []
    
    key_points = []
    
    # Look for file descriptions
    file_patterns = [
        (r'\*\*([^*]+\.csv)\*\s*[-–—]\s*([^\n]+)', 'csv'),
        (r'\*\*([^*]+\.json)\*\*?\s*[-–—]\s*([^\n]+)', 'json'),
        (r'\*\*([^*]+\.zip)\*\*?\s*[-–—]\s*([^\n]+)', 'zip'),
        (r'\*\*([^*]+/)\*\*?\s*[-–—]\s*([^\n]+)', 'dir'),
    ]
    
    for pattern, ftype in file_patterns:
        matches = re.findall(pattern, text)
        for match in matches[:3]:  # Limit to first 3 matches
            if isinstance(match, tuple):
                name, desc = match
            else:
                name = match
                desc = ""
            name = name.strip()
            desc = simplify_text(desc, 100) if desc else ""
            if name and len(name) < 50:
                key_points.append(f"**{name}**" + (f": {desc}" if desc else ""))
    
    # If no structured files found, extract general info
    if not key_points:
        # Look for train/test mentions
        train_match = re.search(r'(?:train|training)[^\n.]{0,100}(?:\.|\n)', text, re.IGNORECASE)
        test_match = re.search(r'(?:test|testing)[^\n.]{0,100}(?:\.|\n)', text, re.IGNORECASE)
        
        if train_match:
            key_points.append(f"Train: {simplify_text(train_match.group(0), 80)}")
        if test_match:
            key_points.append(f"Test: {simplify_text(test_match.group(0), 80)}")
    
    return key_points[:4]  # Limit to 4 points


def generate_lite_description(task_name: str, original_desc: str) -> str:
    """Generate improved lite description for a task."""
    
    sections = extract_sections(original_desc)
    
    # Task description - simplified
    task_desc = simplify_text(sections.get('description', ''), 250)
    
    # Task objective - extract prediction target
    objective = "Predict target variable per sample."
    if 'description' in sections:
        # Look for "predict" statements
        predict_match = re.search(r'(?:predict|forecast|identify|classify|detect)[^.]{10,150}', sections['description'], re.IGNORECASE)
        if predict_match:
            objective = simplify_text(predict_match.group(0), 200)
    
    # Metric
    metric = extract_metric_summary(sections.get('evaluation', ''))
    
    # Submission
    sub_header, sub_example = extract_submission_format(sections.get('submission', ''))
    
    # Dataset
    dataset_points = extract_dataset_summary(sections.get('dataset', ''))
    
    # Format title
    title = task_name.replace('-', ' ').title()
    
    # Build output
    lines = [
        f"# {title} — Lite Task Description",
        "",
        "## Task description",
        task_desc if task_desc else f"Complete the task as specified in {task_name} competition.",
        "",
        "## Task objective",
        objective,
        "",
        "## Target metric (evaluation)",
        metric,
        "",
        "## Brief background",
        simplify_text(sections.get('description', '')[:150], 150) if sections.get('description') else "See full description for background.",
        "",
        "## Submission",
        sub_header + ":",
    ]
    
    if sub_example:
        lines.extend([
            "```",
            sub_example.split('\n')[0] if sub_example else "id,prediction",
            "```",
        ])
    
    lines.append("")
    lines.append("## Dataset and construction")
    
    if dataset_points:
        for point in dataset_points:
            lines.append(f"- {point}")
    else:
        lines.append("- See full description for dataset details.")
    
    return '\n'.join(lines)


def main():
    """Process all tasks."""
    if not TASKS_DIR.exists():
        print(f"Tasks directory not found: {TASKS_DIR}")
        return
    
    processed = 0
    errors = []
    
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        
        task_name = task_dir.name
        desc_file = task_dir / "description_lite.md"
        original_file = MLEBENCH_DATA / task_name / "prepared" / "public" / "description.md"
        
        try:
            if original_file.exists():
                original_text = original_file.read_text(encoding='utf-8')
                lite_text = generate_lite_description(task_name, original_text)
                
                # Ensure it's not too long
                line_count = len(lite_text.split('\n'))
                if line_count > 30:
                    print(f"Warning: {task_name} has {line_count} lines, needs trimming")
                
                desc_file.write_text(lite_text, encoding='utf-8')
                processed += 1
            else:
                errors.append(f"Original description not found: {task_name}")
        except Exception as e:
            errors.append(f"Error processing {task_name}: {e}")
    
    print(f"Processed {processed} tasks")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
