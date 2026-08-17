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

"""Batch compress ML description_lite.md files to <= 30 lines."""

from pathlib import Path
import re

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks" / "ml" / "mlebench"
MAX_LINES = 30


def compress_file(filepath: Path) -> str:
    """Compress a single description_lite.md file."""
    text = filepath.read_text(encoding='utf-8')
    lines = text.split('\n')
    
    # If already <= 30 lines, return as-is
    if len(lines) <= MAX_LINES:
        return text
    
    # Extract sections
    sections = {}
    current_section = None
    current_content = []
    
    for line in lines:
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif current_section:
            current_content.append(line)
    
    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()
    
    # Compress each section
    compressed = {}
    for name, content in sections.items():
        # Remove extra whitespace
        content = re.sub(r'\n+', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Limit length per section
        if len(content) > 300:
            # Try to break at sentence
            truncated = content[:300]
            last_period = truncated.rfind('. ')
            if last_period > 150:
                content = truncated[:last_period + 1]
            else:
                content = truncated.rstrip() + '…'
        
        compressed[name] = content
    
    # Rebuild with strict line budget
    title_line = lines[0] if lines else "# Task Description"
    result = [title_line, ""]
    
    section_order = [
        "Task description",
        "Task objective", 
        "Target metric (evaluation)",
        "Brief background",
        "Submission",
        "Dataset and construction"
    ]
    
    for section_name in section_order:
        # Find matching section
        matched_key = None
        for key in compressed.keys():
            if key.lower() == section_name.lower() or key.lower().startswith(section_name.lower()):
                matched_key = key
                break
        
        if matched_key and compressed[matched_key]:
            result.append(f"## {section_name}")
            # Wrap content to fit in remaining lines
            content = compressed[matched_key]
            # Simple wrap at word boundary
            while content and len(result) < MAX_LINES - 1:
                if len(content) <= 100:
                    result.append(content)
                    break
                # Find break point
                break_at = content.rfind(' ', 0, 100)
                if break_at < 50:
                    break_at = 100
                result.append(content[:break_at])
                content = content[break_at:].strip()
            result.append("")
    
    # Ensure exactly MAX_LINES or fewer
    while len(result) > MAX_LINES:
        # Remove from longest content section
        result.pop(-2)  # Remove last content line before final empty line
    
    return '\n'.join(result)


def main():
    processed = 0
    skipped = 0
    
    for task_dir in sorted(TASKS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        
        desc_file = task_dir / "description_lite.md"
        if not desc_file.exists():
            continue
        
        # Skip AI4Code (already manually optimized)
        if task_dir.name == "AI4Code":
            skipped += 1
            continue
        
        original = desc_file.read_text(encoding='utf-8')
        original_lines = len(original.split('\n'))
        
        if original_lines > MAX_LINES:
            compressed = compress_file(desc_file)
            desc_file.write_text(compressed, encoding='utf-8')
            new_lines = len(compressed.split('\n'))
            print(f"{task_dir.name}: {original_lines} → {new_lines} lines")
            processed += 1
        else:
            skipped += 1
    
    print(f"\nProcessed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
