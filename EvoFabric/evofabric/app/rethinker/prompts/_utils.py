# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.



def repeat_prompt(prompt: str, repeat_time: int = 3) -> str:
    if repeat_time < 1:
        raise ValueError('repeat time must be greater than 0')
    repeat_lines = [
        "Let me repeat that:",
        "Let me repeat that one more time:"
    ]
    prompts = [prompt]
    for i in range(repeat_time - 1):
        line_index = min(i, len(repeat_lines) - 1)
        prompts.append(repeat_lines[line_index])
        prompts.append(prompt)
    return "\n\n".join(prompts)
