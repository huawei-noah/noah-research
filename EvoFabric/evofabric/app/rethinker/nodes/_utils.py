# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.


def parse_answer(context):
    """Get the last assistant response and extract Conclusion part if there is any."""
    final_answer = context

    if "<｜Assistant｜>" in context:
        final_answer = context.split("<｜Assistant｜>")[-1]
        if "</think>" in final_answer:
            final_answer = final_answer.split("</think>")[-1]

    if "# Conclusion" in final_answer:
        final_answer = final_answer.split("# Conclusion")[-1]
        final_answer = "# Conclusion" + final_answer

    return final_answer


def strip_response(response: str):
    """Get the last assistant response."""
    return response.split("<｜Assistant｜>")[-1]


def clip_str(string, thres=2000):
    """Clip a string to a certain threshold."""
    if len(string) > thres:
        return string[thres:]
    return string


def strip_think_and_exec(text: str) -> str:
    """Keep only the visible answer part by removing </think> ... and </execution_results> tails."""
    if text is None:
        return ""
    out = text
    if "<｜Assistant｜>" in out:
        out = out.split("<｜Assistant｜>")[-1]
    if "</think>" in out:
        out = out.split("</think>")[-1]
    if "</execution_results>" in out:
        out = out.split("</execution_results>")[-1]
    return out.strip()
