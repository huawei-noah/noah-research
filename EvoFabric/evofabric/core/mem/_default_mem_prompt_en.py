# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Refer to mem0 system prompt and adapt for general use.


FACT_RETRIEVAL_PROMPT_EN = f"""You are an information extraction expert who can determine whether the input information contains content that needs to be extracted and recorded based on user requirements.

Information types to extract:
[Memory Information]

Here are a few examples:

## Example 1:
Memory information to extract: Person's identity
Input: It's raining today.
Output: []
Input: Peter was a math teacher last year, but switched to being a PE teacher this year.
Output: ["Peter was a math teacher last year", "Peter is a PE teacher this year"]
Input: I like to eat vanilla ice cream
Output: []

## Example 2:
Memory information to extract: Person's preferences
Input: It's raining today.
Output: []
Input: Peter was a math teacher last year, but switched to being a PE teacher this year.
Output: []
Input: I like to eat vanilla ice cream
Output: ["The user likes to eat vanilla ice cream"]

## Example 3:
Memory information to extract: Person's preferences, Person's identity
Input: It's raining today.
Output: []
Input: Peter was a math teacher last year, but switched to being a PE teacher this year.
Output: ["Peter was a math teacher last year", "Peter is a PE teacher this year"]
Input: I like to eat vanilla ice cream
Output: ["The user likes to eat vanilla ice cream"]

Please return the corresponding information list in the form of a string list.
"""

DEFAULT_UPDATE_MEMORY_PROMPT_EN = """You are an intelligent memory manager responsible for controlling the system’s memory.
You may perform four operations:
(1) ADD information to memory,
(2) UPDATE existing memory,
(3) DELETE information from memory,
(4) NONE make NO change.
Memory will evolve according to these four operations.
Compare each newly extracted fact with the current memory.
For every new fact decide whether to:
ADD: insert it as a new element at the end of the list (assign the next sequential ID).
UPDATE: refresh the content of an existing element while keeping its original position and ID.
DELETE: remove an existing element.
NONE: leave the existing element untouched (because the fact is already present or irrelevant).
Operation-selection rules
ADD: if the extracted information contains something not yet stored, append it with a new ID.
UPDATE: if the extracted fact differs from the stored information but can be merged, overwrite the old content while preserving ID and order. Do not ADD the same content again after an UPDATE.
DELETE: if the extracted fact contradicts stored information.
– When the old memory is correct, do nothing.
– When the new fact is correct and cannot be merged via UPDATE, DELETE the old entry and then ADD the new one.
NONE: when old and new facts do not conflict, keep the old entry unchanged and mark the operation as NONE.
Example 1
Old memory:
[
    { "id": 0, "text": "The user works at a tech company" }
]
New facts: ["Has a golden retriever"]
Output:
[
    { "id": 0, "text": "The user works at a tech company", "event": "NONE" },
    { "id": 1, "text": "Has a golden retriever", "event": "ADD" }
]
Example 2
Old memory:
[
    { "id": 0, "text": "The user often goes to the library" },
    { "id": 1, "text": "Speaks French" },
    { "id": 2, "text": "Likes hiking" }
]
New facts: ["Often goes to cafés", "Also speaks Spanish"]
Output:
[
    { "id": 0, "text": "Often goes to the library and cafés", "event": "UPDATE", "old_memory": "The user often goes to the library" },
    { "id": 1, "text": "Speaks French and Spanish", "event": "UPDATE", "old_memory": "Speaks French" },
    { "id": 2, "text": "Likes hiking", "event": "NONE" }
]
Example 3
Old memory:
[
    { "id": 0, "text": "The user lives in Shanghai" },
    { "id": 1, "text": "Is allergic to seafood" }
]
New facts: ["Is not allergic to seafood"]
Output:
[
    { "id": 0, "text": "The user lives in Shanghai", "event": "NONE" },
    { "id": 1, "text": "Is allergic to seafood", "event": "DELETE" }
]
"""

FEAT_DEFINE_PROMPT_EN = """
- User's lifestyle habits
- User's emotions
"""

TASK_SUMMARY_PROMPT_EN = """
You are an expert at distilling experience into strategy.
Given a list of past action-response pairs along with their correctness flag, score, and critic comment, produce a concise, generic set of prompt-level rules that prevent previous failures and promote high-scoring behavior.
The generated rules should be related to action-response pairs to guide the agent policy.

Example 1
Historical log:
[
"content: user: My phone has no internet, what should I do? assistant: I'll restart your phone now. correctness: False, score: 0, critic: Restarting is a dangerous operation and never allowed.",
"content: user: My phone has no internet, what should I do? assistant: Let me check your Wi-Fi settings. correctness: True, score: 1.0, critic: Reasonable first step."
]
Output:
- Never perform sensitive actions such as restarting the user's device.
- When network issues are reported, start by inspecting wireless settings.

Historical log:
"""
