IMPLICIT_ACTION_PLANNING_PROMPT = """Plan memory feedback actions for all implicit feedback signals in one conversation round.

Inputs:
- signals: one or more feedback signals detected in the same conversation round. Each signal includes:
  - task_temporary: current-task-only feedback.
  - scenario_specific: durable feedback only within the current task type/scenario.
  - long_term: generally durable feedback.
- round: the compact single-round context containing the original user query and final assistant reply. Use this
  complete round as the source of truth for the feedback content and correction, but use signals[].reason to identify
  which specific feedback points should drive memory actions.
- memories: deduplicated candidate memories from add records, search records, and supplemental recall.

Use only these action types:
- add: a signal expresses a new durable user preference, fact, correction, or long-term instruction not covered by the memory pool.
- update: an existing memory conflicts with a signal, or needs more conditions/details instead of a separate memory.
- delete: an existing memory is stale, wrong, or should not have been saved, and there is no suitable replacement content.
- noop: the memory already matches the signal, or the issue is retrieval/usage rather than memory content.

Rules:
- Follow each signal.category when deciding actions.
- If no memory action is needed, return {"actions":[]}.
- Every action must be grounded in one or more input signals. In action.reason, explicitly cite the signal category and
  reason that justify the action.
- Only mutate memories that are directly related to the grounded signal(s). Do not mutate a memory merely because it
  appears in the same round, same session, same task, or candidate memory pool.
- If a memory is related to other content in the round but not directly related to any input signal, leave it unchanged.
- Plan actions from all signals in this round, and ensure the actions are mutually consistent: they must not conflict
  with, duplicate, or cancel each other out.
- Preserve signal boundaries when multiple signals are provided for the same round:
  - task_temporary signals may only delete or correct memories that store that temporary task-only fact/correction.
  - scenario_specific signals may only add or update memories for that scenario-conditioned rule.
  - long_term signals may only add or update memories for that general preference, fact, or rule.
  - A task_temporary signal must not suppress, delete, or override a long_term or scenario_specific signal from the same round.
- Write after_content in the same language as the user's signal/round. If the signal is Chinese, after_content must be Chinese. This is required for later lexical retrieval.
- Before adding a memory, inspect the candidate memories for semantically similar content, including memories with a different scope, condition, or temporal wording. If a similar memory already exists, prefer update over add so the stored memory has the correct scope and avoids duplicates.
- If an existing narrow-scope memory can be correctly updated into a broader durable statement that covers the same content without conflict, update that memory to the broader statement and do not preserve the original narrow-scope wording as a separate memory.
- Only use scope explicitly stated by the user's signal as memory scope. Do not infer a narrower scope from the current task, artifact, project, code file, or assistant implementation unless the user's signal names that scope as the reason or condition.
- When long-term signals are expressed without an explicit scope, keep after_content general. Do not narrow it to the current round context during update or scope-union.
- Durable memory content must not include unbounded temporal wording unless the user gave a concrete time range or version range. Avoid phrases such as "from now on", "以后", "之后", "目前", "当前", "now", "currently", "recently", "today", or "this time" in long-term memory unless they are part of an explicit bounded condition. Write timeless durable memories instead. Example: write "代码注释使用中文。" rather than "从现在开始，代码注释都用中文。"
- Before returning an action, check that after_content satisfies all rules above, including language, explicit scope, temporal wording, scope union/conflict handling, scenario preconditions, and durability. If after_content violates any rule, revise it until it satisfies the rules.
- Return JSON only.

Category-specific rules:
- task_temporary:
  - Do not add or update memory to retain this signal.
  - If any existing candidate memory stores this temporary task-only fact/correction as durable memory, delete every matching memory, not just the best one.
  - For temporary workaround signals, delete all candidate memories that preserve the same temporary workaround, including direct user-request memories, current-state memories, implementation-result memories, derived detail memories, semantically equivalent memories, and related same-round memories that would make the temporary workaround retrievable later.
  - Do not rely on the executor to cascade deletes. Every memory that should be removed must appear as its own delete action with its own target_memory_id.
  - If no existing memory should be removed for this signal, produce no action for this signal. Continue planning actions for other signals in the same round.
- scenario_specific:
  - Keep the signal as memory only with an explicit scenario/task-type precondition in after_content.
  - Derive the scenario condition from the user's stated reason or content property, not merely from the artifact name.
  - The reusable scenario condition should describe what must be true in a future task before applying the memory.
  - If no related memory exists, add a new memory whose after_content starts from or clearly includes the scenario condition, for example "When working on <scenario>, ...".
  - If a related memory exists, update it so after_content includes the scenario condition and the corrected behavior/preference/fact.
  - If an existing candidate memory records the same artifact/content item but has the wrong behavior or over-expanded content, prefer update over delete so the corrected scenario-specific instruction is retained.
  - For content-specific edits, keep artifact names only as optional evidence if needed; do not use the artifact identity as the primary scenario when the signal reason gives a reusable condition. If the artifact name is included, it must not make the memory appear limited only to that artifact.
  - Example: if the user says "this paper's method is too simple, no need to expand", write memory like "When summarizing a paper whose method implementation is simple, keep the method section concise; two sentences are enough." If the user says "这篇方法太简单，没必要展开", write "当总结方法实现简单的论文时，方法部分保持简短，两句概括即可；不要应用到方法复杂的论文。"
  - Do not write "When summarizing Paper A's method section..." and do not generalize to "Always keep method sections concise."
  - Do not generalize scenario-specific signals into unconditional long-term memory.
- long_term:
  - Treat as normal durable memory feedback for the grounded signal.
  - Do not include temporal wording in after_content unless the user gave an explicit bounded time/version range. If the user says "from now on", "以后", or "之后" only to express a stable preference, omit that temporal phrase from the memory.
  - If an existing memory expresses the same preference/fact/rule with a narrower, broader, or time-bound scope, update that memory instead of adding a duplicate. Preserve compatible scopes by taking their union; replace conflicting scopes with the new signal scope.
  - Add a new memory only when no semantically similar candidate memory exists.
  - Add, update, delete, or noop according to the candidate memories and the signal content.

Action object formats:
- AddAction:
  {
    "action": "add",
    "after_content": "new durable memory content",
    "reason": "short reason for this action",
    "status": "ok"
  }
- UpdateAction:
  {
    "action": "update",
    "target_memory_id": "one of memories[].id",
    "before_content": "old memory content copied from the target memory",
    "after_content": "new corrected or supplemented memory content",
    "reason": "short reason for this action",
    "status": "ok"
  }
- DeleteAction:
  {
    "action": "delete",
    "target_memory_id": "one of memories[].id",
    "before_content": "old memory content copied from the target memory",
    "reason": "short reason for this action",
    "status": "ok"
  }
- NoopAction:
  {
    "action": "noop",
    "target_memory_id": "one of memories[].id, or null if no specific memory applies",
    "before_content": "old memory content if a target memory applies, otherwise null",
    "reason": "short reason why no memory mutation is needed",
    "status": "ok"
  }

Return shape:
{
  "actions": [
    AddAction or UpdateAction or DeleteAction or NoopAction
  ]
}
"""

EXPLICIT_ACTION_PLANNING_PROMPT = """You plan memory updates from explicit user feedback.

Use only these action types:
- add: the current memories do not cover the user's durable feedback.
- update: an existing memory conflicts with the feedback, or should be supplemented instead of creating a separate memory.
- delete: an existing memory conflicts with the feedback, and the user did not provide replacement memory content.
- noop: existing memory is already consistent with the feedback.

Rules:
- Only use durable user preferences, facts, or corrections as memory content.
- Do not create actions for temporary task corrections.
- Before adding a memory, inspect recalled_memories for semantically similar content, including memories with a different scope, condition, or temporal wording. If a similar memory already exists, prefer update over add so the stored memory has the correct scope and avoids duplicates.
- When updating a similar memory with a different scope, compare the old scope and new scope. If the scopes are different but compatible, write after_content as the union of both scopes. If the scopes conflict, let the new feedback override the old conflicting scope.
- Only use scope explicitly stated by the user's feedback as memory scope. Do not infer a narrower scope from the current task, artifact, project, code file, or assistant implementation unless the user's feedback names that scope as the reason or condition.
- When long-term feedback is expressed without an explicit scope, keep after_content general. Do not narrow it to the current round context during update or scope-union.
- Durable memory content must not include unbounded temporal wording unless the user gave a concrete time range or version range. Avoid phrases such as "from now on", "以后", "之后", "目前", "当前", "now", "currently", "recently", "today", or "this time" unless they are part of an explicit bounded condition. Write timeless durable memories instead.
- Before returning an action, check that after_content satisfies all rules above, including language, explicit scope, temporal wording, scope union/conflict handling, and durability. If after_content violates any rule, revise it until it satisfies the rules.
- Return JSON only.

Action object formats:
- AddAction:
  {
    "action": "add",
    "after_content": "new durable memory content",
    "reason": "short reason for this action",
    "status": "ok"
  }
- UpdateAction:
  {
    "action": "update",
    "target_memory_id": "one of recalled_memories[].id",
    "before_content": "old memory content copied from the target memory",
    "after_content": "new corrected or supplemented memory content",
    "reason": "short reason for this action",
    "status": "ok"
  }
- DeleteAction:
  {
    "action": "delete",
    "target_memory_id": "one of recalled_memories[].id",
    "before_content": "old memory content copied from the target memory",
    "reason": "short reason for this action",
    "status": "ok"
  }
- NoopAction:
  {
    "action": "noop",
    "target_memory_id": "one of recalled_memories[].id, or null if no specific memory applies",
    "before_content": "old memory content if a target memory applies, otherwise null",
    "reason": "short reason why no memory mutation is needed",
    "status": "ok"
  }

Return shape:
{
  "actions": [
    AddAction or UpdateAction or DeleteAction or NoopAction
  ]
}

If no memory action is needed, return {"actions":[]}.
"""
