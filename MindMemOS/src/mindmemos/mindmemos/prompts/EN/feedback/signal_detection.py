SIGNAL_DETECTION_PROMPT = """Filter and classify implicit feedback signals in session conversation rounds.

Input contains compact rounds with only the original user query and the final assistant reply.

Return JSON only, shaped as:
{
  "signals": [
    {
      "round_index": 0,
      "category": "task_temporary" | "scenario_specific" | "long_term",
      "reason": "..."
    }
  ]
}

Detect every round where the user expresses actionable feedback, including negative feedback, correction,
dissatisfaction, requested revision, changed preference, future behavior instruction, or durable working rule.
Do not omit feedback simply because the assistant successfully complied in the same round.
Do not include purely positive feedback with no requested change, preference, correction, or future instruction.
Use the zero-based round_index from the input.

One round may contain multiple independent feedback signals. Return one signal object for each independent signal,
even when multiple signal objects share the same round_index. Do not merge multiple feedback points into one
conservative category.

For every selected signal, first infer the concrete reason behind that specific feedback point from the surrounding
conversation:
- What did the user object to or correct?
- Why did the user want the change?
- Is that reason caused by a property of only the current task/artifact, by a reusable class of scenarios with the same feature, or by an unconditional preference/fact/rule?

Classify each signal based on that inferred reason:
- task_temporary: The reason depends on the current task's concrete state, current artifact, current implementation, current answer, current unavailable dependency, or a one-off execution problem. The feedback should not become retained memory by itself. Examples: "this command failed" because this run failed; "change this line" because the current patch has a bug; "explain this loss function more" because the current paper/concept was unclear; "use fake data here for now" because the current API/dependency is not implemented yet.
- scenario_specific: The reason depends on a reusable scenario feature shared by a class of future tasks. It should become durable only with an explicit scenario precondition. Examples: "for papers whose method implementation is simple, keep the method summary short"; "when reviewing PRs, be stricter about tests"; "for this project's deploy scripts, always use uv"; "for internal database access that is known to be unstable, add bounded retry and timeout".
- long_term: The reason has no task/scenario precondition and is a stable user preference, objective rule, fixed fact, correction to user knowledge, or generally applicable behavior. Examples: "I use uv, not conda"; "my team uses Beijing time"; "backend API request parameters should never use `any` placeholders"; "write future code comments in Chinese"; "I prefer detailed answers".

The category must follow the reason, not just the surface form of the user's wording. A correction can be long_term if the reason is a general rule; a request can be scenario_specific if the reason is a reusable scenario feature; a strong complaint can still be task_temporary if it only concerns the current implementation.
In the reason field, explicitly state the inferred reason and why that reason maps to the chosen category.
The reason may cite or paraphrase the relevant signal briefly for explainability.
The reason must describe only this signal. If the same round contains another independent feedback point with a
different category or reason, return it as a separate signal with the same round_index.

Important boundary cases:
- If the user asks a follow-up for more detail because the previous answer was unclear, incomplete, too shallow, or missed a specific concept, include it as a negative implicit feedback signal. Classify it as task_temporary unless the user states a reusable preference or scenario. Example: "Can you explain this loss function in more detail? The previous answer did not make it clear." is task_temporary for the current paper/concept, not a long-term preference for all loss functions.
- If the user uses a question to reveal that the previous answer does not fit a current one-off constraint, include it as a task_temporary signal. Example: "Can this be finished in two hours? This is a short flight and I do not have much time." is a current trip/time constraint, not a long-term preference for short works.
- If the user edits or rejects the assistant's output because of a reusable property of an artifact/content item, classify it as scenario_specific, not task_temporary. The named artifact is evidence, but the reusable scenario should come from the property/reason. Example: "this paper's method is simple, no need to expand" means the scenario is papers with simple method implementation, not only that named paper.
- If the user edits or rejects the assistant's output only because of a named one-off artifact with no reusable property, classify it as task_temporary. Example: "for this document, keep the background short" is current-task-only unless the user explains a reusable condition such as "because the background is already well known".
- If the user only corrects the current execution step without a reusable content/task condition, classify it as task_temporary. Examples: "that command failed", "you edited the wrong file", "undo the previous change".
- If the user states a general preference without limiting it to a specific artifact/task/scenario, classify it as long_term. Example: "I always prefer concise method sections".
- If a current correction sentence also states an unconditional user preference, fact, or known-item fact, return separate signals for the current correction and the durable statement. Example: "I have heard of this book, and I dislike horror or bloody books; recommend something else." contains a task_temporary signal for changing the current recommendation, a long_term known-item fact, and a long_term genre preference.
- If the user says future outputs should follow a style without limiting it to a task/scenario, classify it as long_term. Example: "use Chinese comments in code from now on" is a long-term preference, not just the current code block.
- If the user corrects a general coding rule without limiting it to the current implementation, classify it as long_term. Example: "backend interface parameter types must not use `any`; define strict types" is a general coding rule.
- If the user asks for a temporary workaround because a current dependency/API is not implemented, classify it as task_temporary. Example: "the internal database access API is not implemented yet, use fake data for now" is about the current task state and should not be retained as a future preference to use fake data.

If uncertain between task_temporary and scenario_specific, choose task_temporary unless the user clearly describes a reusable task type or scenario.
If uncertain between scenario_specific and long_term, choose scenario_specific when the feedback depends on a task/project/workflow condition; choose long_term when it is generally true.
Do not output a feedback field.
"""
