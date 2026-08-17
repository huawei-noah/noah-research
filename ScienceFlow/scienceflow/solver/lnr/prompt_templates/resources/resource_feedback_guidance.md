# Resource Feedback Guidance

advisory_stop_boundary_note: For resource low-progress or efficiency proposals, safe_to_stop means the current command can be interrupted; it does not mean abandon the route. Keep the best artifact, then change the method, search space, schedule, validation target, or stopping condition before continuing the same loop unchanged.

budget_deliverable_value_note: Do not equate low-level liveness with research value. Batch, epoch, stdout, CPU activity, or log growth is only useful if it can plausibly become a valid task deliverable within the remaining useful budget; for submission-style tasks, prefer a shorter path to root submission.csv and a recorded validation score before long optimization runs.

resource_execution_stop_reason: main-agent advisory and high-confidence resource facts agree this execution should stop; keep the best artifact and improve the method, search space, schedule, validation target, or stopping condition before continuing

execution_optimization_focus: change_method_search_space_schedule_validation_target_or_stopping_condition

research_cadence_note: A research-cadence violation is a fact about time to the next comparable metric, not a prescribed scientific route. For an unproven route, the main agent should respond with a bounded pilot that preserves the validation contract. Simple work whose first comparable metric fits the budget may run at full scale.
