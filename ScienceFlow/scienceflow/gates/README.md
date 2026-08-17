# Gate plugins

`GateService` is the authoritative candidate-admission entry point. Its logical
call order is:

```text
GateService -> gates/evaluator/EvaluatorManager/Backend -> MetricEvent -> GatePolicy
            -> GateDecision.accepted -> Stage
```

An evaluator result is evidence, not a Stage. In primary mode, exactly one
`EvaluationOutcome` is required and only `decision.accepted=True` may append the
hidden ledger and capture a Stage snapshot. Missing, duplicate, unknown-policy,
invalid-config, or exception outcomes fail closed for Stage admission.

Tool callbacks are only triggers. Before running Gate, primary mode compares the
candidate artifact SHA with committed Stage snapshots; an unchanged artifact is
logged as `duplicate_candidate_pre_gate_skipped` and does not produce another
evaluation or Gate decision. After an accept, malformed agent-authored Stage
metadata is retried and then replaced with a deterministic ledger-valid fallback
so bookkeeping text cannot discard accepted evaluator evidence. Snapshot or I/O
failures remain explicit transaction failures and are rolled back.

Global finals keep their existing validation contract because some task types
can validate a final artifact but cannot recompute its training metric. They
record a Gate trace for audit, but they are not Stage creation events.

## Lightweight configuration

The default behavior needs no configuration:

```yaml
gate:
  policy: default
```

The block can be omitted. To override an allowed policy parameter:

```yaml
gate:
  policy: default
  params:
    minimum_metric_validity: high
```

`high` is the historical default, not a newly introduced threshold. The
default policy otherwise preserves the pre-plugin predicate: validation and
candidate readiness must pass, the primary metric must exist and be finite,
metric direction is required only when
`evaluator.metric.selection_requires_direction` says so, and metric validity
must meet the configured evidence level.

Task policy selection is authoritative. Runtime configuration may provide a
fallback policy when the task does not name one and may override `params`, but
cannot disable core invariants or change the trigger.

Evaluator defaults are task-neutral (`task_profile: auto`, `backend: auto`). A
registered task package supplies the effective profile, backend, candidate
artifact, and metric contract. Provider families may keep common manifest
fields in `_shared/task_defaults.yaml`; each task's `task.yaml` is recursively
merged on top, so task-specific values remain authoritative.

## Plugin contract

```python
class GatePolicy(Protocol):
    name: str
    version: str

    def decide(self, ctx, event, *, trigger, params) -> GateDecision: ...
```

Register trusted policies explicitly with `GateManager.register()`. Workspace
scanning and arbitrary `module:object` loading are intentionally unsupported.
The policy name, version, parameters, trigger, and decision are recorded in the
metric event trace.

Stage reporting keeps `gate_metric_validity` (the evaluator evidence used by
the Gate decision) separate from `metric_validity` (the later Stage-result
audit used by selection). A post-commit audit may lower the latter without
rewriting the historical Gate decision.

Built-ins:

- `default`: behavior-compatible general candidate gate.
- `optimization_feasibility`: opt-in default checks plus a constraint-violation
  bound.

The filesystem mirrors that ownership:

```text
scienceflow/gates/
├── service.py              # authoritative Gate transaction
├── policy.py               # plugin registry and policy implementations
├── invariants.py           # non-configurable safety boundary
├── feedback.py             # agent-facing rejection feedback
└── evaluator/              # fact-producing plugins owned by Gate
    ├── manager.py
    ├── models.py
    ├── providers/          # provider-specific validation integrations
    └── backends/
```

Consumers that need an admission decision import `GateService` and policies
from `scienceflow.gates`. Evaluator plugin authors import the fact model and
manager from `scienceflow.gates.evaluator`; invoking an evaluator directly does
not authorize Stage creation.
