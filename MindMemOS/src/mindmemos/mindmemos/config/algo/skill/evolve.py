from dataclasses import dataclass, field


@dataclass
class SkillEvolutionConfig:
    """Tuning for the skill self-evolution pipeline (``pipelines/skill/evolution``).

    The pipeline summarizes each injected ``/v1/memory/add`` trajectory, and once
    enough unconsumed summaries accumulate it aggregates them into a skill patch
    and mints a new (draft/cloud) version. When more than ``max_aggregate``
    summaries are pending, several versions are minted serially in add-time order,
    each consuming ``min_aggregate``..``max_aggregate`` summaries.
    """

    min_aggregate: int = field(default=8)
    """Evolution threshold: minimum pending summaries required to mint a version."""

    max_aggregate: int = field(default=8)
    """Maximum trajectory summaries aggregated into one patch / version."""

    summary_concurrency: int = field(default=8)
    """Maximum trajectories summarized in parallel."""

    rewrite_skill: bool = field(default=False)
    """Whether to run an extra LLM pass that reformats the patched SKILL.md."""

    use_trajectory_score: bool = field(default=True)
    """Whether to feed per-trajectory evaluation scores into patch proposal.

    When enabled and a batch carries any score, the proposer uses the scored
    (supervised) prompt to reinforce high-score behaviors and avoid low-score
    ones; batches without scores, or this flag off, fall back to the
    unsupervised prompt.
    """

    evolved_status: str = field(default="draft")
    """Lifecycle status assigned to a freshly evolved version (design §3)."""

    transcript_max_chars: int = field(default=1500)
    """Per-message truncation budget when rendering a trajectory transcript."""

    max_trace_scan: int = field(default=2000)
    """Safety cap on how many add records are scanned per evolve call."""

    summary_task: str = field(default="skill_trajectory_summary")
    """LLM task tag for the trajectory-summary call."""

    patch_task: str = field(default="skill_patch_propose")
    """LLM task tag for the patch-proposal call."""

    apply_task: str = field(default="skill_patch_apply")
    """LLM task tag for the patch-apply call."""

    rewrite_task: str = field(default="skill_rewrite")
    """LLM task tag for the optional reformat call."""
