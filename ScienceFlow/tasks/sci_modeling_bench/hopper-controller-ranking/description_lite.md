# Hopper Controller Candidate Ranking

## Objective

Select and order 32 frozen policy vectors predicted to have high expected
Hopper-v5 episodic return. Candidates must come from the explicit finite pool;
this task does not evaluate newly trained or generated policies.

## Agent-visible input

Read `dataset/dataset_manifest.json` and its policy-layout context before
reshaping vectors. `dataset/views/observations.parquet` contains 1,920
lower-return policies with 500 raw returns, episode lengths, and termination
flags. `dataset/views/candidates.parquet` contains 1,280 unlabeled policy
vectors. The manifest specifies layer widths, flattened parameter order,
activation, action distribution and clipping, environment, rollout count, and
target aggregation.

## Submission

Write at least 32 unique candidate vectors to `artifacts/submission.json`:

```json
{"candidates": [{"policy_weights": [0.01, -0.02, 0.03]}]}
```

The short vector only illustrates the JSON field shape. Each submitted vector
must be the complete exact vector from the candidate
view. Only the first 32 are scored; exactly 32 is recommended.

This is a strictly offline prediction task. Use the disclosed frozen data to
train supervised surrogate or ranking models. Do not execute candidate
policies, instantiate Hopper-v5 or MuJoCo, collect fresh rollouts, or train or
fine-tune PPO/RL policies. The environment and policy metadata are provided
only to interpret the frozen measurements and parameter layout.

## Evaluation

The primary metric is `global_ndcg`. Feedback includes
`best_k_mean_regret`, `normalized_enrichment`, and remaining queries, but no
per-policy returns. At most 5 distinct lists receive official evaluation.
