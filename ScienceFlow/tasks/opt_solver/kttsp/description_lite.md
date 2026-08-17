# KTTSP opt_solver task

Keplerian Tomato Traveling Salesperson Problem artifact evaluator package.

Candidate artifacts use `artifacts/best_solution.json` with an Optimize-style `decisionVector` or local `x` chromosome. The evaluator validates route permutation, time bounds, Lambert transfer feasibility, and delta-v exception limits.

The metric is `mission_duration_days`; lower is better.
