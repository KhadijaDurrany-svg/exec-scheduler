# ADR 0002 — Heuristic local search instead of an exact solver

- Status: accepted
- Date: 2026-11-07

## Context

Maximising weighted on-time priority on a fragmented calendar with precedence
constraints is NP-hard (it generalises `1 | prec | Σ wⱼUⱼ`). An exact approach
(ILP or CP) would need a heavyweight solver dependency and would not scale to the
benchmark's instance count within a quick, offline run. The project's constraints
also explicitly rule out heavy solver dependencies.

## Decision

Implement a greedy constructive seed (earliest-deadline, value-density tie-break)
followed by seeded hill-climbing local search over `(priority order, dropped
set)`. Moves are: swap two tasks in the order, drop a currently-late task, or
un-drop one. Non-worsening moves are accepted to cross plateaus. Everything is
pure Python + numpy.

## Consequences

- Fast, dependency-light, fully reproducible given a seed.
- **No optimality guarantee and no bound on the gap to optimum.** This is stated
  plainly in the README "Known gaps". Quality depends on `--iters` and the seed.
- Because there is no exact oracle, we cannot report "percent of optimal". A small
  ILP/CP reference on tiny instances is noted as future work.
- The honest objective (`weighted_on_time`) is also what the search optimises, so
  there is no train/report metric mismatch.
