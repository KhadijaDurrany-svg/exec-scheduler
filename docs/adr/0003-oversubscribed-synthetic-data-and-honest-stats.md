# ADR 0003 — Over-subscribed synthetic data and honest statistics

- Status: accepted
- Date: 2026-11-07

## Context

Benchmarks are only meaningful if (a) the instances are hard enough to separate
methods, and (b) the reported numbers quantify uncertainty rather than a single
lucky run. A generator that produces easily-feasible instances would let every
method score ~1.0 and hide all signal — the classic "suspiciously perfect"
result.

## Decision

- The generator (`generator.py`) is deliberately **over-subscribed**: total
  task-minutes are `load × free-capacity` with `load = 1.45 > 1`, so no scheduler
  can finish everything. A test (`test_generated_instance_is_oversubscribed`)
  asserts demand exceeds capacity.
- Data is seeded (`numpy.random.default_rng`) and instances are generated from a
  derived seed family, so runs are reproducible and a dataset checksum is recorded
  in the manifest.
- Reporting uses a **percentile bootstrap 95% CI** (10,000 resamples) for each
  method's mean, and a **two-sided paired permutation test** for slotfit vs. the
  strongest baseline. The permutation p-value is floored at `1/(n_perm+1)` so a
  finite sample never reports `p = 0`.

## Consequences

- Published numbers (README) are a real run: slotfit ≈ 0.715 utility vs. the best
  baseline ≈ 0.585, delta ≈ +0.130, permutation p at the 10k-sample floor.
- No method is perfect, which is the intended, honest outcome.
- The absolute numbers are a property of this synthetic distribution and will move
  on differently-shaped workloads; this is disclosed, not hidden.
