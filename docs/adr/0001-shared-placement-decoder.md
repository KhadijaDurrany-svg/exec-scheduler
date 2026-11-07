# ADR 0001 — One shared placement decoder for every scheduler

- Status: accepted
- Date: 2026-11-07

## Context

We compare a proposed planner against several baselines (FIFO, earliest-deadline,
priority-greedy, random). If each scheduler had its own placement logic, any
measured difference could come from implementation quirks rather than the
scheduling policy, and the benchmark would not be trustworthy.

## Decision

All schedulers share a single deterministic decoder, `schedulers.decode(instance,
order, dropped)`. It greedily places tasks into the earliest feasible slot given
a priority order and a dropped set, respecting dependencies, meetings and working
hours via the `Timeline` engine. Baselines are nothing more than different
*orderings* fed to this decoder, with an empty dropped set. The `slotfit` planner
is the same decoder driven by a local search over `(order, dropped)`.

## Consequences

- Differences between methods are attributable to policy, not plumbing.
- The decoder is the single most important thing to test; `conftest.py` ships an
  `assert_valid_schedule` invariant checker used by both unit and property tests.
- The decoder is greedy/earliest-fit, so it cannot express "leave a slot empty now
  to use it later"; that expressiveness is pushed into the search layer (drops and
  re-ordering) rather than the decoder.
