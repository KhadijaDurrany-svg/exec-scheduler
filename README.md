# slotfit — offline constraint-based day planner

`slotfit` turns a set of tasks (each with a duration, deadline, priority weight
and dependencies), a list of fixed meetings, and working-hours windows into a
feasible, non-preemptive schedule. It optimises one **transparent** objective:

> **weighted on-time priority** — the total priority weight of tasks that are
> scheduled *and* finish on or before their deadline.

Everything is **offline, deterministic and seeded**: no network, no API keys,
no external services. The only runtime dependencies are `numpy` and
`pydantic` (v2). The improvement search (greedy + local search) is implemented
from scratch — there is no heavyweight solver dependency.

This is a portfolio project. The benchmark numbers below are produced by a real
run of the code in this repository; none of them are hand-written.

---

## What it does

The scheduling problem is a calendar-packing problem under deadlines: pick
*which* tasks to do and *when*, so that as much weighted work as possible lands
before its deadline, without overlapping meetings, violating working hours, or
starting a task before its dependencies finish. Because the instances are
deliberately over-subscribed (more task-minutes than free capacity), the planner
must decide what to drop or delay — the interesting part.

The optimisation target is NP-hard (it generalises `1 | prec | Σ wⱼUⱼ` onto a
fragmented calendar), so `slotfit` uses a heuristic, not an exact solver:

1. **Greedy seed** — order tasks by earliest deadline, breaking ties by value
   density (weight ÷ duration), and place each one in the earliest feasible slot
   that respects its dependencies.
2. **Local search** — seeded hill-climbing over `(priority order, dropped set)`.
   Moves: swap two tasks in the order, *drop* a task that is currently late (to
   free its slot for something more valuable), or *un-drop* a previous sacrifice.
   Non-worsening moves are accepted so the search can cross plateaus.

All five schedulers share one deterministic placement engine; the baselines
differ only in how they order tasks and never drop, which makes the comparison
clean.

## Benchmark results

Measured on **200 seeded synthetic instances** (`seed=7`, `iters=400`), each
~24 tasks over a 3-day, 8h/day calendar with fixed meetings. Utility is the
weighted-on-time fraction (of total instance weight); hit-rate is the fraction
of tasks finishing on time. 95% CIs are percentile bootstrap (10,000 resamples).

| method     | utility (mean) | utility 95% CI   | deadline hit-rate | hit-rate 95% CI  |
|------------|---------------:|------------------|------------------:|------------------|
| **slotfit**| **0.715**      | [0.703, 0.727]   | **0.647**         | [0.636, 0.657]   |
| priority   | 0.585          | [0.573, 0.598]   | 0.411             | [0.400, 0.422]   |
| random     | 0.402          | [0.388, 0.416]   | 0.401             | [0.391, 0.411]   |
| fifo       | 0.391          | [0.376, 0.407]   | 0.391             | [0.379, 0.404]   |
| edf        | 0.360          | [0.342, 0.378]   | 0.367             | [0.351, 0.382]   |

**slotfit vs. the strongest baseline (priority-greedy):** utility delta
**+0.130**, paired permutation test **p ≈ 1.0e-4** (floor of a 10,000-permutation
test; the true p-value is smaller). Reproduce with:

```bash
slotfit bench --instances 200 --seed 7 --iters 400
```

Notes on reading these honestly:
- **No method is perfect.** slotfit leaves ~28% of weighted work late or
  unplaced. The instances are over-subscribed on purpose, so 1.0 is unreachable.
- **EDF is worst on utility** here — minimising lateness ignores weights, so on
  over-subscribed instances it spends slots on low-value urgent work. This is
  expected, not a bug.
- The win over priority-greedy comes almost entirely from the drop/re-order
  local search, which the baselines cannot do.

## How this differs from related work

- **vs. a LangGraph/LLM "executive assistant" agent** (the project this was
  modelled to be distinct from): that is an *online* tool that calls Gmail,
  Calendar and Notion through an LLM router. `slotfit` is the opposite end:
  purely offline, deterministic, no model and no I/O — it is the *optimiser* such
  an assistant would need, studied in isolation and benchmarked.
- **vs. a DAG task scheduler / execution engine:** those schedule compute tasks
  on workers for makespan/critical-path. `slotfit` schedules *calendar* work
  against *deadlines and priorities* under working-hours and meeting constraints,
  optimising weighted on-time value rather than completion time.
- **vs. an ops/incident triage engine:** no classification or routing here — this
  is combinatorial scheduling with a measurable objective and statistical
  baselines.

## Install

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"      # dev extras: pytest, hypothesis, ruff
```

Requires Python ≥ 3.10.

## Use

```bash
# Benchmark slotfit against all baselines (writes benchmarks/{results,manifest}.json)
slotfit bench --instances 200 --seed 7 --iters 400
slotfit bench --no-write                         # print only

# Plan a single tasks file (an Instance as JSON; see examples/sample_tasks.json)
slotfit plan examples/sample_tasks.json --method slotfit
slotfit plan examples/sample_tasks.json --method edf      # try a baseline
```

A tasks file is an `Instance`: a list of `tasks` (`id`, `duration`, `deadline`,
`weight`, optional `deps`), optional `meetings`, and `work_windows` as
`(start, end)` minute pairs from a single origin (day 2's 09:00 is
`1*1440 + 540`).

## Reproducibility

- All randomness flows through seeded `numpy.random.default_rng`.
- `benchmarks/manifest.json` records the git short SHA (falls back to `"n/a"`
  outside a git checkout), the Python version, the platform, and a SHA-256
  checksum of the exact dataset the run scored.
- `benchmarks/results.json` is the full result set behind the table above.

## Layout

```
src/slotfit/
  models.py      # pydantic v2 domain models (Task, Meeting, Instance, Schedule)
  timeline.py    # free-interval bookkeeping + earliest-fit placement
  generator.py   # seeded, over-subscribed synthetic instance generator
  schedulers.py  # placement decoder, baselines, slotfit greedy + local search
  metrics.py     # weighted on-time utility, deadline hit-rate
  stats.py       # bootstrap CI + paired permutation test (numpy only)
  benchmark.py   # harness: run all methods, aggregate, write results/manifest
  cli.py         # argparse CLI: `bench`, `plan`
tests/           # 78 tests (pytest + Hypothesis)
benchmarks/      # results.json + manifest.json (generated)
docs/adr/        # architecture decision records
examples/        # sample_tasks.json
```

## Known gaps / limitations

- **Heuristic, not optimal.** The local search gives no optimality guarantee and
  no bound on the gap to the true optimum. It will sometimes settle in a local
  optimum; results depend on `--iters` and the seed.
- **No exact baseline.** There is no ILP/branch-and-bound oracle, so "how far
  from optimal" is unquantified. Adding one (e.g. a small CP/ILP reference on tiny
  instances) is future work.
- **Synthetic data only.** The generator's distribution is a modelling choice;
  numbers will shift on differently-shaped workloads. It is not calibrated to any
  real calendar dataset.
- **Minute granularity, single resource.** One person, one timeline, integer
  minutes, non-preemptive tasks. No travel time, no soft preferences, no
  partial-credit for finishing a task slightly late.
- **Dependencies are hard and finish-to-start only.** A dropped/unplaced task
  cascades to make its dependents unschedulable.

## License

MIT — see [LICENSE](LICENSE).
