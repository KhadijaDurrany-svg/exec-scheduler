"""Seeded synthetic instance generator.

Instances are deliberately *over-subscribed*: the total task-minutes exceed the
free working capacity, so no scheduler can finish everything on time and the
choice of *which* tasks to drop or delay actually matters. This keeps the
benchmark honest -- a trivially-feasible instance would let every method score
1.0 and hide any real difference between them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from slotfit.models import Instance, Meeting, Task

DAY = 1440


@dataclass(frozen=True)
class GenConfig:
    """Knobs for the synthetic generator."""

    n_tasks: int = 24
    n_days: int = 3
    work_start: int = 9 * 60       # 09:00
    work_len: int = 8 * 60         # 8h working day
    meetings_per_day: int = 2
    meeting_len_choices: tuple[int, ...] = (30, 45, 60)
    duration_choices: tuple[int, ...] = (30, 45, 60, 90, 120)
    weight_choices: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0)
    load: float = 1.45            # task-minutes / free capacity (>1 == oversubscribed)
    dep_prob: float = 0.22        # chance a task depends on an earlier one
    tight_frac: float = 0.45      # fraction of tasks given a tight deadline


def _work_windows(cfg: GenConfig) -> list[tuple[int, int]]:
    return [(d * DAY + cfg.work_start, d * DAY + cfg.work_start + cfg.work_len) for d in range(cfg.n_days)]


def _meetings(cfg: GenConfig, rng: np.random.Generator) -> list[Meeting]:
    meetings: list[Meeting] = []
    mid = 0
    for d in range(cfg.n_days):
        base = d * DAY + cfg.work_start
        # carve non-overlapping meeting slots within the working day
        slots = sorted(rng.choice(np.arange(0, cfg.work_len - 90, 30), size=cfg.meetings_per_day, replace=False))
        for off in slots:
            length = int(rng.choice(cfg.meeting_len_choices))
            start = base + int(off)
            meetings.append(Meeting(id=f"m{mid}", name=f"meeting {mid}", start=start, end=start + length))
            mid += 1
    return meetings


def generate(seed: int, cfg: GenConfig | None = None) -> Instance:
    """Produce a deterministic instance for ``seed``."""
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(seed)
    windows = _work_windows(cfg)
    meetings = _meetings(cfg, rng)

    capacity = sum(e - s for s, e in windows) - sum(m.end - m.start for m in meetings)
    budget = int(cfg.load * capacity)

    durations: list[int] = []
    used = 0
    while len(durations) < cfg.n_tasks:
        d = int(rng.choice(cfg.duration_choices))
        durations.append(d)
        used += d
        if used >= budget and len(durations) >= max(4, cfg.n_tasks // 2):
            break
    # pad to n_tasks if the budget filled early
    while len(durations) < cfg.n_tasks:
        durations.append(int(rng.choice(cfg.duration_choices)))

    n = len(durations)
    horizon_start = windows[0][0]
    horizon_end = windows[-1][1]
    span = horizon_end - horizon_start

    tasks: list[Task] = []
    for i in range(n):
        dur = durations[i]
        weight = float(rng.choice(cfg.weight_choices))
        # earliest plausible finish grows through the horizon
        base_floor = horizon_start + int(span * (i / max(1, n)) * 0.5)
        if rng.random() < cfg.tight_frac:
            slack = int(rng.integers(dur, dur * 3 + 1))
        else:
            slack = int(rng.integers(dur * 3, span))
        deadline = min(horizon_end, base_floor + dur + slack)

        deps: list[str] = []
        if i > 0 and rng.random() < cfg.dep_prob:
            # depend on 1..2 earlier tasks (keeps the graph acyclic by construction)
            k = int(rng.integers(1, 3))
            choices = rng.choice(i, size=min(k, i), replace=False)
            deps = [f"t{int(c)}" for c in np.atleast_1d(choices)]
        tasks.append(
            Task(id=f"t{i}", name=f"task {i}", duration=dur, deadline=deadline, weight=weight, deps=tuple(deps))
        )

    return Instance(tasks=tuple(tasks), meetings=tuple(meetings), work_windows=tuple(windows))


def generate_many(base_seed: int, count: int, cfg: GenConfig | None = None) -> list[Instance]:
    """A reproducible family of instances derived from ``base_seed``."""
    seeds = np.random.default_rng(base_seed).integers(0, 2**31 - 1, size=count)
    return [generate(int(s), cfg) for s in seeds]
