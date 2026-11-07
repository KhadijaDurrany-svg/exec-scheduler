"""Placement engine, baseline orderings, and the slotfit planner.

Design
------
Every scheduler shares one deterministic *decoder* (:func:`decode`): given a
priority order over task ids and a set of dropped tasks, it greedily places
tasks into the earliest feasible slot, respecting dependencies, meetings and
working hours. Baselines differ only in how they order tasks; they never drop.

The slotfit planner starts from a deadline/value heuristic order and then runs
a seeded hill-climbing local search over (order, dropped-set). Dropping lets it
sacrifice a cheap-but-late task to free a slot for a more valuable one -- the
move the simple baselines cannot make. The search is a heuristic and is *not*
guaranteed optimal (see README "Known gaps").
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from slotfit.metrics import weighted_on_time
from slotfit.models import Instance, PlacedTask, Schedule
from slotfit.timeline import Timeline, free_intervals


def decode(instance: Instance, order: Sequence[str], dropped: frozenset[str] = frozenset()) -> Schedule:
    """Greedily place tasks into the earliest feasible slot.

    ``order`` is a priority ranking over all task ids; among tasks whose
    dependencies are already resolved the earliest-ranked one is placed first.
    A task is left unplaced if any dependency was dropped/unplaced or no slot
    of its duration remains after ``not_before``.
    """
    tasks = {t.id: t for t in instance.tasks}
    rank = {tid: i for i, tid in enumerate(order)}
    for tid in tasks:
        rank.setdefault(tid, len(rank))

    active = {tid for tid in tasks if tid not in dropped}
    # indegree counts only dependencies that are themselves active
    indeg: dict[str, int] = {}
    dependents: dict[str, list[str]] = {tid: [] for tid in tasks}
    for tid in active:
        deps_active = [d for d in tasks[tid].deps if d in active]
        indeg[tid] = len(deps_active)
        for d in deps_active:
            dependents[d].append(tid)

    ready = sorted((tid for tid in active if indeg[tid] == 0), key=lambda t: rank[t])
    timeline = Timeline(free_intervals(instance.work_windows, [(m.start, m.end) for m in instance.meetings]))

    finish: dict[str, int] = {}
    placed: list[PlacedTask] = []
    unplaced: list[str] = []

    while ready:
        tid = ready.pop(0)
        task = tasks[tid]
        dep_ok = all(d in finish for d in task.deps)
        if dep_ok:
            not_before = max((finish[d] for d in task.deps), default=0)
            start = timeline.place(task.duration, not_before)
            if start is None:
                unplaced.append(tid)
            else:
                placed.append(PlacedTask(task_id=tid, start=start, finish=start + task.duration))
                finish[tid] = start + task.duration
        else:
            unplaced.append(tid)
        for child in dependents[tid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                # insert keeping ready sorted by rank
                lo, hi = 0, len(ready)
                while lo < hi:
                    mid = (lo + hi) // 2
                    if rank[ready[mid]] < rank[child]:
                        lo = mid + 1
                    else:
                        hi = mid
                ready.insert(lo, child)

    for tid in dropped:
        unplaced.append(tid)
    return Schedule(placed=tuple(placed), unplaced=tuple(unplaced))


# --------------------------------------------------------------------------- #
# Baseline orderings
# --------------------------------------------------------------------------- #

def _order_fifo(instance: Instance) -> list[str]:
    return [t.id for t in instance.tasks]


def _order_edf(instance: Instance) -> list[str]:
    idx = {t.id: i for i, t in enumerate(instance.tasks)}
    return sorted((t.id for t in instance.tasks), key=lambda tid: (_deadline(instance, tid), idx[tid]))


def _order_priority(instance: Instance) -> list[str]:
    idx = {t.id: i for i, t in enumerate(instance.tasks)}
    return sorted((t.id for t in instance.tasks), key=lambda tid: (-_weight(instance, tid), idx[tid]))


def _order_random(instance: Instance, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    ids = [t.id for t in instance.tasks]
    perm = rng.permutation(len(ids))
    return [ids[i] for i in perm]


def _deadline(instance: Instance, tid: str) -> int:
    return next(t.deadline for t in instance.tasks if t.id == tid)


def _weight(instance: Instance, tid: str) -> float:
    return next(t.weight for t in instance.tasks if t.id == tid)


def schedule_fifo(instance: Instance, seed: int = 0) -> Schedule:
    return decode(instance, _order_fifo(instance))


def schedule_edf(instance: Instance, seed: int = 0) -> Schedule:
    return decode(instance, _order_edf(instance))


def schedule_priority(instance: Instance, seed: int = 0) -> Schedule:
    return decode(instance, _order_priority(instance))


def schedule_random(instance: Instance, seed: int = 0) -> Schedule:
    return decode(instance, _order_random(instance, seed))


# --------------------------------------------------------------------------- #
# slotfit planner: heuristic seed + local search
# --------------------------------------------------------------------------- #

def _heuristic_order(instance: Instance) -> list[str]:
    """Blend of earliest-deadline and value-density (weight per minute)."""
    idx = {t.id: i for i, t in enumerate(instance.tasks)}
    return sorted(
        (t.id for t in instance.tasks),
        key=lambda tid: (
            _deadline(instance, tid),
            -_weight(instance, tid) / _duration(instance, tid),
            idx[tid],
        ),
    )


def _duration(instance: Instance, tid: str) -> int:
    return next(t.duration for t in instance.tasks if t.id == tid)


def plan_slotfit(instance: Instance, seed: int = 0, iters: int = 400) -> Schedule:
    """Greedy heuristic seed improved by seeded hill-climbing local search."""
    rng = np.random.default_rng(seed)
    order = _heuristic_order(instance)
    dropped: frozenset[str] = frozenset()
    best_sched = decode(instance, order, dropped)
    best_util = weighted_on_time(instance, best_sched)

    cur_order, cur_dropped, cur_util = order, dropped, best_util
    n = len(order)
    if n < 2:
        return best_sched

    for _ in range(iters):
        move = rng.integers(0, 3)
        new_order = list(cur_order)
        new_dropped = set(cur_dropped)

        if move == 0:
            # swap two positions in the priority order
            i, j = rng.integers(0, n), rng.integers(0, n)
            new_order[i], new_order[j] = new_order[j], new_order[i]
        elif move == 1:
            # drop a currently-late / unplaced task to free its slot
            late = _late_or_unplaced(instance, cur_order, frozenset(cur_dropped))
            if not late:
                continue
            new_dropped.add(late[int(rng.integers(0, len(late)))])
        else:
            # un-drop a task (reconsider a previous sacrifice)
            if not new_dropped:
                continue
            victim = list(new_dropped)[int(rng.integers(0, len(new_dropped)))]
            new_dropped.discard(victim)

        cand = decode(instance, new_order, frozenset(new_dropped))
        util = weighted_on_time(instance, cand)
        # accept non-worsening moves (plateau moves help escape ties)
        if util >= cur_util:
            cur_order, cur_dropped, cur_util = new_order, frozenset(new_dropped), util
            if util > best_util:
                best_util, best_sched = util, cand

    return best_sched


def _late_or_unplaced(instance: Instance, order: Sequence[str], dropped: frozenset[str]) -> list[str]:
    sched = decode(instance, order, dropped)
    deadlines = {t.id: t.deadline for t in instance.tasks}
    late = [p.task_id for p in sched.placed if p.finish > deadlines[p.task_id]]
    late.extend(tid for tid in sched.unplaced if tid not in dropped)
    return late


SCHEDULERS: dict[str, Callable[[Instance, int], Schedule]] = {
    "fifo": schedule_fifo,
    "edf": schedule_edf,
    "priority": schedule_priority,
    "random": schedule_random,
    "slotfit": plan_slotfit,
}

BASELINES = ("fifo", "edf", "priority", "random")


def plan(instance: Instance, method: str = "slotfit", seed: int = 0) -> Schedule:
    """Run a named scheduler (``slotfit`` or a baseline) on an instance."""
    if method not in SCHEDULERS:
        raise KeyError(f"unknown scheduler {method!r}; choose from {sorted(SCHEDULERS)}")
    return SCHEDULERS[method](instance, seed)
