"""Transparent objective and reporting metrics.

The single objective the planner optimises is **weighted on-time priority**:
the sum of priority weights of tasks that are scheduled *and* finish on or
before their deadline. We report it as a fraction of the instance's total
weight so instances of different sizes are comparable.
"""

from __future__ import annotations

from slotfit.models import Instance, Schedule


def on_time_task_ids(instance: Instance, schedule: Schedule) -> set[str]:
    deadlines = {t.id: t.deadline for t in instance.tasks}
    return {p.task_id for p in schedule.placed if p.finish <= deadlines[p.task_id]}


def weighted_on_time(instance: Instance, schedule: Schedule) -> float:
    """Absolute utility: total weight of tasks finishing on time."""
    weights = {t.id: t.weight for t in instance.tasks}
    done = on_time_task_ids(instance, schedule)
    return sum(weights[tid] for tid in done)


def utility_fraction(instance: Instance, schedule: Schedule) -> float:
    """Utility normalised to [0, 1] by the instance's total weight."""
    total = instance.total_weight()
    if total <= 0:
        return 0.0
    return weighted_on_time(instance, schedule) / total


def deadline_hit_rate(instance: Instance, schedule: Schedule) -> float:
    """Fraction of all tasks (by count) that finished on time."""
    if not instance.tasks:
        return 0.0
    return len(on_time_task_ids(instance, schedule)) / len(instance.tasks)
