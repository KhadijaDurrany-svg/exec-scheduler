from __future__ import annotations

from slotfit.metrics import deadline_hit_rate, on_time_task_ids, utility_fraction, weighted_on_time
from slotfit.models import Instance, PlacedTask, Schedule, Task


def _inst():
    return Instance(
        tasks=(
            Task(id="a", duration=30, deadline=100, weight=2.0),
            Task(id="b", duration=30, deadline=50, weight=3.0),
            Task(id="c", duration=30, deadline=100, weight=5.0),
        ),
        work_windows=((0, 1000),),
    )


def test_weighted_on_time_counts_only_on_time():
    inst = _inst()
    # a finishes on time (2.0); b finishes late (0); c not placed (0)
    sched = Schedule(
        placed=(PlacedTask(task_id="a", start=0, finish=30), PlacedTask(task_id="b", start=60, finish=90)),
        unplaced=("c",),
    )
    assert weighted_on_time(inst, sched) == 2.0
    assert on_time_task_ids(inst, sched) == {"a"}


def test_utility_fraction_normalises():
    inst = _inst()
    sched = Schedule(placed=(PlacedTask(task_id="c", start=0, finish=30),), unplaced=("a", "b"))
    assert utility_fraction(inst, sched) == 5.0 / 10.0


def test_deadline_hit_rate_counts():
    inst = _inst()
    sched = Schedule(
        placed=(PlacedTask(task_id="a", start=0, finish=30), PlacedTask(task_id="b", start=0, finish=30)),
        unplaced=("c",),
    )
    assert deadline_hit_rate(inst, sched) == 2 / 3


def test_empty_schedule_is_zero():
    inst = _inst()
    sched = Schedule(placed=(), unplaced=("a", "b", "c"))
    assert utility_fraction(inst, sched) == 0.0
    assert deadline_hit_rate(inst, sched) == 0.0


def test_exact_deadline_counts_as_on_time():
    inst = Instance(tasks=(Task(id="a", duration=30, deadline=30, weight=1.0),), work_windows=((0, 100),))
    sched = Schedule(placed=(PlacedTask(task_id="a", start=0, finish=30),))
    assert weighted_on_time(inst, sched) == 1.0
