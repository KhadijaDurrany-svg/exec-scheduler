from __future__ import annotations

import pytest
from pydantic import ValidationError

from slotfit.models import Instance, Meeting, PlacedTask, Schedule, Task


def test_task_basic_ok():
    t = Task(id="a", duration=30, deadline=100, weight=2.0)
    assert t.duration == 30 and t.weight == 2.0 and t.deps == ()


def test_task_rejects_nonpositive_duration():
    with pytest.raises(ValidationError):
        Task(id="a", duration=0, deadline=100, weight=1.0)


def test_task_rejects_nonpositive_weight():
    with pytest.raises(ValidationError):
        Task(id="a", duration=30, deadline=100, weight=0.0)


def test_task_rejects_self_dependency():
    with pytest.raises(ValidationError):
        Task(id="a", duration=30, deadline=100, weight=1.0, deps=("a",))


def test_task_rejects_duplicate_deps():
    with pytest.raises(ValidationError):
        Task(id="a", duration=30, deadline=100, weight=1.0, deps=("b", "b"))


def test_task_is_frozen():
    t = Task(id="a", duration=30, deadline=100, weight=1.0)
    with pytest.raises(ValidationError):
        t.duration = 40  # type: ignore[misc]


def test_meeting_rejects_bad_order():
    with pytest.raises(ValidationError):
        Meeting(id="m", start=100, end=100)


def test_instance_rejects_duplicate_ids():
    with pytest.raises(ValidationError):
        Instance(
            tasks=(Task(id="a", duration=1, deadline=1, weight=1.0), Task(id="a", duration=1, deadline=1, weight=1.0)),
            work_windows=((0, 100),),
        )


def test_instance_rejects_unknown_dependency():
    with pytest.raises(ValidationError):
        Instance(
            tasks=(Task(id="a", duration=1, deadline=1, weight=1.0, deps=("ghost",)),),
            work_windows=((0, 100),),
        )


def test_instance_rejects_cycle():
    with pytest.raises(ValidationError):
        Instance(
            tasks=(
                Task(id="a", duration=1, deadline=1, weight=1.0, deps=("b",)),
                Task(id="b", duration=1, deadline=1, weight=1.0, deps=("a",)),
            ),
            work_windows=((0, 100),),
        )


def test_instance_rejects_unsorted_windows():
    with pytest.raises(ValidationError):
        Instance(
            tasks=(Task(id="a", duration=1, deadline=1, weight=1.0),),
            work_windows=((100, 200), (0, 50)),
        )


def test_instance_rejects_overlapping_windows():
    with pytest.raises(ValidationError):
        Instance(
            tasks=(Task(id="a", duration=1, deadline=1, weight=1.0),),
            work_windows=((0, 100), (50, 150)),
        )


def test_instance_total_weight():
    inst = Instance(
        tasks=(
            Task(id="a", duration=1, deadline=1, weight=2.0),
            Task(id="b", duration=1, deadline=1, weight=3.0),
        ),
        work_windows=((0, 100),),
    )
    assert inst.total_weight() == 5.0


def test_schedule_finish_of():
    s = Schedule(placed=(PlacedTask(task_id="a", start=0, finish=30),), unplaced=("b",))
    assert s.finish_of("a") == 30
    assert s.finish_of("b") is None
