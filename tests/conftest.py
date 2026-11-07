"""Shared fixtures and a schedule validator used across the test suite."""

from __future__ import annotations

from slotfit.models import Instance, Schedule
from slotfit.timeline import free_intervals


def assert_valid_schedule(instance: Instance, schedule: Schedule) -> None:
    """Assert a schedule obeys every hard constraint of the instance."""
    tasks = {t.id: t for t in instance.tasks}
    placed_ids = [p.task_id for p in schedule.placed]

    # every task appears exactly once, placed or unplaced
    all_ids = set(placed_ids) | set(schedule.unplaced)
    assert all_ids == set(tasks), "schedule must cover every task exactly once"
    assert len(placed_ids) == len(set(placed_ids)), "a task placed twice"
    assert not (set(placed_ids) & set(schedule.unplaced)), "task both placed and unplaced"

    # durations honoured
    for p in schedule.placed:
        assert p.finish - p.start == tasks[p.task_id].duration

    # no two placed tasks overlap
    blocks = sorted((p.start, p.finish) for p in schedule.placed)
    for (_, e1), (s2, _) in zip(blocks, blocks[1:], strict=False):
        assert e1 <= s2, "placed tasks overlap"

    # every placed block lies inside a free interval (within hours, off meetings)
    free = free_intervals(instance.work_windows, [(m.start, m.end) for m in instance.meetings])
    for p in schedule.placed:
        assert any(fs <= p.start and p.finish <= fe for fs, fe in free), "task outside free time"

    # dependencies finish before dependents start
    finish = {p.task_id: p.finish for p in schedule.placed}
    for p in schedule.placed:
        for dep in tasks[p.task_id].deps:
            assert dep in finish, "dependent placed while dependency was not"
            assert finish[dep] <= p.start, "dependency finishes after dependent starts"
