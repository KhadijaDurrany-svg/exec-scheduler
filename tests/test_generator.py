from __future__ import annotations

from slotfit.generator import DAY, GenConfig, generate, generate_many
from slotfit.timeline import free_intervals


def test_generate_is_deterministic():
    assert generate(99) == generate(99)


def test_generate_differs_by_seed():
    assert generate(1) != generate(2)


def test_generate_task_count():
    cfg = GenConfig(n_tasks=20)
    inst = generate(5, cfg)
    assert len(inst.tasks) == 20


def test_generated_instance_is_oversubscribed():
    cfg = GenConfig(n_tasks=24, n_days=3, load=1.45)
    inst = generate(7, cfg)
    free = free_intervals(inst.work_windows, [(m.start, m.end) for m in inst.meetings])
    capacity = sum(e - s for s, e in free)
    demand = sum(t.duration for t in inst.tasks)
    assert demand > capacity  # not everything can fit -> choices matter


def test_generated_deps_reference_earlier_tasks():
    inst = generate(13, GenConfig(n_tasks=20))
    index = {t.id: i for i, t in enumerate(inst.tasks)}
    for t in inst.tasks:
        for d in t.deps:
            assert index[d] < index[t.id]


def test_work_windows_span_days():
    cfg = GenConfig(n_days=3)
    inst = generate(3, cfg)
    assert len(inst.work_windows) == 3
    assert inst.work_windows[1][0] - inst.work_windows[0][0] == DAY


def test_generate_many_count_and_determinism():
    a = generate_many(42, 10)
    b = generate_many(42, 10)
    assert len(a) == 10
    assert a == b
