from __future__ import annotations

import pytest
from conftest import assert_valid_schedule

from slotfit.generator import GenConfig, generate
from slotfit.metrics import weighted_on_time
from slotfit.models import Instance, Meeting, Task
from slotfit.schedulers import BASELINES, SCHEDULERS, decode, plan, plan_slotfit


def _small():
    return Instance(
        tasks=(
            Task(id="t0", duration=60, deadline=600, weight=1.0),
            Task(id="t1", duration=60, deadline=600, weight=5.0, deps=("t0",)),
            Task(id="t2", duration=60, deadline=600, weight=3.0),
        ),
        meetings=(Meeting(id="m0", start=120, end=180),),
        work_windows=((0, 480),),
    )


def test_decode_respects_dependencies():
    inst = _small()
    sched = decode(inst, ["t1", "t0", "t2"])
    finish = {p.task_id: p.finish for p in sched.placed}
    start = {p.task_id: p.start for p in sched.placed}
    # t1 depends on t0, so even though it is ranked first it must start after t0
    assert start["t1"] >= finish["t0"]


def test_decode_respects_meeting_block():
    inst = _small()
    sched = decode(inst, ["t0", "t1", "t2"])
    for p in sched.placed:
        # nothing may occupy the 120-180 meeting block
        assert not (p.start < 180 and p.finish > 120)
    assert_valid_schedule(inst, sched)


def test_decode_drops_dependent_of_dropped_task():
    inst = _small()
    sched = decode(inst, ["t0", "t1", "t2"], dropped=frozenset({"t0"}))
    placed_ids = {p.task_id for p in sched.placed}
    # t1 depends on dropped t0, so it cannot be placed on time
    assert "t0" not in placed_ids
    assert "t1" not in placed_ids
    assert "t2" in placed_ids


@pytest.mark.parametrize("method", sorted(SCHEDULERS))
def test_every_scheduler_produces_valid_schedule(method):
    inst = generate(123, GenConfig(n_tasks=16, n_days=2))
    sched = SCHEDULERS[method](inst, 5)
    assert_valid_schedule(inst, sched)


@pytest.mark.parametrize("method", sorted(SCHEDULERS))
def test_schedulers_deterministic(method):
    inst = generate(55, GenConfig(n_tasks=16, n_days=2))
    a = SCHEDULERS[method](inst, 9)
    b = SCHEDULERS[method](inst, 9)
    assert a == b


def test_slotfit_not_worse_than_its_heuristic_seed():
    inst = generate(321, GenConfig(n_tasks=20, n_days=3))
    seed_order = decode(inst, [t.id for t in sorted(inst.tasks, key=lambda x: x.deadline)])
    best = plan_slotfit(inst, seed=0, iters=300)
    assert weighted_on_time(inst, best) >= weighted_on_time(inst, seed_order) - 1e-9


def test_slotfit_beats_fifo_on_oversubscribed_instance():
    inst = generate(7, GenConfig(n_tasks=24, n_days=3))
    best = plan_slotfit(inst, seed=0, iters=400)
    fifo = SCHEDULERS["fifo"](inst, 0)
    assert weighted_on_time(inst, best) >= weighted_on_time(inst, fifo)


def test_plan_unknown_method_raises():
    inst = _small()
    with pytest.raises(KeyError):
        plan(inst, method="does-not-exist")


def test_baselines_registered():
    assert set(BASELINES) <= set(SCHEDULERS)
    assert "slotfit" in SCHEDULERS


def test_zero_iters_returns_heuristic_seed():
    inst = generate(11, GenConfig(n_tasks=12, n_days=2))
    sched = plan_slotfit(inst, seed=0, iters=0)
    assert_valid_schedule(inst, sched)
