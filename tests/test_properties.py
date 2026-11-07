"""Property-based tests (Hypothesis).

Instead of fixed examples these assert invariants that must hold for *any*
instance the strategy produces: schedules stay feasible and metrics stay in
range no matter the inputs.
"""

from __future__ import annotations

from conftest import assert_valid_schedule
from hypothesis import given, settings
from hypothesis import strategies as st

from slotfit.metrics import deadline_hit_rate, utility_fraction
from slotfit.models import Instance, Meeting, Task
from slotfit.schedulers import SCHEDULERS, decode
from slotfit.timeline import Timeline, free_intervals


@st.composite
def instances(draw: st.DrawFn) -> Instance:
    n = draw(st.integers(min_value=1, max_value=10))
    n_days = draw(st.integers(min_value=1, max_value=3))
    work_len = draw(st.integers(min_value=120, max_value=480))
    windows = tuple((d * 1440 + 540, d * 1440 + 540 + work_len) for d in range(n_days))
    horizon_end = windows[-1][1]

    tasks = []
    for i in range(n):
        dur = draw(st.integers(min_value=15, max_value=120))
        deadline = draw(st.integers(min_value=540, max_value=horizon_end))
        weight = float(draw(st.integers(min_value=1, max_value=8)))
        # only depend on strictly-earlier tasks -> acyclic by construction
        deps = tuple(sorted(set(draw(st.lists(st.integers(min_value=0, max_value=i - 1), max_size=2))))) if i else ()
        tasks.append(
            Task(id=f"t{i}", duration=dur, deadline=deadline, weight=weight, deps=tuple(f"t{d}" for d in deps))
        )

    meetings = []
    if draw(st.booleans()) and work_len > 120:
        ms = windows[0][0] + 30
        meetings.append(Meeting(id="m0", start=ms, end=ms + 30))
    return Instance(tasks=tuple(tasks), meetings=tuple(meetings), work_windows=windows)


@given(inst=instances())
@settings(max_examples=120, deadline=None)
def test_decode_always_feasible(inst: Instance):
    order = [t.id for t in inst.tasks]
    assert_valid_schedule(inst, decode(inst, order))


@given(inst=instances())
@settings(max_examples=80, deadline=None)
def test_all_schedulers_feasible_and_metrics_in_range(inst: Instance):
    for fn in SCHEDULERS.values():
        sched = fn(inst, 0)
        assert_valid_schedule(inst, sched)
        assert 0.0 <= utility_fraction(inst, sched) <= 1.0 + 1e-9
        assert 0.0 <= deadline_hit_rate(inst, sched) <= 1.0 + 1e-9


@given(inst=instances())
@settings(max_examples=120, deadline=None)
def test_free_intervals_disjoint_and_within_windows(inst: Instance):
    free = free_intervals(inst.work_windows, [(m.start, m.end) for m in inst.meetings])
    for s, e in free:
        assert s < e
        assert any(ws <= s and e <= we for ws, we in inst.work_windows)
    for (_, e1), (s2, _) in zip(free, free[1:], strict=False):
        assert e1 <= s2


@given(
    blocks=st.lists(
        st.tuples(st.integers(0, 500), st.integers(1, 200)).map(lambda t: (t[0], t[0] + t[1])),
        min_size=0, max_size=6,
    )
)
@settings(max_examples=100, deadline=None)
def test_timeline_reservations_never_overlap(blocks):
    tl = Timeline([(0, 1000)])
    placed = []
    for dur in [b[1] - b[0] for b in blocks]:
        start = tl.place(dur)
        if start is not None:
            placed.append((start, start + dur))
    placed.sort()
    for (_, e1), (s2, _) in zip(placed, placed[1:], strict=False):
        assert e1 <= s2
