from __future__ import annotations

import pytest

from slotfit.timeline import Timeline, free_intervals


def test_free_intervals_no_meetings():
    assert free_intervals([(0, 100)], []) == [(0, 100)]


def test_free_intervals_single_meeting_splits_window():
    assert free_intervals([(0, 100)], [(40, 60)]) == [(0, 40), (60, 100)]


def test_free_intervals_meeting_at_edge():
    assert free_intervals([(0, 100)], [(0, 30)]) == [(30, 100)]


def test_free_intervals_overlapping_meetings_merge():
    assert free_intervals([(0, 100)], [(20, 50), (40, 70)]) == [(0, 20), (70, 100)]


def test_free_intervals_meeting_outside_window_ignored():
    assert free_intervals([(0, 100)], [(200, 300)]) == [(0, 100)]


def test_free_intervals_multi_day():
    out = free_intervals([(0, 100), (1000, 1100)], [(50, 60)])
    assert out == [(0, 50), (60, 100), (1000, 1100)]


def test_timeline_place_earliest_fit():
    tl = Timeline([(0, 100)])
    assert tl.place(30) == 0
    assert tl.place(30) == 30


def test_timeline_place_respects_not_before():
    tl = Timeline([(0, 100)])
    assert tl.place(10, not_before=50) == 50
    assert tl.free == [(0, 50), (60, 100)]


def test_timeline_place_skips_small_gap():
    tl = Timeline([(0, 20), (100, 200)])
    assert tl.place(50) == 100


def test_timeline_place_returns_none_when_no_fit():
    tl = Timeline([(0, 20)])
    assert tl.place(50) is None


def test_timeline_capacity_tracks_reservations():
    tl = Timeline([(0, 100)])
    assert tl.capacity() == 100
    tl.place(30)
    assert tl.capacity() == 70


def test_timeline_rejects_nonpositive_duration():
    tl = Timeline([(0, 100)])
    with pytest.raises(ValueError):
        tl.place(0)
