"""Free-time bookkeeping for the placement engine.

A :class:`Timeline` holds a sorted, disjoint set of free intervals (working
hours minus fixed meetings). It answers one question efficiently: given a task
of some duration that cannot start before ``not_before``, what is the earliest
minute it can be placed, and reserve that block.

The engine is non-preemptive: a task occupies one contiguous block.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def free_intervals(
    work_windows: Sequence[tuple[int, int]],
    meetings: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Subtract meeting blocks from working-hours windows.

    Meetings are clipped to the windows; overlap between meetings is handled by
    merging them first. Returns sorted, disjoint free intervals.
    """
    blocks = sorted((s, e) for s, e in meetings if e > s)
    merged: list[tuple[int, int]] = []
    for s, e in blocks:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    free: list[tuple[int, int]] = []
    for ws, we in work_windows:
        cursor = ws
        for ms, me in merged:
            if me <= cursor or ms >= we:
                continue
            if ms > cursor:
                free.append((cursor, min(ms, we)))
            cursor = max(cursor, me)
            if cursor >= we:
                break
        if cursor < we:
            free.append((cursor, we))
    return free


class Timeline:
    """Mutable free-interval set supporting earliest-fit placement."""

    __slots__ = ("_free",)

    def __init__(self, free: Iterable[tuple[int, int]]):
        # keep sorted + disjoint; defensive copy
        self._free: list[tuple[int, int]] = [(s, e) for s, e in free if e > s]
        self._free.sort()

    @property
    def free(self) -> list[tuple[int, int]]:
        return list(self._free)

    def capacity(self) -> int:
        return sum(e - s for s, e in self._free)

    def place(self, duration: int, not_before: int = 0) -> int | None:
        """Reserve the earliest block of ``duration`` starting >= ``not_before``.

        Returns the start minute, or ``None`` if no feasible block exists.
        """
        if duration <= 0:
            raise ValueError("duration must be positive")
        for idx, (s, e) in enumerate(self._free):
            cand = s if s >= not_before else not_before
            if cand + duration <= e:
                self._reserve(idx, s, e, cand, cand + duration)
                return cand
        return None

    def _reserve(self, idx: int, s: int, e: int, rs: int, re: int) -> None:
        pieces: list[tuple[int, int]] = []
        if rs > s:
            pieces.append((s, rs))
        if re < e:
            pieces.append((re, e))
        self._free[idx : idx + 1] = pieces
