"""Typed domain models for slotfit (pydantic v2).

All times are integer minutes measured from a single timeline origin (minute 0).
A multi-day horizon is expressed purely as working-hours windows, e.g. day 2's
09:00 is ``1 * 1440 + 540``. The planner never needs wall-clock dates.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(BaseModel):
    """A unit of work to be scheduled without preemption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = ""
    duration: int = Field(gt=0, description="minutes of focused work")
    deadline: int = Field(ge=0, description="latest finish minute to count as on-time")
    weight: float = Field(gt=0, description="priority weight / value of finishing on time")
    deps: tuple[str, ...] = Field(default=())

    @model_validator(mode="after")
    def _no_self_dependency(self) -> Task:
        if self.id in self.deps:
            raise ValueError(f"task {self.id!r} depends on itself")
        if len(set(self.deps)) != len(self.deps):
            raise ValueError(f"task {self.id!r} has duplicate dependencies")
        return self


class Meeting(BaseModel):
    """A fixed, immovable block of time (clipped to working hours when used)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = ""
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _ordered(self) -> Meeting:
        if self.end <= self.start:
            raise ValueError(f"meeting {self.id!r} ends before it starts")
        return self


class Instance(BaseModel):
    """A complete scheduling problem instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tasks: tuple[Task, ...]
    meetings: tuple[Meeting, ...] = ()
    work_windows: tuple[tuple[int, int], ...] = Field(
        description="disjoint (start, end) minute windows available for work",
    )

    @model_validator(mode="after")
    def _validate(self) -> Instance:
        ids = [t.id for t in self.tasks]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate task ids")
        known = set(ids)
        for t in self.tasks:
            for d in t.deps:
                if d not in known:
                    raise ValueError(f"task {t.id!r} depends on unknown task {d!r}")
        for s, e in self.work_windows:
            if e <= s:
                raise ValueError(f"empty work window ({s}, {e})")
        # windows must be sorted and disjoint for the timeline engine
        prev_end = -1
        for s, e in self.work_windows:
            if s < prev_end:
                raise ValueError("work windows must be sorted and disjoint")
            prev_end = e
        if self._has_cycle():
            raise ValueError("dependency graph contains a cycle")
        return self

    def _has_cycle(self) -> bool:
        adj = {t.id: list(t.deps) for t in self.tasks}
        color: dict[str, int] = {}

        def visit(node: str) -> bool:
            color[node] = 1
            for nxt in adj.get(node, ()):
                c = color.get(nxt, 0)
                if c == 1 or (c == 0 and visit(nxt)):
                    return True
            color[node] = 2
            return False

        return any(color.get(t.id, 0) == 0 and visit(t.id) for t in self.tasks)

    def total_weight(self) -> float:
        return sum(t.weight for t in self.tasks)


class PlacedTask(BaseModel):
    """A task assigned a concrete start/finish on the timeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    start: int
    finish: int


class Schedule(BaseModel):
    """A decoded assignment of (some) tasks to timeline slots."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placed: tuple[PlacedTask, ...] = ()
    unplaced: tuple[str, ...] = ()

    def finish_of(self, task_id: str) -> int | None:
        for p in self.placed:
            if p.task_id == task_id:
                return p.finish
        return None
