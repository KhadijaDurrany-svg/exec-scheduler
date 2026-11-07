"""slotfit: offline constraint-based day planner.

Given tasks with durations, deadlines, priority weights and dependencies,
plus fixed meetings and working-hours windows, build a feasible non-preemptive
schedule that maximises a transparent utility (weighted on-time priority).

Everything here is offline, deterministic and seeded. No network, no keys.
"""

from __future__ import annotations

from slotfit.models import Instance, Meeting, PlacedTask, Schedule, Task
from slotfit.schedulers import SCHEDULERS, plan

__all__ = [
    "SCHEDULERS",
    "Instance",
    "Meeting",
    "PlacedTask",
    "Schedule",
    "Task",
    "plan",
]

__version__ = "0.1.0"
