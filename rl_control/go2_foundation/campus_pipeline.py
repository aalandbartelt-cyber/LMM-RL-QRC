"""High-level campus mission pipeline above a reusable locomotion policy.

This module intentionally stops at a body-velocity command.  The next layer is
`Go2ControlCycle`, which converts that command and robot state into a safe
12-joint action.  Keeping the boundary explicit lets patrol and delivery share
the same learned locomotion policy.
"""

from dataclasses import replace
from typing import Protocol

from .command import CommandLimiter
from .mission_scheduler import MissionCommand, PatrolMissionScheduler, WaypointCommandLike


class WaypointControllerLike(Protocol):
    def update(self, x: float, y: float, yaw: float) -> WaypointCommandLike: ...


class CampusMissionPipeline:
    """Compose waypoint navigation, dwell scheduling and velocity limits."""

    def __init__(
        self,
        controller: WaypointControllerLike,
        scheduler: PatrolMissionScheduler,
        limiter: CommandLimiter | None = None,
    ) -> None:
        self.controller = controller
        self.scheduler = scheduler
        self.limiter = limiter or CommandLimiter()

    def step(self, *, time_s: float, x: float, y: float, yaw: float) -> MissionCommand:
        raw_command = self.controller.update(float(x), float(y), float(yaw))
        mission = self.scheduler.update(raw_command, float(time_s))
        return replace(mission, command=self.limiter.apply(mission.command))

    def reset(self) -> None:
        """Reset scheduler and, when offered, the route controller state."""
        self.scheduler.reset()
        reset_controller = getattr(self.controller, "reset", None)
        if callable(reset_controller):
            reset_controller()
