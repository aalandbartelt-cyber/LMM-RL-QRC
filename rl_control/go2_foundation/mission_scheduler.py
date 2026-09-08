"""Mission-level dwell scheduling above the low-level locomotion policy.

The existing waypoint controller provides a target velocity command.  It does
not own inspection dwell time: after it reports a waypoint reached, its next
call targets the following waypoint.  This wrapper adds the task semantics
needed by patrol and delivery without putting waypoint coordinates into the
low-level locomotion policy.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Protocol

from .types import VelocityCommand


class WaypointCommandLike(Protocol):
    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float
    target_index: int
    reached: bool
    mission_complete: bool


@dataclass(frozen=True)
class MissionCommand:
    command: VelocityCommand
    phase: str
    target_index: int
    dwell_remaining_s: float
    mission_complete: bool
    reason: str


class PatrolMissionScheduler:
    """Enforce waypoint inspection dwell while preserving controller ownership.

    ``dwell_by_index`` uses the waypoint index from the original mission route.
    The common controller increments its target index before returning
    ``reached=True``; therefore a non-final arrival is treated as index
    ``target_index - 1``.
    """

    def __init__(self, dwell_by_index: Mapping[int, float]) -> None:
        self._dwell_by_index = {int(index): float(seconds) for index, seconds in dwell_by_index.items()}
        if any(seconds < 0 or not isfinite(seconds) for seconds in self._dwell_by_index.values()):
            raise ValueError("dwell durations must be finite and non-negative")
        self._dwell_until_s: float | None = None
        self._complete_after_dwell = False
        self._last_time_s: float | None = None
        self._last_target_index = 0
        self._complete = False

    @classmethod
    def from_waypoints(cls, waypoints: tuple[object, ...]) -> "PatrolMissionScheduler":
        """Build the dwell map from the existing `Waypoint` dataclass tuple."""
        return cls({index: float(item.dwell_s) for index, item in enumerate(waypoints)})

    def update(self, controller_command: WaypointCommandLike, time_s: float) -> MissionCommand:
        if not isfinite(time_s):
            raise ValueError("time_s must be finite")
        if self._last_time_s is not None and time_s < self._last_time_s:
            raise ValueError("time_s must be non-decreasing")
        self._last_time_s = time_s
        self._last_target_index = int(controller_command.target_index)

        if self._complete:
            self._complete = True
            return self._stopped("mission_complete", 0.0)

        if self._dwell_until_s is not None:
            remaining = self._dwell_until_s - time_s
            if remaining > 0:
                return self._stopped("inspection_dwell", remaining)
            self._dwell_until_s = None
            if self._complete_after_dwell:
                self._complete_after_dwell = False
                self._complete = True
                return self._stopped("mission_complete", 0.0)

        if controller_command.reached:
            arrived_index = max(0, int(controller_command.target_index) - 1)
            dwell = self._dwell_by_index.get(arrived_index, 0.0)
            if dwell > 0:
                self._dwell_until_s = time_s + dwell
                self._complete_after_dwell = bool(controller_command.mission_complete)
                return self._stopped("inspection_dwell", dwell)

        if controller_command.mission_complete:
            self._complete = True
            return self._stopped("mission_complete", 0.0)

        return MissionCommand(
            command=VelocityCommand(
                float(controller_command.forward_mps),
                float(controller_command.lateral_mps),
                float(controller_command.yaw_rate_rps),
            ),
            phase="moving",
            target_index=self._last_target_index,
            dwell_remaining_s=0.0,
            mission_complete=False,
            reason="controller_command",
        )

    def reset(self) -> None:
        self._dwell_until_s = None
        self._complete_after_dwell = False
        self._last_time_s = None
        self._last_target_index = 0
        self._complete = False

    def _stopped(self, reason: str, remaining: float) -> MissionCommand:
        return MissionCommand(
            command=VelocityCommand(),
            phase="complete" if reason == "mission_complete" else "inspecting",
            target_index=self._last_target_index,
            dwell_remaining_s=max(0.0, remaining),
            mission_complete=self._complete,
            reason=reason,
        )
