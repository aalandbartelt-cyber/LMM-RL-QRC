"""Reusable high-level route controller for patrol and delivery missions.

This is intentionally a *velocity-command* controller, not a learned policy.
It maps route geometry into ``vx, vy, yaw_rate``.  The learned PPO policy stays
responsible for turning that velocity command into safe joint motion, so a new
delivery route does not require retraining the gait.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi
from pathlib import Path
from typing import Iterable


def wrap_angle(angle_rad: float) -> float:
    """Return an angle in ``[-pi, pi)``."""
    return (float(angle_rad) + pi) % (2.0 * pi) - pi


@dataclass(frozen=True)
class MissionWaypoint:
    """A task-level waypoint; optional dwell is handled by the scheduler."""

    name: str
    x_m: float
    y_m: float
    dwell_s: float = 0.0
    task: str = "pass_through"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("waypoint name cannot be blank")
        if not all(isfinite(value) for value in (self.x_m, self.y_m, self.dwell_s)):
            raise ValueError("waypoint values must be finite")
        if self.dwell_s < 0:
            raise ValueError("waypoint dwell_s must be non-negative")


@dataclass(frozen=True)
class RouteVelocityOutput:
    """Compatibility shape accepted by ``CampusMissionPipeline``."""

    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float
    target_index: int
    target_name: str
    reached: bool
    mission_complete: bool


class RouteVelocityController:
    """A conservative pure-pursuit-like controller for a finite route.

    It owns geometry only.  Obstacle avoidance and global rerouting should live
    above it, while gait execution remains below it.  The controller returns
    zero forward velocity when heading error exceeds ``turn_in_place_rad``.
    """

    def __init__(
        self,
        waypoints: Iterable[MissionWaypoint],
        *,
        max_speed_mps: float = 0.8,
        max_yaw_rate_rps: float = 1.0,
        reach_radius_m: float = 0.8,
        yaw_gain: float = 1.5,
        turn_in_place_rad: float = 1.1,
    ) -> None:
        self.waypoints = tuple(waypoints)
        if len(self.waypoints) < 2:
            raise ValueError("a route needs at least a start and a target waypoint")
        values = (max_speed_mps, max_yaw_rate_rps, reach_radius_m, yaw_gain, turn_in_place_rad)
        if not all(isfinite(value) and value > 0 for value in values):
            raise ValueError("controller limits must be finite and positive")
        self.max_speed_mps = float(max_speed_mps)
        self.max_yaw_rate_rps = float(max_yaw_rate_rps)
        self.reach_radius_m = float(reach_radius_m)
        self.yaw_gain = float(yaw_gain)
        self.turn_in_place_rad = float(turn_in_place_rad)
        self.target_index = 1

    def update(self, x: float, y: float, yaw: float) -> RouteVelocityOutput:
        if not all(isfinite(value) for value in (x, y, yaw)):
            raise ValueError("robot pose must be finite")
        if self.target_index >= len(self.waypoints):
            return self._complete()

        target = self.waypoints[self.target_index]
        dx, dy = target.x_m - x, target.y_m - y
        distance = hypot(dx, dy)
        reached = distance <= self.reach_radius_m
        if reached:
            self.target_index += 1
            if self.target_index >= len(self.waypoints):
                return RouteVelocityOutput(0.0, 0.0, 0.0, self.target_index, target.name, True, True)
            target = self.waypoints[self.target_index]
            dx, dy = target.x_m - x, target.y_m - y
            distance = hypot(dx, dy)

        yaw_error = wrap_angle(atan2(dy, dx) - yaw)
        yaw_rate = max(-self.max_yaw_rate_rps, min(self.max_yaw_rate_rps, self.yaw_gain * yaw_error))
        alignment = max(0.0, cos(yaw_error))
        forward = min(self.max_speed_mps, 0.6 * distance) * alignment
        if abs(yaw_error) > self.turn_in_place_rad:
            forward = 0.0
        return RouteVelocityOutput(
            forward_mps=forward,
            lateral_mps=0.0,
            yaw_rate_rps=yaw_rate,
            target_index=self.target_index,
            target_name=target.name,
            reached=reached,
            mission_complete=False,
        )

    def reset(self) -> None:
        self.target_index = 1

    def _complete(self) -> RouteVelocityOutput:
        return RouteVelocityOutput(0.0, 0.0, 0.0, self.target_index, "mission_complete", False, True)


def load_route_file(path: str | Path) -> tuple[MissionWaypoint, ...]:
    """Load a route JSON file without accepting silently misspelled fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("waypoints"), list):
        raise ValueError("route JSON must be an object containing a waypoints list")
    allowed = {"name", "x_m", "y_m", "dwell_s", "task"}
    result: list[MissionWaypoint] = []
    for index, raw in enumerate(payload["waypoints"]):
        if not isinstance(raw, dict):
            raise ValueError(f"waypoint {index} must be an object")
        unexpected = set(raw) - allowed
        missing = {"name", "x_m", "y_m"} - set(raw)
        if unexpected or missing:
            raise ValueError(f"waypoint {index} has unexpected={sorted(unexpected)} missing={sorted(missing)}")
        result.append(
            MissionWaypoint(
                name=str(raw["name"]),
                x_m=float(raw["x_m"]),
                y_m=float(raw["y_m"]),
                dwell_s=float(raw.get("dwell_s", 0.0)),
                task=str(raw.get("task", "pass_through")),
            )
        )
    if len(result) < 2:
        raise ValueError("a route file needs at least two waypoints")
    return tuple(result)
