"""Pure geometry checks for route segments and static campus obstacles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from .route_mission import MissionWaypoint


@dataclass(frozen=True)
class AxisAlignedObstacle:
    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not self.name.strip() or not all(isfinite(value) for value in values):
            raise ValueError("obstacle name and bounds must be valid")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("obstacle bounds must have positive area")


@dataclass(frozen=True)
class RouteClearanceIssue:
    segment_index: int
    start_name: str
    end_name: str
    obstacle_name: str


def _segment_intersects_box(
    start: MissionWaypoint,
    end: MissionWaypoint,
    obstacle: AxisAlignedObstacle,
    clearance_m: float,
) -> bool:
    x_min = obstacle.x_min - clearance_m
    x_max = obstacle.x_max + clearance_m
    y_min = obstacle.y_min - clearance_m
    y_max = obstacle.y_max + clearance_m
    dx = end.x_m - start.x_m
    dy = end.y_m - start.y_m
    lower = 0.0
    upper = 1.0

    for direction, distance in (
        (-dx, start.x_m - x_min),
        (dx, x_max - start.x_m),
        (-dy, start.y_m - y_min),
        (dy, y_max - start.y_m),
    ):
        if abs(direction) < 1e-12:
            if distance < 0.0:
                return False
            continue
        ratio = distance / direction
        if direction < 0.0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)
    return lower <= upper


def validate_route_clearance(
    waypoints: Iterable[MissionWaypoint],
    obstacles: Iterable[AxisAlignedObstacle],
    *,
    clearance_m: float,
) -> tuple[RouteClearanceIssue, ...]:
    """Return every route segment intersecting an expanded obstacle box."""
    if not isfinite(clearance_m) or clearance_m < 0.0:
        raise ValueError("clearance_m must be finite and non-negative")
    route = tuple(waypoints)
    obstacle_list = tuple(obstacles)
    issues: list[RouteClearanceIssue] = []
    for index, (start, end) in enumerate(zip(route[:-1], route[1:])):
        for obstacle in obstacle_list:
            if _segment_intersects_box(start, end, obstacle, clearance_m):
                issues.append(RouteClearanceIssue(index, start.name, end.name, obstacle.name))
    return tuple(issues)
