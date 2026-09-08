"""Mission logic shared by the scene preview and the real locomotion policy.

The preview runner uses :class:`KinematicPatrol` to animate the robot model.
An existing Go2 low-level velocity policy can use :class:`WaypointVelocityController`
to obtain forward and yaw-rate commands from the same route.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    dwell_s: float = 2.0
    inspection: str = "visual_check"


# 60 m x 40 m representative campus. The first and last points are the gate.
SECURITY_WAYPOINTS: tuple[Waypoint, ...] = (
    Waypoint("P0_主入口起点", 0.0, -16.0, 1.0, "mission_start"),
    Waypoint("P1_办公楼门禁", -16.0, -10.0, 2.5, "door_and_access_control"),
    Waypoint("P2_西侧围墙盲区", -24.0, 5.0, 3.0, "perimeter_intrusion"),
    Waypoint("P3_北侧电子围栏", -7.0, 15.0, 2.5, "fence_integrity"),
    Waypoint("P4_仓库防火门", 15.0, 7.0, 3.0, "door_fire_smoke"),
    Waypoint("P5_动力机房", 21.0, -8.0, 3.0, "equipment_temperature"),
    Waypoint("P6_停车区", 13.0, -13.0, 2.0, "vehicle_and_object_check"),
    Waypoint("P7_主入口返航", 0.0, -16.0, 1.0, "mission_finish"),
)


EVENTS = {
    "normal": {
        "name": "无异常",
        "position": None,
        "expected_result": "all_clear",
    },
    "intrusion": {
        "name": "可疑人员侵入西侧围墙盲区",
        "position": (-24.0, 5.0),
        "expected_result": "person_intrusion",
    },
    "open_door": {
        "name": "仓库防火门异常开启",
        "position": (15.0, 7.0),
        "expected_result": "door_open",
    },
    "fire": {
        "name": "动力机房外发现烟火",
        "position": (21.0, -8.0),
        "expected_result": "smoke_or_fire",
    },
}


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def route_length(waypoints: tuple[Waypoint, ...] = SECURITY_WAYPOINTS) -> float:
    return sum(
        math.hypot(b.x - a.x, b.y - a.y)
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )


class KinematicPatrol:
    """Simple route animation used only for scene and mission acceptance tests."""

    def __init__(self, speed_mps: float = 3.0):
        if speed_mps <= 0:
            raise ValueError("speed_mps must be positive")
        self.speed_mps = speed_mps
        self.x = SECURITY_WAYPOINTS[0].x
        self.y = SECURITY_WAYPOINTS[0].y
        self.yaw = 0.0
        self.target_index = 1
        self.dwell_remaining = SECURITY_WAYPOINTS[0].dwell_s
        self.complete = False

    def advance(self, dt: float) -> dict:
        arrived: Waypoint | None = None
        phase = "moving"

        if self.complete:
            return self._state("complete", None)

        if self.dwell_remaining > 0.0:
            self.dwell_remaining = max(0.0, self.dwell_remaining - dt)
            phase = "inspecting"
            if self.dwell_remaining > 0.0:
                return self._state(phase, None)
            if self.target_index >= len(SECURITY_WAYPOINTS):
                self.complete = True
                return self._state("complete", None)

        target = SECURITY_WAYPOINTS[self.target_index]
        dx = target.x - self.x
        dy = target.y - self.y
        distance = math.hypot(dx, dy)
        if distance > 1e-6:
            self.yaw = math.atan2(dy, dx)

        travel = self.speed_mps * dt
        if distance <= max(travel, 0.05):
            self.x, self.y = target.x, target.y
            arrived = target
            self.target_index += 1
            self.dwell_remaining = target.dwell_s
            phase = "inspecting"
        else:
            self.x += travel * dx / distance
            self.y += travel * dy / distance

        return self._state(phase, arrived)

    def _state(self, phase: str, arrived: Waypoint | None) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "yaw": self.yaw,
            "phase": phase,
            "arrived": asdict(arrived) if arrived else None,
            "complete": self.complete,
            "target_index": self.target_index,
        }


@dataclass(frozen=True)
class VelocityCommand:
    forward_mps: float
    lateral_mps: float
    yaw_rate_rps: float
    target_index: int
    target_name: str
    reached: bool
    mission_complete: bool


class WaypointVelocityController:
    """High-level command generator for an existing Go2 velocity policy."""

    def __init__(
        self,
        max_speed_mps: float = 0.8,
        max_yaw_rate_rps: float = 1.0,
        reach_radius_m: float = 0.8,
    ):
        self.max_speed_mps = max_speed_mps
        self.max_yaw_rate_rps = max_yaw_rate_rps
        self.reach_radius_m = reach_radius_m
        self.target_index = 1

    def update(self, x: float, y: float, yaw: float) -> VelocityCommand:
        if self.target_index >= len(SECURITY_WAYPOINTS):
            return VelocityCommand(0.0, 0.0, 0.0, self.target_index, "任务完成", False, True)

        target = SECURITY_WAYPOINTS[self.target_index]
        dx, dy = target.x - x, target.y - y
        distance = math.hypot(dx, dy)
        reached = distance <= self.reach_radius_m
        if reached:
            self.target_index += 1
            if self.target_index >= len(SECURITY_WAYPOINTS):
                return VelocityCommand(0.0, 0.0, 0.0, self.target_index, target.name, True, True)
            target = SECURITY_WAYPOINTS[self.target_index]
            dx, dy = target.x - x, target.y - y
            distance = math.hypot(dx, dy)

        desired_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(desired_yaw - yaw)
        yaw_rate = max(-self.max_yaw_rate_rps, min(self.max_yaw_rate_rps, 1.5 * yaw_error))

        # Reduce forward motion while the robot is facing away from the target.
        alignment = max(0.0, math.cos(yaw_error))
        forward = min(self.max_speed_mps, 0.6 * distance) * alignment
        if abs(yaw_error) > 1.1:
            forward = 0.0

        return VelocityCommand(
            forward_mps=forward,
            lateral_mps=0.0,
            yaw_rate_rps=yaw_rate,
            target_index=self.target_index,
            target_name=target.name,
            reached=reached,
            mission_complete=False,
        )
