"""Route-progress evaluation with explicit evidence provenance.

The current campus scene can move the visual robot by setting its root pose.
That is useful for checking a route, but it is not evidence of learned
locomotion.  This evaluator carries the execution mode into each result so a
report cannot silently label a kinematic preview as an RL success.
"""

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Literal, Sequence


EvidenceMode = Literal["kinematic_preview", "rl_physics", "real_robot"]


@dataclass(frozen=True)
class RouteWaypoint:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class RouteSample:
    time_s: float
    x: float
    y: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    fallen: bool = False
    collision: bool = False
    safety_stop: bool = False


@dataclass(frozen=True)
class RouteEvaluation:
    evidence_mode: EvidenceMode
    route_complete: bool
    claimable_as_rl_result: bool
    waypoint_reached: int
    waypoint_total: int
    reached_names: tuple[str, ...]
    elapsed_s: float
    traveled_distance_m: float
    falls: int
    collisions: int
    safety_stops: int
    max_tilt_rad: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reached_names"] = list(self.reached_names)
        result["notes"] = list(self.notes)
        return result


class RouteProgressEvaluator:
    """Evaluate ordered waypoint completion from sampled robot poses."""

    def __init__(self, waypoints: Sequence[RouteWaypoint], reach_radius_m: float = 0.8) -> None:
        if not waypoints:
            raise ValueError("at least one waypoint is required")
        if reach_radius_m <= 0:
            raise ValueError("reach_radius_m must be positive")
        self.waypoints = tuple(waypoints)
        self.reach_radius_m = reach_radius_m
        self._samples: list[RouteSample] = []
        self._next_index = 0
        self._reached_names: list[str] = []

    @classmethod
    def from_waypoint_objects(cls, waypoints: Sequence[object], reach_radius_m: float = 0.8) -> "RouteProgressEvaluator":
        """Adapt the existing campus `Waypoint` objects without importing them."""
        return cls(
            tuple(RouteWaypoint(str(item.name), float(item.x), float(item.y)) for item in waypoints),
            reach_radius_m,
        )

    def update(self, sample: RouteSample) -> None:
        if self._samples and sample.time_s < self._samples[-1].time_s:
            raise ValueError("samples must arrive in non-decreasing time order")
        self._samples.append(sample)
        # One update can only advance one route step.  This preserves ordering
        # and exposes skipped waypoints instead of granting accidental credit.
        if self._next_index < len(self.waypoints):
            target = self.waypoints[self._next_index]
            if hypot(sample.x - target.x, sample.y - target.y) <= self.reach_radius_m:
                self._reached_names.append(target.name)
                self._next_index += 1

    def evaluate(self, evidence_mode: EvidenceMode) -> RouteEvaluation:
        if evidence_mode not in ("kinematic_preview", "rl_physics", "real_robot"):
            raise ValueError(f"unsupported evidence_mode: {evidence_mode}")
        if not self._samples:
            raise ValueError("at least one sample is required before evaluation")
        elapsed = self._samples[-1].time_s - self._samples[0].time_s
        traveled = sum(
            hypot(current.x - previous.x, current.y - previous.y)
            for previous, current in zip(self._samples[:-1], self._samples[1:])
        )
        max_tilt = max(max(abs(sample.roll_rad), abs(sample.pitch_rad)) for sample in self._samples)
        notes: list[str] = []
        if evidence_mode == "kinematic_preview":
            notes.append("运动学预览仅验证路线/任务流程，不能作为 RL 行走证据。")
        elif evidence_mode == "rl_physics":
            notes.append("RL 物理仿真结果仍需保留策略版本、配置、日志和视频。")
        else:
            notes.append("真机结果需额外保留人工接管、急停和硬件状态证据。")
        if self._next_index < len(self.waypoints):
            notes.append(f"未完成：下一目标为 {self.waypoints[self._next_index].name}。")
        return RouteEvaluation(
            evidence_mode=evidence_mode,
            route_complete=self._next_index == len(self.waypoints),
            claimable_as_rl_result=evidence_mode in ("rl_physics", "real_robot"),
            waypoint_reached=self._next_index,
            waypoint_total=len(self.waypoints),
            reached_names=tuple(self._reached_names),
            elapsed_s=elapsed,
            traveled_distance_m=traveled,
            falls=sum(sample.fallen for sample in self._samples),
            collisions=sum(sample.collision for sample in self._samples),
            safety_stops=sum(sample.safety_stop for sample in self._samples),
            max_tilt_rad=max_tilt,
            notes=tuple(notes),
        )
