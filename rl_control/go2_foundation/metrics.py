"""Small, JSON-friendly episode metrics for reproducible evaluation."""

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Dict, List


@dataclass
class EpisodeMetrics:
    episode_id: str
    success: bool
    completion_time_s: float = 0.0
    waypoint_reached: int = 0
    waypoint_total: int = 0
    falls: int = 0
    collisions: int = 0
    safety_stops: int = 0
    velocity_error_mps: float = 0.0
    max_tilt_rad: float = 0.0
    path_error_m: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MetricsAccumulator:
    episodes: List[EpisodeMetrics] = field(default_factory=list)

    def add(self, metrics: EpisodeMetrics) -> None:
        self.episodes.append(metrics)

    def summary(self) -> Dict[str, object]:
        if not self.episodes:
            return {"episodes": 0, "success_rate": 0.0}
        n = len(self.episodes)
        return {
            "episodes": n,
            "success_rate": sum(m.success for m in self.episodes) / n,
            "mean_completion_time_s": mean(m.completion_time_s for m in self.episodes),
            "mean_waypoint_reached": mean(m.waypoint_reached for m in self.episodes),
            "mean_velocity_error_mps": mean(m.velocity_error_mps for m in self.episodes),
            "max_tilt_rad": max(m.max_tilt_rad for m in self.episodes),
            "total_falls": sum(m.falls for m in self.episodes),
            "total_collisions": sum(m.collisions for m in self.episodes),
            "total_safety_stops": sum(m.safety_stops for m in self.episodes),
        }
