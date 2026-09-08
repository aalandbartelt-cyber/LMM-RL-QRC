"""Evidence-based promotion gates for locomotion experiments.

Training reward is not a promotion criterion.  A configuration progresses only
when independent evaluation episodes across several seeds meet the declared
task, robustness and safety thresholds.
"""

from dataclasses import dataclass
from statistics import mean
from typing import Mapping, Sequence

from .metrics import EpisodeMetrics
from .route_evaluator import EvidenceMode


@dataclass(frozen=True)
class PerformanceThresholds:
    """Initial gates for low-speed campus locomotion, not universal claims."""

    minimum_episodes_per_seed: int = 10
    minimum_distinct_seeds: int = 3
    minimum_success_rate: float = 0.90
    maximum_mean_velocity_error_mps: float = 0.20
    maximum_falls_per_episode: float = 0.02
    maximum_collisions_per_episode: float = 0.02
    maximum_mean_safety_stops_per_episode: float = 0.10
    maximum_tilt_rad: float = 0.70

    def __post_init__(self) -> None:
        if self.minimum_episodes_per_seed < 1 or self.minimum_distinct_seeds < 1:
            raise ValueError("episode and seed requirements must be positive")
        for name in (
            "minimum_success_rate",
            "maximum_mean_velocity_error_mps",
            "maximum_falls_per_episode",
            "maximum_collisions_per_episode",
            "maximum_mean_safety_stops_per_episode",
            "maximum_tilt_rad",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class PerformanceDecision:
    promoted: bool
    evidence_mode: EvidenceMode
    episodes: int
    seeds: int
    summary: dict[str, float]
    reasons: tuple[str, ...]


class PerformanceGate:
    """Evaluate a candidate against a baseline-independent quality gate."""

    def __init__(self, thresholds: PerformanceThresholds | None = None) -> None:
        self.thresholds = thresholds or PerformanceThresholds()

    def assess(
        self,
        results_by_seed: Mapping[int, Sequence[EpisodeMetrics]],
        evidence_mode: EvidenceMode,
    ) -> PerformanceDecision:
        reasons: list[str] = []
        if evidence_mode == "kinematic_preview":
            reasons.append("运动学预览不能通过 RL 性能晋级门槛。")
        usable = {seed: tuple(results) for seed, results in results_by_seed.items() if results}
        if len(usable) < self.thresholds.minimum_distinct_seeds:
            reasons.append(
                f"随机种子不足：需要 {self.thresholds.minimum_distinct_seeds} 个，当前 {len(usable)} 个。"
            )
        for seed, episodes in usable.items():
            if len(episodes) < self.thresholds.minimum_episodes_per_seed:
                reasons.append(
                    f"seed={seed} 的评估 episode 不足：需要 {self.thresholds.minimum_episodes_per_seed}，当前 {len(episodes)}。"
                )
        all_episodes = tuple(metric for episodes in usable.values() for metric in episodes)
        if not all_episodes:
            reasons.append("没有可评估 episode。")
            return PerformanceDecision(False, evidence_mode, 0, len(usable), {}, tuple(reasons))
        summary = {
            "success_rate": sum(metric.success for metric in all_episodes) / len(all_episodes),
            "mean_velocity_error_mps": mean(metric.velocity_error_mps for metric in all_episodes),
            "falls_per_episode": sum(metric.falls for metric in all_episodes) / len(all_episodes),
            "collisions_per_episode": sum(metric.collisions for metric in all_episodes) / len(all_episodes),
            "safety_stops_per_episode": sum(metric.safety_stops for metric in all_episodes) / len(all_episodes),
            "max_tilt_rad": max(metric.max_tilt_rad for metric in all_episodes),
        }
        checks = (
            ("success_rate", summary["success_rate"], self.thresholds.minimum_success_rate, ">="),
            (
                "mean_velocity_error_mps",
                summary["mean_velocity_error_mps"],
                self.thresholds.maximum_mean_velocity_error_mps,
                "<=",
            ),
            ("falls_per_episode", summary["falls_per_episode"], self.thresholds.maximum_falls_per_episode, "<="),
            (
                "collisions_per_episode",
                summary["collisions_per_episode"],
                self.thresholds.maximum_collisions_per_episode,
                "<=",
            ),
            (
                "safety_stops_per_episode",
                summary["safety_stops_per_episode"],
                self.thresholds.maximum_mean_safety_stops_per_episode,
                "<=",
            ),
            ("max_tilt_rad", summary["max_tilt_rad"], self.thresholds.maximum_tilt_rad, "<="),
        )
        for label, value, limit, operator in checks:
            passed = value >= limit if operator == ">=" else value <= limit
            if not passed:
                reasons.append(f"{label}={value:.4f} 未满足 {operator} {limit:.4f}。")
        return PerformanceDecision(
            promoted=not reasons,
            evidence_mode=evidence_mode,
            episodes=len(all_episodes),
            seeds=len(usable),
            summary=summary,
            reasons=tuple(reasons),
        )
