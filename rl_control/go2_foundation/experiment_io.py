"""Portable JSON input/output for repeatable performance-gate decisions."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .metrics import EpisodeMetrics
from .performance import PerformanceDecision, PerformanceGate, PerformanceThresholds
from .route_evaluator import EvidenceMode


_EPISODE_FIELDS = set(EpisodeMetrics.__dataclass_fields__)


def load_results_by_seed(path: str | Path) -> tuple[EvidenceMode, dict[int, list[EpisodeMetrics]]]:
    """Load a documented evaluation artifact.

    Expected JSON shape:
    ``{"evidence_mode": "rl_physics", "results_by_seed": {"1": [{...}]}}``.
    Unknown episode fields are rejected so a typo cannot silently disappear.
    """
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation JSON must be an object")
    evidence_mode = payload.get("evidence_mode")
    if evidence_mode not in ("kinematic_preview", "rl_physics", "real_robot"):
        raise ValueError("evidence_mode must be kinematic_preview, rl_physics, or real_robot")
    raw_by_seed = payload.get("results_by_seed")
    if not isinstance(raw_by_seed, Mapping):
        raise ValueError("results_by_seed must be an object mapping seeds to episodes")
    results: dict[int, list[EpisodeMetrics]] = {}
    for seed, raw_episodes in raw_by_seed.items():
        if not isinstance(raw_episodes, list):
            raise ValueError(f"seed {seed} must contain an episode list")
        parsed: list[EpisodeMetrics] = []
        for item in raw_episodes:
            if not isinstance(item, Mapping):
                raise ValueError(f"seed {seed} contains a non-object episode")
            unexpected = set(item) - _EPISODE_FIELDS
            if unexpected:
                raise ValueError(f"seed {seed} contains unexpected episode fields: {sorted(unexpected)}")
            parsed.append(EpisodeMetrics(**item))
        results[int(seed)] = parsed
    return evidence_mode, results


def assess_file(
    path: str | Path,
    thresholds: PerformanceThresholds | None = None,
) -> PerformanceDecision:
    evidence_mode, results = load_results_by_seed(path)
    return PerformanceGate(thresholds).assess(results, evidence_mode)


def decision_to_json(decision: PerformanceDecision) -> str:
    return json.dumps(asdict(decision), ensure_ascii=False, indent=2, sort_keys=True)
