"""Policy-side safety gate.

This is a pre-actuation decision layer.  A later robot adapter must still have
an independent hardware emergency stop and communication watchdog.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence, Tuple

from .types import ACTION_DIM


@dataclass(frozen=True)
class SafetyDecision:
    action: Tuple[float, ...]
    allow_motion: bool
    reason: str
    clipped: bool = False


@dataclass(frozen=True)
class SafetySupervisor:
    max_action_abs: float = 1.0
    max_tilt_rad: float = 0.9
    max_observation_age_s: float = 0.2

    def __post_init__(self) -> None:
        if self.max_action_abs <= 0 or self.max_tilt_rad <= 0 or self.max_observation_age_s <= 0:
            raise ValueError("safety thresholds must be positive")

    def evaluate(
        self,
        action: Sequence[float],
        *,
        observation_age_s: float,
        roll_rad: float,
        pitch_rad: float,
        fallen: bool = False,
        emergency_stop: bool = False,
        communication_ok: bool = True,
    ) -> SafetyDecision:
        if len(action) != ACTION_DIM:
            return self._stop("invalid_action_dimension")
        raw = tuple(float(v) for v in action)
        if not all(isfinite(v) for v in raw):
            return self._stop("nonfinite_action")
        if emergency_stop:
            return self._stop("emergency_stop")
        if not communication_ok:
            return self._stop("communication_lost")
        if fallen:
            return self._stop("fall_detected")
        if not isfinite(observation_age_s) or observation_age_s > self.max_observation_age_s:
            return self._stop("stale_observation")
        if not isfinite(roll_rad) or not isfinite(pitch_rad):
            return self._stop("nonfinite_attitude")
        if max(abs(float(roll_rad)), abs(float(pitch_rad))) > self.max_tilt_rad:
            return self._stop("tilt_limit")
        clipped_action = tuple(max(-self.max_action_abs, min(self.max_action_abs, v)) for v in raw)
        clipped = clipped_action != raw
        return SafetyDecision(clipped_action, True, "ok", clipped)

    @staticmethod
    def _stop(reason: str) -> SafetyDecision:
        return SafetyDecision((0.0,) * ACTION_DIM, False, reason, False)
