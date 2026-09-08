"""Fail-closed checks for policy/export/runtime interface compatibility."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

from .types import ACTION_DIM, OBSERVATION_DIM, Go2ObservationConfig


@dataclass(frozen=True)
class PolicyInterfaceMetadata:
    policy_name: str
    profile: str
    observation_dim: int
    action_dim: int
    action_scale: float
    control_dt_s: float
    command_scale: tuple[float, float, float]
    ang_vel_scale: float


@dataclass(frozen=True)
class InterfaceAuditResult:
    compatible: bool
    metadata: PolicyInterfaceMetadata
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"compatible": self.compatible, "metadata": asdict(self.metadata), "reasons": list(self.reasons)}


_PROFILES = {
    "mujoco": Go2ObservationConfig.mujoco,
    "real_deployment": Go2ObservationConfig.real_deployment,
    "unitree_rl_lab_relative_joint": Go2ObservationConfig.unitree_rl_lab_relative_joint,
}


def audit_policy_interface(payload: Mapping[str, object]) -> InterfaceAuditResult:
    """Validate an exported-policy metadata record against a named profile.

    This deliberately validates only observable contracts, not the learned
    policy quality.  A compatible result means the policy can be tested safely;
    it does not mean that it walks well.
    """
    required = {
        "policy_name",
        "profile",
        "observation_dim",
        "action_dim",
        "action_scale",
        "control_dt_s",
        "command_scale",
        "ang_vel_scale",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"policy metadata missing fields: {sorted(missing)}")
    profile_name = str(payload["profile"])
    if profile_name not in _PROFILES:
        raise ValueError(f"unknown policy profile '{profile_name}'; choose from {sorted(_PROFILES)}")
    command_scale_raw = payload["command_scale"]
    if not isinstance(command_scale_raw, (list, tuple)) or len(command_scale_raw) != 3:
        raise ValueError("command_scale must contain exactly three values")
    metadata = PolicyInterfaceMetadata(
        policy_name=str(payload["policy_name"]),
        profile=profile_name,
        observation_dim=int(payload["observation_dim"]),
        action_dim=int(payload["action_dim"]),
        action_scale=float(payload["action_scale"]),
        control_dt_s=float(payload["control_dt_s"]),
        command_scale=tuple(float(value) for value in command_scale_raw),
        ang_vel_scale=float(payload["ang_vel_scale"]),
    )
    expected = _PROFILES[profile_name]()
    reasons: list[str] = []
    if not metadata.policy_name.strip():
        reasons.append("policy_name cannot be blank")
    if metadata.observation_dim != OBSERVATION_DIM:
        reasons.append(f"observation_dim={metadata.observation_dim}; expected {OBSERVATION_DIM}")
    if metadata.action_dim != ACTION_DIM:
        reasons.append(f"action_dim={metadata.action_dim}; expected {ACTION_DIM}")
    if not isfinite(metadata.action_scale) or metadata.action_scale <= 0:
        reasons.append("action_scale must be finite and positive")
    elif metadata.action_scale != expected.action_scale:
        reasons.append(f"action_scale={metadata.action_scale}; profile expects {expected.action_scale}")
    if not all(isfinite(value) and value > 0 for value in metadata.command_scale):
        reasons.append("command_scale values must be finite and positive")
    elif metadata.command_scale != expected.command_scale:
        reasons.append(f"command_scale={metadata.command_scale}; profile expects {expected.command_scale}")
    if not isfinite(metadata.ang_vel_scale) or metadata.ang_vel_scale <= 0:
        reasons.append("ang_vel_scale must be finite and positive")
    elif metadata.ang_vel_scale != expected.ang_vel_scale:
        reasons.append(f"ang_vel_scale={metadata.ang_vel_scale}; profile expects {expected.ang_vel_scale}")
    if not isfinite(metadata.control_dt_s) or metadata.control_dt_s <= 0:
        reasons.append("control_dt_s must be finite and positive")
    return InterfaceAuditResult(not reasons, metadata, tuple(reasons))
