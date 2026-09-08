"""Shared data contracts and Go2 policy constants.

The constants mirror the audited upstream Go2 policy interface: 45 observation
values and 12 actions.  Keep these values in one place so a simulator and a
deployment adapter cannot silently drift apart.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Sequence, Tuple

OBSERVATION_DIM = 45
ACTION_DIM = 12

# Order used by the audited Go2 policy configuration.
OBSERVATION_LAYOUT = (
    "base_ang_vel[3]",
    "projected_gravity[3]",
    "command[vx,vy,yaw_rate][3]",
    "dof_pos_minus_default[12]",
    "dof_vel[12]",
    "previous_action[12]",
)

# The upstream real-deployment configuration uses this motor ordering.
JOINT2MOTOR_INDEX: Tuple[int, ...] = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)

DEFAULT_JOINT_ANGLES: Tuple[float, ...] = (
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    1.0,
    -1.5,
    -0.1,
    1.0,
    -1.5,
)


def _validate_vector(name: str, value: Sequence[float], size: int) -> Tuple[float, ...]:
    if len(value) != size:
        raise ValueError(f"{name} must have length {size}, got {len(value)}")
    result = tuple(float(v) for v in value)
    if not all(isfinite(v) for v in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class VelocityCommand:
    """Desired body velocity in m/s and rad/s before policy scaling."""

    vx: float = 0.0
    vy: float = 0.0
    yaw_rate: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return _validate_vector("velocity command", (self.vx, self.vy, self.yaw_rate), 3)


@dataclass(frozen=True)
class RobotState:
    """Policy-facing state, independent of Isaac Lab or a hardware SDK."""

    base_ang_vel: Sequence[float]
    projected_gravity: Sequence[float]
    dof_pos: Sequence[float]
    dof_vel: Sequence[float]
    previous_action: Sequence[float]
    command: VelocityCommand = VelocityCommand()

    def validate(self) -> None:
        _validate_vector("base_ang_vel", self.base_ang_vel, 3)
        _validate_vector("projected_gravity", self.projected_gravity, 3)
        _validate_vector("dof_pos", self.dof_pos, ACTION_DIM)
        _validate_vector("dof_vel", self.dof_vel, ACTION_DIM)
        _validate_vector("previous_action", self.previous_action, ACTION_DIM)
        self.command.as_tuple()


@dataclass(frozen=True)
class Go2ObservationConfig:
    """Scaling values from a specific policy export.

    The MuJoCo and real-deployment examples in the reference repository use
    different command scales.  They must be selected explicitly per policy.
    """

    ang_vel_scale: float = 0.25
    command_scale: Tuple[float, float, float] = (2.0, 2.0, 0.25)
    dof_pos_scale: float = 1.0
    dof_vel_scale: float = 0.05
    action_scale: float = 0.25
    default_joint_angles: Tuple[float, ...] = DEFAULT_JOINT_ANGLES

    @classmethod
    def mujoco(cls) -> "Go2ObservationConfig":
        """Configuration matching the audited MuJoCo deployment example."""
        return cls(command_scale=(2.0, 2.0, 0.25))

    @classmethod
    def real_deployment(cls) -> "Go2ObservationConfig":
        """Configuration matching the audited real-deployment example.

        This only controls policy preprocessing; it does not enable hardware
        output or bypass the safety layer.
        """
        return cls(command_scale=(3.0, 2.0, 0.5))

    @classmethod
    def unitree_rl_lab_relative_joint(cls) -> "Go2ObservationConfig":
        """Match the current official Isaac Lab Go2 velocity observation.

        Unitree RL Lab's Go2 task uses angular-velocity scale ``0.2`` and
        unscaled velocity commands.  It exposes ``joint_pos_rel`` rather than
        absolute joint angles, so the offset must be all zero here.  This
        profile is only valid when the adapter is fed that relative-joint
        observation term; it must not be used with absolute joint positions.
        """
        return cls(
            ang_vel_scale=0.2,
            command_scale=(1.0, 1.0, 1.0),
            default_joint_angles=(0.0,) * ACTION_DIM,
        )

    def __post_init__(self) -> None:
        _validate_vector("command_scale", self.command_scale, 3)
        _validate_vector("default_joint_angles", self.default_joint_angles, ACTION_DIM)
        for name in ("ang_vel_scale", "dof_pos_scale", "dof_vel_scale", "action_scale"):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
