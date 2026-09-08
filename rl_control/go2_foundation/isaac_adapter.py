"""Small, dependency-free adapter for Isaac Lab-style batched state values.

Isaac Lab commonly exposes tensors with a leading environment dimension.  This
module deliberately does not import Isaac Lab: it extracts one environment's
values and hands them to the stable :class:`RobotState` contract.  The actual
task implementation can therefore remain in the user's simulator project.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .types import ACTION_DIM, RobotState, VelocityCommand


def _materialize(value: Any) -> Any:
    """Convert torch-like values to Python lists without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _is_scalar(value: Any) -> bool:
    return not isinstance(value, (list, tuple))


def _vector(value: Any, size: int, env_index: int, name: str) -> tuple[float, ...]:
    value = _materialize(value)
    if not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence or tensor")

    # Accept either an unbatched [size] vector or a batched [num_envs, size].
    if len(value) == size and all(_is_scalar(v) for v in value):
        selected = value
    else:
        if env_index < 0 or env_index >= len(value):
            raise IndexError(f"env_index {env_index} is out of range for {name}")
        selected = _materialize(value[env_index])
    if not isinstance(selected, Sequence) or len(selected) != size:
        raise ValueError(f"{name} for env {env_index} must have length {size}")
    return tuple(float(v) for v in selected)


class IsaacLabStateAdapter:
    """Extract one Go2 environment from a raw mapping returned by a task."""

    _ALIASES = {
        "base_ang_vel": ("base_ang_vel", "base_ang_vel_b", "root_ang_vel_b"),
        "projected_gravity": ("projected_gravity", "projected_gravity_b"),
        # `joint_pos_rel` is the policy observation supplied by Unitree RL Lab.
        # Put it first so callers passing a complete Isaac Lab observation map
        # do not accidentally apply a default-angle subtraction twice.
        "dof_pos": ("joint_pos_rel", "dof_pos", "joint_pos"),
        "dof_vel": ("dof_vel", "joint_vel"),
        "previous_action": ("previous_action", "last_action", "actions_prev"),
        "command": ("command", "commands"),
    }

    def __init__(self, env_index: int = 0) -> None:
        if env_index < 0:
            raise ValueError("env_index must be non-negative")
        self.env_index = env_index

    def from_mapping(self, raw: Mapping[str, Any]) -> RobotState:
        values = {key: self._find(raw, key) for key in self._ALIASES}
        command_values = _vector(values["command"], 3, self.env_index, "command")
        state = RobotState(
            base_ang_vel=_vector(values["base_ang_vel"], 3, self.env_index, "base_ang_vel"),
            projected_gravity=_vector(values["projected_gravity"], 3, self.env_index, "projected_gravity"),
            dof_pos=_vector(values["dof_pos"], ACTION_DIM, self.env_index, "dof_pos"),
            dof_vel=_vector(values["dof_vel"], ACTION_DIM, self.env_index, "dof_vel"),
            previous_action=_vector(values["previous_action"], ACTION_DIM, self.env_index, "previous_action"),
            command=VelocityCommand(*command_values),
        )
        state.validate()
        return state

    def _find(self, raw: Mapping[str, Any], key: str) -> Any:
        for candidate in self._ALIASES[key]:
            if candidate in raw:
                return raw[candidate]
        aliases = ", ".join(self._ALIASES[key])
        raise KeyError(f"raw Isaac state is missing {key}; accepted names: {aliases}")
