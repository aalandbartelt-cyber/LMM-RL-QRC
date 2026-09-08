"""Construction of the fixed 45-dimensional Go2 policy observation."""

from typing import Tuple

from .types import ACTION_DIM, OBSERVATION_DIM, Go2ObservationConfig, RobotState


class ObservationBuilder:
    """Build and validate an observation using a policy-specific config."""

    def __init__(self, config: Go2ObservationConfig | None = None) -> None:
        self.config = config or Go2ObservationConfig()

    def build(self, state: RobotState) -> Tuple[float, ...]:
        state.validate()
        cfg = self.config
        command = state.command.as_tuple()
        obs = (
            tuple(float(v) * cfg.ang_vel_scale for v in state.base_ang_vel)
            + tuple(float(v) for v in state.projected_gravity)
            + tuple(command[i] * cfg.command_scale[i] for i in range(3))
            + tuple(
                (float(state.dof_pos[i]) - cfg.default_joint_angles[i]) * cfg.dof_pos_scale
                for i in range(ACTION_DIM)
            )
            + tuple(float(v) * cfg.dof_vel_scale for v in state.dof_vel)
            + tuple(float(v) for v in state.previous_action)
        )
        if len(obs) != OBSERVATION_DIM:
            raise AssertionError(f"internal observation layout error: {len(obs)}")
        return obs

    @staticmethod
    def index_ranges() -> dict[str, tuple[int, int]]:
        """Return half-open index ranges useful for debugging and logging."""
        return {
            "base_ang_vel": (0, 3),
            "projected_gravity": (3, 6),
            "command": (6, 9),
            "dof_pos": (9, 21),
            "dof_vel": (21, 33),
            "previous_action": (33, 45),
        }
