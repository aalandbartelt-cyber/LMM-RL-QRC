"""Safety-oriented velocity command handling."""

from dataclasses import dataclass

from .types import VelocityCommand


@dataclass(frozen=True)
class CommandLimiter:
    """Clamp high-level commands before they enter the policy observation."""

    max_vx: float = 0.8
    max_vy: float = 0.4
    max_yaw_rate: float = 1.2

    def __post_init__(self) -> None:
        if min(self.max_vx, self.max_vy, self.max_yaw_rate) <= 0:
            raise ValueError("command limits must be positive")

    def apply(self, command: VelocityCommand) -> VelocityCommand:
        return VelocityCommand(
            vx=max(-self.max_vx, min(self.max_vx, float(command.vx))),
            vy=max(-self.max_vy, min(self.max_vy, float(command.vy))),
            yaw_rate=max(-self.max_yaw_rate, min(self.max_yaw_rate, float(command.yaw_rate))),
        )
