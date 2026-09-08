"""Bridge the existing campus mission controller to the policy contract.

The local controller currently exposes names such as ``forward_mps`` and
``yaw_rate_rps``.  This adapter accepts either that dictionary-shaped output or
an object with the same attributes, keeping the existing scene code untouched.
"""

from collections.abc import Mapping
from typing import Any

from .types import VelocityCommand


def command_from_mission_output(output: Mapping[str, Any] | object) -> VelocityCommand:
    """Convert a waypoint controller result to a validated velocity command."""

    def read(name: str, aliases: tuple[str, ...] = ()) -> float:
        names = (name,) + aliases
        for candidate in names:
            if isinstance(output, Mapping) and candidate in output:
                return float(output[candidate])
            if hasattr(output, candidate):
                return float(getattr(output, candidate))
        raise KeyError(f"mission output is missing '{name}'")

    return VelocityCommand(
        vx=read("forward_mps", ("vx",)),
        vy=read("lateral_mps", ("vy",)),
        yaw_rate=read("yaw_rate_rps", ("yaw_rate",)),
    )
