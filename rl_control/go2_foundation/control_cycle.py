"""One deterministic policy-control cycle shared by simulation and deployment."""

from dataclasses import dataclass
from typing import Sequence, Tuple

from .observation import ObservationBuilder
from .policy import PolicyRunner
from .safety import SafetyDecision, SafetySupervisor
from .types import RobotState


@dataclass(frozen=True)
class ControlOutput:
    observation: Tuple[float, ...]
    policy_action: Tuple[float, ...]
    safety: SafetyDecision


class Go2ControlCycle:
    """Compose observation, policy inference and the safety gate.

    The caller remains responsible for sending ``safety.action`` to a simulator
    or a separate hardware adapter.  This class intentionally has no side
    effects and is therefore safe to exercise in unit tests.
    """

    def __init__(
        self,
        policy_runner: PolicyRunner,
        observation_builder: ObservationBuilder | None = None,
        safety: SafetySupervisor | None = None,
    ) -> None:
        self.policy_runner = policy_runner
        self.observation_builder = observation_builder or ObservationBuilder()
        self.safety = safety or SafetySupervisor()

    def step(
        self,
        state: RobotState,
        *,
        observation_age_s: float,
        roll_rad: float,
        pitch_rad: float,
        fallen: bool = False,
        emergency_stop: bool = False,
        communication_ok: bool = True,
    ) -> ControlOutput:
        observation = self.observation_builder.build(state)
        policy_action = self.policy_runner.infer(observation)
        decision = self.safety.evaluate(
            policy_action,
            observation_age_s=observation_age_s,
            roll_rad=roll_rad,
            pitch_rad=pitch_rad,
            fallen=fallen,
            emergency_stop=emergency_stop,
            communication_ok=communication_ok,
        )
        return ControlOutput(observation, policy_action, decision)
