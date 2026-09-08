"""Run the dependency-free foundation pipeline without a simulator or robot."""

from go2_foundation import (
    CommandLimiter,
    ObservationBuilder,
    PolicyRunner,
    RobotState,
    SafetySupervisor,
    VelocityCommand,
    ZeroPolicy,
    command_from_mission_output,
)


def main() -> None:
    mission_result = {"forward_mps": 0.25, "lateral_mps": 0.0, "yaw_rate_rps": 0.1}
    command = CommandLimiter().apply(command_from_mission_output(mission_result))
    state = RobotState(
        base_ang_vel=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        dof_pos=(0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5),
        dof_vel=(0.0,) * 12,
        previous_action=(0.0,) * 12,
        command=command,
    )
    observation = ObservationBuilder().build(state)
    action = PolicyRunner(ZeroPolicy()).infer(observation)
    decision = SafetySupervisor().evaluate(
        action,
        observation_age_s=0.0,
        roll_rad=0.0,
        pitch_rad=0.0,
    )
    print({"observation_dim": len(observation), "action_dim": len(decision.action), "safety": decision.reason})


if __name__ == "__main__":
    main()
