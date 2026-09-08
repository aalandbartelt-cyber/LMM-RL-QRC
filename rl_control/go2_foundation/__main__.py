"""Command-line entry point for portable Go2 campus-RL checks."""

import argparse
import json

from .experiment_io import assess_file, decision_to_json
from .observation import ObservationBuilder
from .policy import PolicyRunner, ZeroPolicy
from .safety import SafetySupervisor
from .types import ACTION_DIM, Go2ObservationConfig, RobotState, VelocityCommand


def _smoke() -> dict[str, object]:
    state = RobotState(
        base_ang_vel=(0.0, 0.0, 0.0),
        projected_gravity=(0.0, 0.0, -1.0),
        dof_pos=Go2ObservationConfig.mujoco().default_joint_angles,
        dof_vel=(0.0,) * ACTION_DIM,
        previous_action=(0.0,) * ACTION_DIM,
        command=VelocityCommand(0.2, 0.0, 0.1),
    )
    observation = ObservationBuilder().build(state)
    action = PolicyRunner(ZeroPolicy()).infer(observation)
    decision = SafetySupervisor().evaluate(action, observation_age_s=0.0, roll_rad=0.0, pitch_rad=0.0)
    return {"observation_dim": len(observation), "action_dim": len(action), "safety": decision.reason}


def _profiles() -> dict[str, object]:
    return {
        "mujoco": Go2ObservationConfig.mujoco().__dict__,
        "real_deployment": Go2ObservationConfig.real_deployment().__dict__,
        "unitree_rl_lab_relative_joint": Go2ObservationConfig.unitree_rl_lab_relative_joint().__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Go2 campus-RL portable project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("smoke", help="run dependency-free control-pipeline smoke test")
    subparsers.add_parser("profiles", help="print explicit policy observation profiles")
    assess_parser = subparsers.add_parser("assess", help="assess an RL evaluation JSON file")
    assess_parser.add_argument("evaluation_json")
    args = parser.parse_args()
    if args.command == "smoke":
        print(json.dumps(_smoke(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "profiles":
        print(json.dumps(_profiles(), ensure_ascii=False, indent=2))
        return 0
    decision = assess_file(args.evaluation_json)
    print(decision_to_json(decision))
    return 0 if decision.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
