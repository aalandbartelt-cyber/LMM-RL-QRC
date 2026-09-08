#!/usr/bin/env python3
"""Evaluate an exported Go2 JIT policy on the collision-enabled campus route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--policy", required=True, type=Path, help="exported policy.pt from the pinned training stack")
parser.add_argument("--route", type=Path, default=SCRIPT_DIR / "physical_route.json")
parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs" / "physical_route")
parser.add_argument("--max-seconds", type=float, default=240.0)
parser.add_argument("--no-progress-seconds", type=float, default=8.0)
parser.add_argument("--max-speed", type=float, default=0.55)
parser.add_argument("--disable-fabric", action="store_true")
parser.add_argument("--video", action="store_true", help="record an RGB video under the output directory")
parser.add_argument("--video-seconds", type=float, default=60.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True
app = AppLauncher(args_cli).app


def yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import gymnasium as gym
    import torch
    import unitree_rl_lab.tasks  # noqa: F401
    import campus_rl  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    from go2_foundation import (
        CampusMissionPipeline,
        CommandLimiter,
        PatrolMissionScheduler,
        RouteProgressEvaluator,
        RouteSample,
        RouteVelocityController,
        RouteWaypoint,
        load_route_file,
    )

    policy_path = args_cli.policy.expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(policy_path)
    route = load_route_file(args_cli.route)
    controller = RouteVelocityController(
        route,
        max_speed_mps=args_cli.max_speed,
        max_yaw_rate_rps=0.8,
        reach_radius_m=0.8,
        yaw_gain=1.4,
        turn_in_place_rad=0.95,
    )
    pipeline = CampusMissionPipeline(
        controller,
        PatrolMissionScheduler.from_waypoints(route),
        CommandLimiter(max_vx=args_cli.max_speed, max_vy=0.2, max_yaw_rate=0.8),
    )
    evaluator = RouteProgressEvaluator(
        tuple(RouteWaypoint(point.name, point.x_m, point.y_m) for point in route),
        reach_radius_m=0.8,
    )

    env_cfg = parse_env_cfg(
        "Unitree-Go2-Campus-Route-Eval",
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    env = gym.make(
        "Unitree-Go2-Campus-Route-Eval",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    dt = float(env.unwrapped.step_dt)
    max_steps = int(args_cli.max_seconds / dt)
    if args_cli.video:
        video_steps = min(max_steps, max(1, int(args_cli.video_seconds / dt)))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args_cli.output_dir / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=video_steps,
            disable_logger=True,
        )
    policy = torch.jit.load(str(policy_path), map_location=args_cli.device).eval()
    observations, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    no_progress_steps = max(1, int(args_cli.no_progress_seconds / dt))
    trajectory: list[dict[str, object]] = []
    last_progress_position = tuple(float(value) for value in robot.data.root_pos_w[0, :2])
    last_progress_step = 0
    stalled = False
    fallen = False

    try:
        for step in range(max_steps):
            if not app.is_running():
                break
            time_s = step * dt
            position = robot.data.root_pos_w[0]
            quaternion = robot.data.root_quat_w[0]
            x, y, z = (float(value) for value in position)
            yaw = yaw_from_wxyz(quaternion)
            mission = pipeline.step(time_s=time_s, x=x, y=y, yaw=yaw)
            command = torch.tensor(
                [[mission.command.vx, mission.command.vy, mission.command.yaw_rate]],
                dtype=torch.float32,
                device=args_cli.device,
            )
            command_term.vel_command_b[:] = command
            observations = env.unwrapped.observation_manager.compute()
            with torch.inference_mode():
                actions = policy(observations["policy"])
                observations, rewards, terminated, truncated, extras = env.step(actions)

            evaluator.update(RouteSample(time_s, x, y, fallen=bool(terminated[0])))
            trajectory.append(
                {
                    "time_s": round(time_s, 4),
                    "x_m": x,
                    "y_m": y,
                    "base_height_m": z,
                    "yaw_rad": yaw,
                    "target_index": mission.target_index,
                    "phase": mission.phase,
                    "vx_command": mission.command.vx,
                    "yaw_command": mission.command.yaw_rate,
                    "reward": float(rewards[0]),
                }
            )
            if math.hypot(x - last_progress_position[0], y - last_progress_position[1]) > 0.10:
                last_progress_position = (x, y)
                last_progress_step = step
            stalled = step - last_progress_step >= no_progress_steps
            fallen = bool(terminated[0]) or z < 0.18
            if mission.mission_complete or stalled or fallen:
                break
    finally:
        env.close()

    result = evaluator.evaluate("rl_physics").to_dict()
    result.update(
        {
            "status": "PASS" if result["route_complete"] and not stalled and not fallen else "FAIL",
            "policy": str(policy_path),
            "policy_sha256": sha256_file(policy_path),
            "route": str(args_cli.route.resolve()),
            "stalled": stalled,
            "fallen": fallen,
            "samples": len(trajectory),
        }
    )
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    (args_cli.output_dir / "route_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if trajectory:
        with (args_cli.output_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(trajectory[0]))
            writer.writeheader()
            writer.writerows(trajectory)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        app.close()
