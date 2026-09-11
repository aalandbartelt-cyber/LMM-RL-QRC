#!/usr/bin/env python3
"""Batch benchmark a Go2 checkpoint on the repository's Isaac Gym terrains.

The script has two modes:

1. Set QRC_RUN_ALL=1 to launch one fresh process per terrain and aggregate
   the resulting JSON files into CSV and JSON summaries.
2. Set QRC_TERRAIN=<name> to evaluate one terrain directly.

Custom benchmark settings are supplied through QRC_* environment variables
so the script remains compatible with legged_gym's existing argument parser.
"""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TERRAIN_INDEX = {
    "wave": 0,
    "slope": 1,
    "rough_slope": 2,
    "stairs_up": 3,
    "stairs_down": 4,
    "obstacles": 5,
    "stepping_stones": 6,
    "gap": 7,
    "flat": 8,
}

DEFAULT_TERRAINS = (
    "flat",
    "slope",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "obstacles",
    "wave",
)


def output_dir() -> Path:
    return Path(
        os.environ.get(
            "QRC_OUTPUT_DIR",
            "/home/jovyan/go2_work/benchmark_model3501",
        )
    ).expanduser()


def selected_terrains():
    value = os.environ.get("QRC_TERRAINS", "")
    terrains = tuple(item.strip() for item in value.split(",") if item.strip())
    return terrains or DEFAULT_TERRAINS


def aggregate_results(directory: Path, terrains):
    results = []
    for terrain in terrains:
        result_path = directory / f"{terrain}.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing benchmark result: {result_path}")
        results.append(json.loads(result_path.read_text(encoding="utf-8")))

    fieldnames = list(results[0])
    csv_path = directory / "benchmark_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary_path = directory / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("\n===== GO2 TERRAIN BENCHMARK SUMMARY =====")
    print(
        f"{'terrain':<17} {'success':>9} {'falls':>7} "
        f"{'rmse':>9} {'tilt':>9} {'collision':>10}"
    )
    for result in results:
        print(
            f"{result['terrain']:<17} "
            f"{result['success_rate']:>8.2%} "
            f"{result['fall_count']:>7d} "
            f"{result['tracking_rmse_mps']:>9.4f} "
            f"{result['mean_tilt_rad']:>9.4f} "
            f"{result['collision_sample_rate']:>9.2%}"
        )
    print(f"\nCSV:  {csv_path}")
    print(f"JSON: {summary_path}")
    print("GO2 MULTI-TERRAIN BENCHMARK PASS")


def run_all():
    directory = output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    terrains = selected_terrains()

    for terrain in terrains:
        if terrain not in TERRAIN_INDEX:
            raise ValueError(f"Unknown terrain: {terrain}")
        result_path = directory / f"{terrain}.json"
        result_path.unlink(missing_ok=True)

    for terrain in terrains:
        print(f"\n===== BENCHMARK TERRAIN: {terrain} =====", flush=True)
        child_env = os.environ.copy()
        child_env.pop("QRC_RUN_ALL", None)
        child_env["QRC_TERRAIN"] = terrain
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            env=child_env,
            check=True,
        )

    aggregate_results(directory, terrains)


if os.environ.get("QRC_RUN_ALL") == "1":
    run_all()
    raise SystemExit(0)


# Isaac Gym must be imported before torch in this environment.
from isaacgym import gymapi, gymtorch  # noqa: E402,F401
import torch  # noqa: E402

from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402


def disable_randomization(env_cfg):
    env_cfg.noise.add_noise = False
    for name in (
        "randomize_friction",
        "randomize_base_mass",
        "randomize_link_mass",
        "randomize_base_com",
        "randomize_restitution",
        "randomize_pd_gains",
        "randomize_motor_zero_offset",
        "randomize_motor_strength",
        "randomize_action_delay",
        "push_robots",
    ):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def make_one_hot_proportions(terrain):
    proportions = [0.0] * len(TERRAIN_INDEX)
    proportions[TERRAIN_INDEX[terrain]] = 1.0
    return proportions


def evaluate(args):
    terrain = os.environ.get("QRC_TERRAIN", "flat")
    if terrain not in TERRAIN_INDEX:
        raise ValueError(f"Unknown QRC_TERRAIN: {terrain}")

    duration_seconds = float(os.environ.get("QRC_DURATION_SECONDS", "20"))
    command_vx = float(os.environ.get("QRC_COMMAND_VX", "0.5"))
    command_vy = float(os.environ.get("QRC_COMMAND_VY", "0.0"))
    command_yaw = float(os.environ.get("QRC_COMMAND_YAW", "0.0"))

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    env_cfg.env.num_envs = args.num_envs or 256
    env_cfg.env.test = True
    env_cfg.terrain.num_rows = 8
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.selected = False
    env_cfg.terrain.terrain_proportions = make_one_hot_proportions(terrain)
    env_cfg.commands.resampling_time = duration_seconds + 5.0
    disable_randomization(env_cfg)

    env, _ = task_registry.make_env(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
    )

    train_cfg.runner.resume = True
    runner, _ = task_registry.make_alg_runner(
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
    )
    policy = runner.get_inference_policy(device=env.device)

    num_envs = env.num_envs
    total_steps = max(1, round(duration_seconds / env.dt))
    total_samples = num_envs * total_steps
    device = env.device

    fall_counts = torch.zeros(num_envs, dtype=torch.long, device=device)
    tracking_sse = torch.zeros((), dtype=torch.float64, device=device)
    tracking_abs = torch.zeros((), dtype=torch.float64, device=device)
    reward_sum = torch.zeros((), dtype=torch.float64, device=device)
    power_sum = torch.zeros((), dtype=torch.float64, device=device)
    action_delta_sum = torch.zeros((), dtype=torch.float64, device=device)
    roll_abs_sum = torch.zeros((), dtype=torch.float64, device=device)
    pitch_abs_sum = torch.zeros((), dtype=torch.float64, device=device)
    tilt_sum = torch.zeros((), dtype=torch.float64, device=device)
    max_tilt = torch.zeros((), dtype=torch.float64, device=device)
    base_height_sum = torch.zeros((), dtype=torch.float64, device=device)
    min_base_height = torch.full(
        (),
        float("inf"),
        dtype=torch.float64,
        device=device,
    )
    collision_samples = torch.zeros((), dtype=torch.float64, device=device)

    start_time = time.perf_counter()
    with torch.inference_mode():
        for _ in range(total_steps):
            env.commands[:, 0] = command_vx
            env.commands[:, 1] = command_vy
            env.commands[:, 2] = command_yaw
            if env.commands.shape[1] > 3:
                env.commands[:, 3] = 0.0

            # Commands are observations, so refresh before policy inference.
            env.compute_observations()
            obs = env.get_observations()
            actions = policy(obs.detach())
            action_delta_sum += torch.abs(
                actions - env.last_actions
            ).sum(dtype=torch.float64)

            _, _, rewards, dones, _ = env.step(actions.detach())

            non_timeout_falls = dones.bool() & ~env.time_out_buf.bool()
            fall_counts += non_timeout_falls.long()

            tracking_error = env.base_lin_vel[:, 0] - command_vx
            tracking_sse += torch.square(tracking_error).sum(
                dtype=torch.float64
            )
            tracking_abs += torch.abs(tracking_error).sum(dtype=torch.float64)
            reward_sum += rewards.sum(dtype=torch.float64)
            power_sum += torch.abs(env.torques * env.dof_vel).sum(
                dtype=torch.float64
            )

            roll_abs = torch.abs(env.rpy[:, 0])
            pitch_abs = torch.abs(env.rpy[:, 1])
            tilt = torch.sqrt(torch.square(roll_abs) + torch.square(pitch_abs))
            roll_abs_sum += roll_abs.sum(dtype=torch.float64)
            pitch_abs_sum += pitch_abs.sum(dtype=torch.float64)
            tilt_sum += tilt.sum(dtype=torch.float64)
            max_tilt = torch.maximum(
                max_tilt,
                tilt.max().to(dtype=torch.float64),
            )

            base_height = env._get_base_height()
            base_height_sum += base_height.sum(dtype=torch.float64)
            min_base_height = torch.minimum(
                min_base_height,
                base_height.min().to(dtype=torch.float64),
            )

            collisions = torch.any(
                torch.norm(
                    env.contact_forces[
                        :,
                        env.penalised_contact_indices,
                        :,
                    ],
                    dim=-1,
                )
                > 0.1,
                dim=1,
            )
            collision_samples += collisions.sum(dtype=torch.float64)

    elapsed_seconds = time.perf_counter() - start_time
    successful_envs = torch.count_nonzero(fall_counts == 0).item()
    fall_count = fall_counts.sum().item()

    result = {
        "policy_name": f"{args.task}_checkpoint_{args.checkpoint}",
        "terrain": terrain,
        "seed": int(args.seed if args.seed is not None else train_cfg.seed),
        "num_envs": int(num_envs),
        "duration_seconds": duration_seconds,
        "command_vx_mps": command_vx,
        "command_vy_mps": command_vy,
        "command_yaw_radps": command_yaw,
        "total_samples": int(total_samples),
        "success_rate": successful_envs / num_envs,
        "fall_count": int(fall_count),
        "fall_rate_per_robot_min": (
            fall_count / (num_envs * duration_seconds / 60.0)
        ),
        "tracking_rmse_mps": (
            tracking_sse / total_samples
        ).sqrt().item(),
        "tracking_mae_mps": (tracking_abs / total_samples).item(),
        "mean_reward_per_step": (reward_sum / total_samples).item(),
        "mean_mechanical_power_w": (power_sum / total_samples).item(),
        "mean_action_delta": (
            action_delta_sum / (total_samples * env.num_actions)
        ).item(),
        "mean_abs_roll_rad": (roll_abs_sum / total_samples).item(),
        "mean_abs_pitch_rad": (pitch_abs_sum / total_samples).item(),
        "mean_tilt_rad": (tilt_sum / total_samples).item(),
        "max_tilt_rad": max_tilt.item(),
        "mean_base_height_m": (base_height_sum / total_samples).item(),
        "min_base_height_m": min_base_height.item(),
        "collision_sample_rate": (
            collision_samples / total_samples
        ).item(),
        "elapsed_seconds": elapsed_seconds,
        "throughput_env_steps_per_s": total_samples / elapsed_seconds,
    }

    directory = output_dir()
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / f"{terrain}.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"RESULT: {result_path}")
    print(f"GO2 TERRAIN {terrain.upper()} PASS")


if __name__ == "__main__":
    evaluate(get_args())
