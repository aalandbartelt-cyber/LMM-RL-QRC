#!/usr/bin/env python3
"""Create and step one registered task without rendering or video."""

from __future__ import annotations

import argparse
import json

from isaaclab.app import AppLauncher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Unitree-Go2-Campus-Velocity")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    simulation_app = AppLauncher(headless=True).app
    report = {"task": args.task, "num_envs": args.num_envs, "steps": args.steps, "ready": False}
    env = None
    try:
        import gymnasium as gym
        import torch
        import unitree_rl_lab.tasks  # noqa: F401
        import campus_rl  # noqa: F401
        from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env = gym.make(args.task, cfg=env_cfg)
        observations, _ = env.reset()
        action_dim = env.unwrapped.action_manager.total_action_dim
        actions = torch.zeros((args.num_envs, action_dim), device=args.device)
        for _ in range(args.steps):
            observations, rewards, terminated, truncated, extras = env.step(actions)
        policy_obs = observations["policy"]
        report.update(
            {
                "ready": bool(torch.isfinite(policy_obs).all() and torch.isfinite(rewards).all()),
                "observation_shape": list(policy_obs.shape),
                "action_dim": action_dim,
                "reward_mean": float(rewards.mean()),
            }
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if env is not None:
            env.close()
        simulation_app.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
