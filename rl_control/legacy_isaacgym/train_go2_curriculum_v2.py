"""Conservative Isaac Gym fine-tuning from the validated control model.

This fallback exists for self-5000 hosts that cannot run the pinned Isaac Lab
stack.  It intentionally keeps the legacy 45-observation/12-action interface.
"""

from __future__ import annotations

import os

import isaacgym  # noqa: F401 - must be imported before torch/legged_gym

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry


# wave, slope, rough slope, stairs up, stairs down, obstacles,
# stepping stones, gap, flat
TERRAIN_PROPORTIONS = [0.05, 0.15, 0.05, 0.25, 0.10, 0.20, 0.00, 0.00, 0.20]


def _set_loaded_optimizer_lr(runner, learning_rate: float) -> None:
    algorithm = runner.alg
    if hasattr(algorithm, "learning_rate"):
        algorithm.learning_rate = learning_rate
    optimizer = getattr(algorithm, "optimizer", None)
    if optimizer is None:
        raise RuntimeError("Loaded PPO runner does not expose an optimizer")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def _set_loaded_action_std(runner, action_std: float) -> None:
    policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        policy = getattr(runner.alg, "policy", None)
    std = getattr(policy, "std", None)
    if std is None:
        raise RuntimeError("Loaded PPO policy does not expose action std")
    std.data.fill_(action_std)


def train(args) -> None:
    if not args.resume:
        raise RuntimeError("--resume is required")
    if abs(sum(TERRAIN_PROPORTIONS) - 1.0) > 1e-9:
        raise RuntimeError("Terrain proportions must sum to one")

    expected_source = int(os.environ.get("QRC_EXPECT_SOURCE_ITERATION", "5001"))
    learning_rate = float(os.environ.get("QRC_LEARNING_RATE", "0.0003"))
    entropy_coef = float(os.environ.get("QRC_ENTROPY_COEF", "0.005"))
    action_std = float(os.environ.get("QRC_ACTION_STD", "0.25"))

    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed
        train_cfg.seed = args.seed
    env_cfg.terrain.terrain_proportions = list(TERRAIN_PROPORTIONS)
    env_cfg.terrain.max_init_terrain_level = 2
    env_cfg.commands.ranges.lin_vel_x = [-0.10, 0.70]
    env_cfg.commands.ranges.lin_vel_y = [-0.25, 0.25]
    env_cfg.commands.ranges.ang_vel_yaw = [-0.80, 0.80]
    env_cfg.domain_rand.push_interval_s = 8
    env_cfg.domain_rand.max_push_vel_xy = 0.30
    env_cfg.domain_rand.max_push_ang_vel = 0.40
    train_cfg.algorithm.learning_rate = learning_rate
    train_cfg.algorithm.entropy_coef = entropy_coef
    train_cfg.policy.init_noise_std = action_std
    train_cfg.runner.save_interval = 100

    print("QRC_PROFILE: legacy_curriculum_v2")
    print("TERRAIN_PROPORTIONS:", TERRAIN_PROPORTIONS)
    print("MAX_INIT_TERRAIN_LEVEL:", env_cfg.terrain.max_init_terrain_level)
    print("SOURCE_RUN:", args.load_run)
    print("SOURCE_CHECKPOINT:", args.checkpoint)
    print("LEARNING_RATE:", learning_rate)
    print("ENTROPY_COEF:", entropy_coef)
    print("ACTION_STD:", action_std)
    print("EFFECTIVE_SEED:", env_cfg.seed)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(env=env, args=args, train_cfg=train_cfg)
    source_iteration = int(runner.current_learning_iteration)
    if source_iteration != expected_source:
        raise RuntimeError(
            f"Expected source iteration {expected_source}, got {source_iteration}"
        )

    _set_loaded_optimizer_lr(runner, learning_rate)
    _set_loaded_action_std(runner, action_std)
    if hasattr(env, "common_step_counter"):
        env.common_step_counter = source_iteration * env.num_steps_per_env
    if hasattr(env, "update_reward_curriculum"):
        env.update_reward_curriculum(force_update=True)

    iterations = int(train_cfg.runner.max_iterations)
    print("LOADED_ITERATION:", source_iteration)
    print("ADDITIONAL_ITERATIONS:", iterations)
    runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)

    expected_final = source_iteration + iterations
    print("FINAL_ITERATION:", runner.current_learning_iteration)
    if int(runner.current_learning_iteration) != expected_final:
        raise RuntimeError(
            f"Expected final iteration {expected_final}, got {runner.current_learning_iteration}"
        )
    print("GO2 CURRICULUM V2 TRAINING PASS")


if __name__ == "__main__":
    train(get_args())
