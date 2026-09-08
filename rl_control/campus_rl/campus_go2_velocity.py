"""Campus terrain curriculum layered on the pinned Unitree Go2 task.

The actor interface intentionally remains identical to the official task:
45 observations and 12 joint-position actions.  Only terrain sampling,
command curriculum, randomization intensity and reward weights are adjusted.
"""

from __future__ import annotations

import copy

import gymnasium as gym
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    RobotEnvCfg,
)


CAMPUS_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, 0.35),
            platform_width=2.0,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.45,
            grid_height_range=(0.03, 0.16),
            platform_width=2.0,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.20,
            step_height_range=(0.04, 0.20),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.20),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class CampusGo2VelocityEnvCfg(RobotEnvCfg):
    """Physics training config for mixed campus-like ground conditions."""

    def __post_init__(self):
        self.scene.terrain.terrain_generator = copy.deepcopy(CAMPUS_TERRAINS_CFG)
        super().__post_init__()

        command = self.commands.base_velocity
        command.resampling_time_range = (2.0, 6.0)
        command.rel_standing_envs = 0.08
        command.ranges.lin_vel_x = (-0.15, 0.60)
        command.ranges.lin_vel_y = (-0.20, 0.20)
        command.ranges.ang_vel_z = (-0.70, 0.70)
        command.limit_ranges.lin_vel_x = (-0.35, 1.00)
        command.limit_ranges.lin_vel_y = (-0.40, 0.40)
        command.limit_ranges.ang_vel_z = (-1.00, 1.00)
        command.debug_vis = False

        # Reward changes are conservative so the official walking gait remains
        # the starting point while rough terrain and stair clearance improve.
        self.rewards.track_lin_vel_xy.weight = 2.0
        self.rewards.track_ang_vel_z.weight = 0.9
        self.rewards.feet_air_time.weight = 0.15
        self.rewards.feet_air_time.params["threshold"] = 0.35
        self.rewards.feet_slide.weight = -0.15
        self.rewards.undesired_contacts.weight = -1.5

        self.events.push_robot.interval_range_s = (7.0, 12.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.35, 0.35),
            "y": (-0.35, 0.35),
        }


@configclass
class CampusGo2VelocityPlayEnvCfg(CampusGo2VelocityEnvCfg):
    """Small deterministic layout for local visual inspection."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 14
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 7
        self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False


gym.register(
    id="Unitree-Go2-Campus-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CampusGo2VelocityEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}:CampusGo2VelocityPlayEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
        ),
    },
)
