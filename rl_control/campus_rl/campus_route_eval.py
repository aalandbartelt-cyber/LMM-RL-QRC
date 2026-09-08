"""Single-environment campus layout for feedback-driven policy evaluation."""

from __future__ import annotations

import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotSceneCfg

from .campus_go2_velocity import CampusGo2VelocityEnvCfg


def _static_box(
    prim_path: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
    )


@configclass
class CampusRouteSceneCfg(RobotSceneCfg):
    """Collision layout matching the major geometry in the supplied preview."""

    office = _static_box(
        "{ENV_REGEX_NS}/Campus/office", (-14.0, 12.0, 4.5), (18.0, 10.0, 9.0), (0.30, 0.45, 0.55)
    )
    warehouse = _static_box(
        "{ENV_REGEX_NS}/Campus/warehouse", (13.0, 12.0, 3.5), (19.0, 10.0, 7.0), (0.45, 0.42, 0.34)
    )
    power_room = _static_box(
        "{ENV_REGEX_NS}/Campus/power_room", (-17.0, -2.0, 1.8), (8.0, 6.0, 3.6), (0.58, 0.60, 0.62)
    )
    gate_house = _static_box(
        "{ENV_REGEX_NS}/Campus/gate_house", (-7.0, -17.2, 1.5), (5.0, 3.5, 3.0), (0.85, 0.88, 0.90)
    )
    north_wall = _static_box(
        "{ENV_REGEX_NS}/Campus/north_wall", (0.0, 19.5, 1.25), (60.0, 0.5, 2.5), (0.58, 0.60, 0.62)
    )
    west_wall = _static_box(
        "{ENV_REGEX_NS}/Campus/west_wall", (-29.75, 0.0, 1.25), (0.5, 40.0, 2.5), (0.58, 0.60, 0.62)
    )
    east_wall = _static_box(
        "{ENV_REGEX_NS}/Campus/east_wall", (29.75, 0.0, 1.25), (0.5, 40.0, 2.5), (0.58, 0.60, 0.62)
    )
    south_left_wall = _static_box(
        "{ENV_REGEX_NS}/Campus/south_left_wall", (-17.0, -19.5, 1.25), (26.0, 0.5, 2.5), (0.58, 0.60, 0.62)
    )
    south_right_wall = _static_box(
        "{ENV_REGEX_NS}/Campus/south_right_wall", (17.0, -19.5, 1.25), (26.0, 0.5, 2.5), (0.58, 0.60, 0.62)
    )


@configclass
class CampusRouteEvalEnvCfg(CampusGo2VelocityEnvCfg):
    """Evaluation only: flat campus floor, fixed buildings and one robot."""

    scene: CampusRouteSceneCfg = CampusRouteSceneCfg(num_envs=1, env_spacing=70.0)

    def __post_init__(self):
        self.curriculum.terrain_levels = None
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.num_envs = 1
        self.scene.robot.init_state.pos = (0.0, -16.0, 0.4)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (10000.0, 10000.0)
        self.episode_length_s = 300.0


gym.register(
    id="Unitree-Go2-Campus-Route-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}:CampusRouteEvalEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}:CampusRouteEvalEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
        ),
    },
)
