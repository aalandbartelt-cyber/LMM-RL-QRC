"""Run baseline5001 as a 12-joint policy in local MuJoCo route scenes.

The high-level controller only supplies bounded body-velocity commands. Robot
root pose and joint motion always come from MuJoCo dynamics integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from .scenarios import Point, Scenario


DT = 0.002
DECIMATION = 10
DEFAULT_JOINT_POS = np.array(
    [
        0.1,
        0.8,
        -1.5,
        -0.1,
        0.8,
        -1.5,
        0.1,
        1.0,
        -1.5,
        -0.1,
        1.0,
        -1.5,
    ],
    dtype=np.float32,
)
KP = np.full(12, 20.0, dtype=np.float32)
KD = np.full(12, 0.5, dtype=np.float32)
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ACTION_SCALE = 0.25


@dataclass(frozen=True)
class RouteCommand:
    forward_mps: float
    yaw_rate_rps: float
    reached: bool


@dataclass(frozen=True)
class PhysicsFrame:
    rgb: np.ndarray
    progress: float
    time_s: float
    position: tuple[float, float, float]
    target_name: str
    base_height_m: float
    completed: bool


@dataclass(frozen=True)
class RouteResult:
    status: str
    elapsed_s: float
    checkpoints_reached: int
    checkpoints_total: int
    minimum_base_height_m: float
    final_position: tuple[float, float, float]
    trajectory: tuple[tuple[float, float, float, float], ...]


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compact_route(scenario: Scenario, extent_m: float = 3.5) -> tuple[Point, ...]:
    """Translate a mission route to the origin and scale it into a local arena."""
    origin = scenario.route[0]
    offsets = [(point.x - origin.x, point.y - origin.y) for point in scenario.route]
    largest = max(max(abs(x), abs(y)) for x, y in offsets)
    scale = 1.0 if largest <= extent_m or largest == 0.0 else extent_m / largest
    return tuple(
        Point(point.name, round(x * scale, 6), round(y * scale, 6), point.kind)
        for point, (x, y) in zip(scenario.route, offsets)
    )


def route_command(
    x: float,
    y: float,
    yaw: float,
    target: Point,
    max_speed_mps: float,
    reach_radius_m: float = 0.36,
) -> RouteCommand:
    """Produce a safe body-frame command toward a world-frame waypoint."""
    dx, dy = target.x - x, target.y - y
    distance = math.hypot(dx, dy)
    if distance <= reach_radius_m:
        return RouteCommand(0.0, 0.0, True)
    desired_yaw = math.atan2(dy, dx)
    yaw_error = wrap_angle(desired_yaw - yaw)
    yaw_rate = max(-1.0, min(1.0, 1.6 * yaw_error))
    if abs(yaw_error) >= 0.90:
        forward = 0.0
    else:
        alignment = max(0.0, math.cos(yaw_error))
        forward = min(max_speed_mps, 0.75 * distance) * alignment
    return RouteCommand(float(forward), float(yaw_rate), False)


class CheckpointActor:
    """Minimal NumPy inference wrapper for the actor stored in an RSL-RL checkpoint."""

    def __init__(self, checkpoint: str | Path):
        import torch

        payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload)
        layer_ids = (0, 2, 4, 6)
        self.layers = tuple(
            (
                state[f"actor.{index}.weight"].detach().cpu().numpy().astype(np.float32),
                state[f"actor.{index}.bias"].detach().cpu().numpy().astype(np.float32),
            )
            for index in layer_ids
        )
        if self.layers[0][0].shape[1] != 45 or self.layers[-1][0].shape[0] != 12:
            raise ValueError("checkpoint actor must map 45 observations to 12 actions")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        value = np.asarray(observation, dtype=np.float32).reshape(45)
        for index, (weight, bias) in enumerate(self.layers):
            value = weight @ value + bias
            if index < len(self.layers) - 1:
                value = np.where(value > 0.0, value, np.expm1(value)).astype(np.float32)
        return value.astype(np.float32)


def _route_geoms(route: tuple[Point, ...]) -> str:
    geoms: list[str] = []
    for index, (a, b) in enumerate(zip(route[:-1], route[1:])):
        geoms.append(
            f'<geom name="route_{index}" type="capsule" fromto="{a.x} {a.y} 0.008 {b.x} {b.y} 0.008" '
            'size="0.025" rgba="0.05 0.75 0.95 0.75" contype="0" conaffinity="0" group="2"/>'
        )
    for index, point in enumerate(route):
        color = "0.15 0.9 0.5 1"
        if point.kind in {"alert", "hazard", "blocked"}:
            color = "1 0.18 0.22 1"
        elif point.kind in {"rough", "slope", "detour"}:
            color = "1 0.7 0.2 1"
        geoms.append(
            f'<geom name="waypoint_{index}" type="cylinder" pos="{point.x} {point.y} 0.018" '
            f'size="0.075 0.012" rgba="{color}" contype="0" conaffinity="0" group="2"/>'
        )
    return "\n".join(geoms)


def _campus_geoms() -> str:
    return """
    <geom name="office" type="box" pos="-1.30 2.82 0.55" size="0.75 0.28 0.55" material="building_blue"/>
    <geom name="warehouse" type="box" pos="1.25 2.95 0.45" size="0.80 0.40 0.45" material="building_sand"/>
    <geom name="power_room" type="box" pos="-1.45 1.20 0.32" size="0.42 0.38 0.32" material="building_gray"/>
    <geom name="gatehouse" type="box" pos="-0.55 -0.35 0.26" size="0.30 0.22 0.26" material="building_gray"/>
    <geom name="parking" type="box" pos="1.55 0.90 0.012" size="0.65 0.48 0.012" rgba="0.25 0.32 0.40 1" contype="0" conaffinity="0"/>
    <geom name="intrusion_marker" type="cylinder" pos="-2.47 2.16 0.18" size="0.10 0.18" rgba="1 0.08 0.08 1" contype="0" conaffinity="0"/>
    """


def _disaster_geoms() -> str:
    return """
    <geom name="safe_zone" type="box" pos="0 0 0.008" size="0.50 0.45 0.008" rgba="0.12 0.65 0.42 1" contype="0" conaffinity="0"/>
    <geom name="rubble_1" type="box" pos="1.05 0.67 0.025" size="0.20 0.28 0.025" euler="0 0 0.2" rgba="0.48 0.39 0.32 1"/>
    <geom name="rubble_2" type="box" pos="1.42 1.06 0.035" size="0.22 0.25 0.035" euler="0 0 -0.15" rgba="0.55 0.44 0.34 1"/>
    <geom name="ramp" type="box" pos="1.85 1.45 0.024" size="0.38 0.34 0.022" euler="0 0.10 0.75" rgba="0.50 0.48 0.43 1"/>
    <geom name="collapsed_wall_a" type="box" pos="2.80 0.45 0.30" size="0.62 0.12 0.12" euler="0 0.3 0.4" material="building_sand"/>
    <geom name="collapsed_wall_b" type="box" pos="3.15 0.65 0.22" size="0.45 0.10 0.10" euler="0 -0.2 -0.6" material="building_gray"/>
    <geom name="target_beacon" type="cylinder" pos="3.05 2.08 0.25" size="0.10 0.25" rgba="0.1 0.85 0.95 1" contype="0" conaffinity="0"/>
    <geom name="delivery_pad" type="cylinder" pos="3.50 1.52 0.012" size="0.30 0.012" rgba="0.95 0.65 0.1 1" contype="0" conaffinity="0"/>
    """


def build_scene_xml(scenario: Scenario, go2_xml: str | Path) -> str:
    """Create an MJCF scene that includes the supplied Go2 model and mission map."""
    route = compact_route(scenario)
    scenario_geoms = _campus_geoms() if scenario.key == "campus_security" else _disaster_geoms()
    include_path = escape(Path(go2_xml).resolve().as_posix(), quote=True)
    mesh_path = escape((Path(go2_xml).resolve().parent / "assets").as_posix(), quote=True)
    return f"""<mujoco model="{scenario.key}_baseline5001">
  <include file="{include_path}"/>
  <compiler angle="radian" meshdir="{mesh_path}"/>
  <option timestep="{DT}" gravity="0 0 -9.81"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.15 0.15 0.15"/>
    <rgba haze="0.08 0.13 0.18 1"/>
    <global offwidth="1280" offheight="720" azimuth="135" elevation="-24"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.08 0.16 0.24" rgb2="0.01 0.03 0.06" width="512" height="3072"/>
    <texture type="2d" name="ground_tex" builtin="checker" rgb1="0.12 0.16 0.19" rgb2="0.16 0.22 0.26" width="256" height="256"/>
    <material name="ground_mat" texture="ground_tex" texrepeat="12 12" texuniform="true" reflectance="0.08"/>
    <material name="building_blue" rgba="0.18 0.35 0.48 1"/>
    <material name="building_sand" rgba="0.48 0.42 0.30 1"/>
    <material name="building_gray" rgba="0.38 0.43 0.48 1"/>
  </asset>
  <worldbody>
    <light pos="0 -2 7" dir="0 0 -1" directional="true" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="ground_mat" friction="1.0 0.005 0.0001"/>
    {scenario_geoms}
    {_route_geoms(route)}
  </worldbody>
</mujoco>"""


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quaternion_yaw(quaternion: np.ndarray) -> float:
    qw, qx, qy, qz = quaternion
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _progress(route: tuple[Point, ...], target_index: int, x: float, y: float) -> float:
    if target_index >= len(route):
        return 1.0
    completed = max(0, target_index - 1)
    if completed >= len(route) - 1:
        return 1.0
    start, target = route[completed], route[target_index]
    segment = math.hypot(target.x - start.x, target.y - start.y)
    remaining = math.hypot(target.x - x, target.y - y)
    fraction = 1.0 if segment == 0.0 else min(1.0, max(0.0, 1.0 - remaining / segment))
    return min(1.0, (completed + fraction) / (len(route) - 1))


def simulate_route(
    scenario: Scenario,
    checkpoint: str | Path,
    go2_xml: str | Path,
    fps: int = 30,
    max_seconds: float = 45.0,
    render_size: tuple[int, int] = (1280, 720),
) -> tuple[Iterator[PhysicsFrame], dict[str, RouteResult | None]]:
    """Return a lazy frame iterator and a holder populated with the final result."""
    import mujoco

    model = mujoco.MjModel.from_xml_string(build_scene_xml(scenario, go2_xml))
    model.opt.timestep = DT
    data = mujoco.MjData(model)
    route = compact_route(scenario)
    data.qpos[2] = 0.43
    data.qpos[7:] = DEFAULT_JOINT_POS
    # Face the first route segment so the initial waypoint turn stays well
    # inside the stable basin of the proportional heading controller.
    heading = math.atan2(route[1].y - route[0].y, route[1].x - route[0].x)
    data.qpos[3:7] = (math.cos(heading / 2.0), 0.0, 0.0, math.sin(heading / 2.0))
    mujoco.mj_forward(model, data)
    actor = CheckpointActor(checkpoint)
    renderer = mujoco.Renderer(model, height=render_size[1], width=render_size[0])
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 4.3
    camera.azimuth = 135.0
    camera.elevation = -23.0
    result_holder: dict[str, RouteResult | None] = {"result": None}

    def generate() -> Iterator[PhysicsFrame]:
        action = np.zeros(12, dtype=np.float32)
        target_joint = DEFAULT_JOINT_POS.copy()
        observation = np.zeros(45, dtype=np.float32)
        command = np.zeros(3, dtype=np.float32)
        target_index = 1
        minimum_height = float(data.qpos[2])
        trajectory: list[tuple[float, float, float, float]] = []
        render_interval = max(1, round(1.0 / (DT * fps)))
        total_steps = max(1, int(max_seconds / DT))
        settle_seconds = 1.0
        completion_time: float | None = None

        try:
            for step in range(total_steps):
                torque = (target_joint - data.qpos[7:]) * KP - data.qvel[6:] * KD
                data.ctrl[:] = torque
                mujoco.mj_step(model, data)
                sim_time = (step + 1) * DT
                minimum_height = min(minimum_height, float(data.qpos[2]))

                if (step + 1) % DECIMATION == 0:
                    if sim_time < settle_seconds:
                        command[:] = 0.0
                    elif target_index < len(route):
                        yaw = quaternion_yaw(data.qpos[3:7])
                        target = route[target_index]
                        high_level = route_command(
                            float(data.qpos[0]),
                            float(data.qpos[1]),
                            yaw,
                            target,
                            scenario.max_speed_mps,
                        )
                        if high_level.reached:
                            target_index += 1
                            if target_index >= len(route):
                                completion_time = sim_time
                                command[:] = 0.0
                            else:
                                target = route[target_index]
                                high_level = route_command(
                                    float(data.qpos[0]),
                                    float(data.qpos[1]),
                                    yaw,
                                    target,
                                    scenario.max_speed_mps,
                                )
                        if target_index < len(route):
                            command[:] = (high_level.forward_mps, 0.0, high_level.yaw_rate_rps)
                    else:
                        command[:] = 0.0

                    observation[:3] = (data.qvel[3:6] * 0.25).astype(np.float32)
                    observation[3:6] = gravity_orientation(data.qpos[3:7])
                    observation[6:9] = command * CMD_SCALE
                    observation[9:21] = (data.qpos[7:] - DEFAULT_JOINT_POS).astype(np.float32)
                    observation[21:33] = (data.qvel[6:] * 0.05).astype(np.float32)
                    observation[33:45] = action
                    action = actor(observation)
                    target_joint = action * ACTION_SCALE + DEFAULT_JOINT_POS

                if step % render_interval == 0:
                    camera.lookat[:] = (float(data.qpos[0]) + 0.5, float(data.qpos[1]), 0.30)
                    renderer.update_scene(data, camera=camera)
                    rgb = renderer.render().copy()
                    progress = _progress(route, target_index, float(data.qpos[0]), float(data.qpos[1]))
                    target_name = route[min(target_index, len(route) - 1)].name
                    completed = target_index >= len(route)
                    trajectory.append((sim_time, float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])))
                    yield PhysicsFrame(
                        rgb=rgb,
                        progress=progress,
                        time_s=sim_time,
                        position=(float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])),
                        target_name=target_name,
                        base_height_m=float(data.qpos[2]),
                        completed=completed,
                    )
                    if completion_time is not None and sim_time >= completion_time + 1.5:
                        break

            status = "PASS" if target_index >= len(route) and minimum_height >= 0.18 else "FAILED"
            result_holder["result"] = RouteResult(
                status=status,
                elapsed_s=round(trajectory[-1][0], 3) if trajectory else 0.0,
                checkpoints_reached=min(target_index, len(route)) - 1,
                checkpoints_total=len(route) - 1,
                minimum_base_height_m=minimum_height,
                final_position=(float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])),
                trajectory=tuple(trajectory),
            )
        finally:
            renderer.close()

    return generate(), result_holder
