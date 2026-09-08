"""Campus security patrol scene for Isaac Lab 2.3 + Isaac Sim 5.1.

This standalone application creates a 60 m x 40 m representative industrial
campus, spawns Unitree Go2, previews a patrol mission, detects one configured
incident by a deterministic baseline, exports USD, and optionally records RGB
frames for an MP4.

The Go2 preview is kinematic route validation, not an RL gait claim. Connect
WaypointVelocityController to the existing low-level policy for physical walking.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Unitree Go2 campus security patrol scene")
parser.add_argument("--lighting", choices=("day", "night"), default="day")
parser.add_argument("--event", choices=("normal", "intrusion", "open_door", "fire"), default="intrusion")
parser.add_argument("--robot", choices=("go2", "marker"), default="go2")
parser.add_argument("--preview_speed", type=float, default=3.0, help="Kinematic preview speed in m/s")
parser.add_argument("--record", action="store_true", help="Save RGB frames with Replicator")
parser.add_argument("--record_fps", type=int, default=15)
parser.add_argument("--max_seconds", type=float, default=75.0)
parser.add_argument("--output_dir", default="outputs")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from mission_controller_legacy import EVENTS, KinematicPatrol, SECURITY_WAYPOINTS, route_length


COLORS = {
    "asphalt": (0.10, 0.12, 0.15),
    "concrete": (0.48, 0.50, 0.51),
    "grass": (0.11, 0.32, 0.14),
    "wall": (0.58, 0.60, 0.62),
    "office": (0.30, 0.45, 0.55),
    "warehouse": (0.45, 0.42, 0.34),
    "glass": (0.14, 0.35, 0.48),
    "road_line": (0.93, 0.74, 0.12),
    "route": (0.10, 0.65, 0.95),
    "checkpoint": (0.15, 0.95, 0.35),
    "alert": (0.95, 0.08, 0.05),
    "dark": (0.04, 0.05, 0.06),
    "tree": (0.10, 0.30, 0.09),
    "trunk": (0.28, 0.16, 0.08),
    "white": (0.85, 0.88, 0.90),
    "blue": (0.05, 0.22, 0.70),
    "orange": (1.00, 0.30, 0.02),
    "smoke": (0.22, 0.22, 0.24),
}


def material(stage, name: str, color: tuple[float, float, float], opacity: float = 1.0):
    path = f"/World/Materials/{name}"
    existing = stage.GetPrimAtPath(path)
    if existing.IsValid():
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def xform(prim, pos, scale, yaw_deg: float = 0.0):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    translate = xf.AddTranslateOp()
    translate.Set(Gf.Vec3d(*pos))
    xf.AddRotateZOp().Set(yaw_deg)
    xf.AddScaleOp().Set(Gf.Vec3d(*scale))
    return translate


def cube(stage, path, pos, size, mat, collision=False, yaw_deg=0.0):
    geom = UsdGeom.Cube.Define(stage, path)
    geom.CreateSizeAttr(1.0)
    xform(geom.GetPrim(), pos, size, yaw_deg)
    bind(geom.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return geom


def cylinder(stage, path, pos, radius, height, mat, collision=False):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateRadiusAttr(1.0)
    geom.CreateHeightAttr(1.0)
    geom.CreateAxisAttr("Z")
    xform(geom.GetPrim(), pos, (radius, radius, height))
    bind(geom.GetPrim(), mat)
    if collision:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    return geom


def sphere(stage, path, pos, radius, mat):
    geom = UsdGeom.Sphere.Define(stage, path)
    geom.CreateRadiusAttr(1.0)
    xform(geom.GetPrim(), pos, (radius, radius, radius))
    bind(geom.GetPrim(), mat)
    return geom


def cone(stage, path, pos, radius, height, mat):
    geom = UsdGeom.Cone.Define(stage, path)
    geom.CreateRadiusAttr(1.0)
    geom.CreateHeightAttr(1.0)
    geom.CreateAxisAttr("Z")
    xform(geom.GetPrim(), pos, (radius, radius, height))
    bind(geom.GetPrim(), mat)
    return geom


def route_segment(stage, index, a, b, mat):
    dx, dy = b.x - a.x, b.y - a.y
    length = math.hypot(dx, dy)
    mid = ((a.x + b.x) * 0.5, (a.y + b.y) * 0.5, 0.065)
    yaw = math.degrees(math.atan2(dy, dx))
    cube(stage, f"/World/Route/Segment_{index:02d}", mid, (length, 0.16, 0.035), mat, yaw_deg=yaw)


def create_tree(stage, index, x, y, mats):
    cylinder(stage, f"/World/Vegetation/Tree_{index}/Trunk", (x, y, 1.2), 0.18, 2.4, mats["trunk"], True)
    sphere(stage, f"/World/Vegetation/Tree_{index}/Crown", (x, y, 3.0), 1.25, mats["tree"])


def create_lamp(stage, index, x, y, night, mats):
    cylinder(stage, f"/World/Lamps/Lamp_{index}/Pole", (x, y, 2.5), 0.08, 5.0, mats["dark"], True)
    sphere(stage, f"/World/Lamps/Lamp_{index}/Bulb", (x, y, 5.0), 0.18, mats["white"])
    if night:
        light = UsdLux.SphereLight.Define(stage, f"/World/Lamps/Lamp_{index}/Light")
        light.CreateIntensityAttr(18000.0)
        light.CreateRadiusAttr(0.15)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.78, 0.55))
        UsdGeom.Xformable(light.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(x, y, 4.9))


def create_camera_prop(stage, index, x, y, yaw_deg, mats):
    cylinder(stage, f"/World/Security/CCTV_{index}/Pole", (x, y, 2.4), 0.07, 4.8, mats["dark"], True)
    cube(stage, f"/World/Security/CCTV_{index}/Body", (x, y, 4.8), (0.45, 0.20, 0.20), mats["white"], yaw_deg=yaw_deg)
    cube(stage, f"/World/Security/CCTV_{index}/Lens", (x, y, 4.8), (0.47, 0.07, 0.12), mats["dark"], yaw_deg=yaw_deg)


def create_scene(stage, lighting: str, event_key: str):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    mats = {name: material(stage, name, color) for name, color in COLORS.items()}
    mats["route_transparent"] = material(stage, "route_transparent", COLORS["route"], 0.85)

    # Lighting.
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(900.0 if lighting == "day" else 90.0)
    dome.CreateColorAttr(Gf.Vec3f(*(COLORS["white"] if lighting == "day" else (0.10, 0.14, 0.24))))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(2600.0 if lighting == "day" else 80.0)
    sun.CreateAngleAttr(0.55)
    UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 25.0, -35.0))

    # Terrain and roads. Only the ground owns collision to avoid overlapping contact planes.
    cube(stage, "/World/Ground", (0, 0, -0.10), (60, 40, 0.20), mats["concrete"], True)
    for idx, (pos, size) in enumerate((
        ((0, -12, 0.015), (52, 5, 0.03)),
        ((0, 6, 0.015), (52, 5, 0.03)),
        ((-23.5, -3, 0.015), (5, 23, 0.03)),
        ((23.5, -3, 0.015), (5, 23, 0.03)),
        ((0, -3, 0.016), (5, 14, 0.032)),
    )):
        cube(stage, f"/World/Roads/Road_{idx}", pos, size, mats["asphalt"])
    for idx, (pos, size) in enumerate((
        ((-15, -1.5, 0.012), (12, 8, 0.024)),
        ((14, -2.0, 0.012), (15, 8, 0.024)),
        ((0, 14.0, 0.012), (58, 6, 0.024)),
    )):
        cube(stage, f"/World/Green/Grass_{idx}", pos, size, mats["grass"])

    # Road edge markings and parking bays.
    for idx, x in enumerate((-24.5, 24.5)):
        cube(stage, f"/World/Roads/EdgeVertical_{idx}", (x, -3, 0.04), (0.12, 23, 0.02), mats["road_line"])
    for idx, y in enumerate((-14.5, -9.5, 3.5, 8.5)):
        cube(stage, f"/World/Roads/EdgeHorizontal_{idx}", (0, y, 0.04), (52, 0.12, 0.02), mats["road_line"])
    for idx, x in enumerate((8, 12, 16, 20)):
        cube(stage, f"/World/Parking/Line_{idx}", (x, -4.0, 0.045), (0.10, 6.0, 0.02), mats["white"])
    for idx, x in enumerate((10, 18)):
        cube(stage, f"/World/Parking/Car_{idx}", (x, -3.5, 0.55), (3.6, 1.7, 1.1), mats["blue"], True, 90.0)

    # Buildings and doors/windows.
    cube(stage, "/World/Buildings/Office", (-14, 12, 4.5), (18, 10, 9), mats["office"], True)
    for floor in range(1, 4):
        z = 1.5 + floor * 1.8
        cube(stage, f"/World/Buildings/Office/Window_{floor}", (-14, 6.96, z), (13.0, 0.08, 0.75), mats["glass"])
    cube(stage, "/World/Buildings/Office/Door", (-16, 6.94, 1.25), (2.2, 0.12, 2.5), mats["dark"])
    cube(stage, "/World/Buildings/Warehouse", (13, 12, 3.5), (19, 10, 7), mats["warehouse"], True)
    cube(stage, "/World/Buildings/Warehouse/Door", (15, 6.94, 1.8), (3.2, 0.14, 3.6), mats["dark"])
    cube(stage, "/World/Buildings/PowerRoom", (-17, -2, 1.8), (8, 6, 3.6), mats["wall"], True)
    cube(stage, "/World/Buildings/PowerRoom/Door", (-17, -5.03, 1.2), (1.8, 0.12, 2.4), mats["dark"])
    cube(stage, "/World/Buildings/GateHouse", (-7, -17.2, 1.5), (5, 3.5, 3.0), mats["white"], True)
    cube(stage, "/World/Buildings/GateHouse/Window", (-4.46, -17.2, 1.7), (0.08, 1.8, 1.0), mats["glass"])

    # Perimeter wall with an 8 m entrance gap.
    cube(stage, "/World/Perimeter/NorthWall", (0, 19.5, 1.25), (60, 0.5, 2.5), mats["wall"], True)
    cube(stage, "/World/Perimeter/WestWall", (-29.75, 0, 1.25), (0.5, 40, 2.5), mats["wall"], True)
    cube(stage, "/World/Perimeter/EastWall", (29.75, 0, 1.25), (0.5, 40, 2.5), mats["wall"], True)
    cube(stage, "/World/Perimeter/SouthLeft", (-17, -19.5, 1.25), (26, 0.5, 2.5), mats["wall"], True)
    cube(stage, "/World/Perimeter/SouthRight", (17, -19.5, 1.25), (26, 0.5, 2.5), mats["wall"], True)
    for idx, (pos, size) in enumerate((
        ((0, 19.5, 2.75), (60, 0.08, 0.08)),
        ((-29.75, 0, 2.75), (0.08, 40, 0.08)),
        ((29.75, 0, 2.75), (0.08, 40, 0.08)),
    )):
        cube(stage, f"/World/Perimeter/ElectronicFence_{idx}", pos, size, mats["alert"])
    for idx, x in enumerate((-4.2, 4.2)):
        cylinder(stage, f"/World/Gate/Bollard_{idx}", (x, -18.5, 0.55), 0.22, 1.1, mats["road_line"], True)

    # Vegetation, lamps, security cameras, hydrants and street furniture.
    tree_positions = [(-26, -11), (-26, -3), (-26, 11), (-4, 14), (3, 14), (25, 13), (25, -13), (-13, -7)]
    for idx, (x, y) in enumerate(tree_positions):
        create_tree(stage, idx, x, y, mats)
    lamp_positions = [(-18, -14), (-7, -14), (7, -14), (19, -14), (-25, 0), (25, 0), (-8, 8), (8, 8)]
    for idx, (x, y) in enumerate(lamp_positions):
        create_lamp(stage, idx, x, y, lighting == "night", mats)
    for idx, (x, y, yaw) in enumerate(((-27, -16, 35), (-27, 16, -35), (27, 16, -145), (27, -16, 145), (-5, 7, 0), (4, 7, 180))):
        create_camera_prop(stage, idx, x, y, yaw, mats)
    for idx, (x, y) in enumerate(((-9, -8), (19, 4))):
        cylinder(stage, f"/World/Safety/Hydrant_{idx}", (x, y, 0.45), 0.20, 0.9, mats["alert"], True)
    for idx, (x, y) in enumerate(((-8, -7), (6, -7))):
        cube(stage, f"/World/Furniture/Bench_{idx}", (x, y, 0.45), (2.2, 0.55, 0.18), mats["trunk"], True)
    for idx, (x, y) in enumerate(((-4, -8), (4, -8))):
        cube(stage, f"/World/Furniture/Bin_{idx}", (x, y, 0.45), (0.55, 0.55, 0.9), mats["dark"], True)

    # Patrol route and inspection points.
    for idx, (a, b) in enumerate(zip(SECURITY_WAYPOINTS[:-1], SECURITY_WAYPOINTS[1:])):
        route_segment(stage, idx, a, b, mats["route_transparent"])
    for idx, wp in enumerate(SECURITY_WAYPOINTS):
        wp_mat = mats["alert"] if idx == 2 and event_key == "intrusion" else mats["checkpoint"]
        cylinder(stage, f"/World/Route/Checkpoint_{idx}", (wp.x, wp.y, 0.07), 0.45, 0.06, wp_mat)

    event = EVENTS[event_key]
    if event_key == "intrusion":
        x, y = event["position"]
        cylinder(stage, "/World/Incident/IntruderBody", (x, y, 0.9), 0.30, 1.4, mats["alert"], True)
        sphere(stage, "/World/Incident/IntruderHead", (x, y, 1.75), 0.28, mats["orange"])
    elif event_key == "open_door":
        x, y = event["position"]
        cube(stage, "/World/Incident/OpenDoor", (x + 1.5, y, 1.8), (3.2, 0.14, 3.6), mats["alert"], yaw_deg=55.0)
    elif event_key == "fire":
        x, y = event["position"]
        cone(stage, "/World/Incident/Flame", (x, y, 0.7), 0.55, 1.4, mats["orange"])
        for idx, (dz, radius) in enumerate(((1.5, 0.5), (2.2, 0.7), (3.0, 0.9))):
            sphere(stage, f"/World/Incident/Smoke_{idx}", (x, y, dz), radius, mats["smoke"])

    return mats


def create_marker(stage, mats):
    root = UsdGeom.Xform.Define(stage, "/World/RobotMarker")
    translate = root.AddTranslateOp()
    root.AddRotateZOp()
    cube(stage, "/World/RobotMarker/Body", (0, 0, 0.45), (0.8, 0.42, 0.25), mats["blue"])
    for idx, (x, y) in enumerate(((0.28, 0.22), (0.28, -0.22), (-0.28, 0.22), (-0.28, -0.22))):
        cylinder(stage, f"/World/RobotMarker/Leg_{idx}", (x, y, 0.18), 0.05, 0.36, mats["dark"])
    return root, translate


def set_marker_pose(marker_root, x, y, yaw):
    ops = marker_root.GetOrderedXformOps()
    ops[0].Set(Gf.Vec3d(x, y, 0.0))
    if len(ops) > 1:
        ops[1].Set(math.degrees(yaw))


def set_go2_pose(robot, x, y, yaw, device):
    root_pose = torch.tensor(
        [[x, y, 0.42, math.cos(yaw * 0.5), 0.0, 0.0, math.sin(yaw * 0.5)]],
        dtype=torch.float32,
        device=device,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=device))
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


def setup_recording(output_dir: Path, resolution=(1280, 720)):
    import omni.replicator.core as rep

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rep.orchestrator.set_capture_on_play(False)
    camera = rep.create.camera(position=(46, -52, 38), look_at=(0, 0, 0), focal_length=24)
    render_product = rep.create.render_product(camera, resolution)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frames_dir), rgb=True, frame_padding=5)
    writer.attach([render_product])
    return rep, writer, render_product


def write_report(output_dir: Path, report: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "patrol_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def main():
    output_dir = Path(args_cli.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([46.0, -52.0, 38.0], [0.0, 0.0, 0.0])
    stage = omni.usd.get_context().get_stage()
    mats = create_scene(stage, args_cli.lighting, args_cli.event)

    robot = None
    marker_root = None
    if args_cli.robot == "go2":
        robot_cfg = UNITREE_GO2_CFG.replace(prim_path="/World/Go2")
        robot = Articulation(robot_cfg)
    else:
        marker_root, _ = create_marker(stage, mats)

    usd_path = output_dir / "security_park.usd"
    stage.GetRootLayer().Export(str(usd_path))
    sim.reset()
    if robot is not None:
        robot.reset()

    recorder = setup_recording(output_dir) if args_cli.record else None
    dt = sim.get_physics_dt()
    capture_interval = max(1, round(1.0 / (dt * args_cli.record_fps)))
    max_steps = int(args_cli.max_seconds / dt)
    follower = KinematicPatrol(args_cli.preview_speed)
    event = EVENTS[args_cli.event]
    detection_radius = 3.0
    alert_detected = False
    reached_points = [SECURITY_WAYPOINTS[0].name]
    mission_events = [{"type": "mission_start", "point": SECURITY_WAYPOINTS[0].name, "time_s": 0.0}]
    elapsed = 0.0

    print(f"[INFO] Scene ready: lighting={args_cli.lighting}, event={args_cli.event}, robot={args_cli.robot}")
    print(f"[INFO] Patrol route length: {route_length():.1f} m")

    for step in range(max_steps):
        if not simulation_app.is_running():
            break
        state = follower.advance(dt)
        elapsed = step * dt

        if robot is not None:
            set_go2_pose(robot, state["x"], state["y"], state["yaw"], sim.device)
        else:
            set_marker_pose(marker_root, state["x"], state["y"], state["yaw"])

        if state["arrived"]:
            name = state["arrived"]["name"]
            reached_points.append(name)
            mission_events.append({"type": "checkpoint", "point": name, "time_s": round(elapsed, 2)})
            print(f"[PATROL] Arrived: {name}")

        if event["position"] is not None and not alert_detected:
            ex, ey = event["position"]
            if math.hypot(state["x"] - ex, state["y"] - ey) <= detection_radius:
                alert_detected = True
                mission_events.append({
                    "type": "alert",
                    "result": event["expected_result"],
                    "description": event["name"],
                    "time_s": round(elapsed, 2),
                    "position": [ex, ey],
                })
                print(f"[ALERT] {event['name']} @ ({ex:.1f}, {ey:.1f})")

        sim.step(render=(args_cli.record or not args_cli.headless))
        if recorder is not None and step % capture_interval == 0:
            # Capture without advancing the simulation timeline a second time.
            recorder[0].orchestrator.step(rt_subframes=1, delta_time=0.0)

        if state["complete"]:
            mission_events.append({"type": "mission_complete", "time_s": round(elapsed, 2)})
            print("[PATROL] Route complete; robot returned to the main gate.")
            break

    if recorder is not None:
        recorder[1].detach()
        recorder[2].destroy()
        recorder[0].orchestrator.wait_until_complete()

    expected_detection = args_cli.event != "normal"
    report = {
        "scene": "60m x 40m 园区巡逻安防代表场景",
        "platform": "Isaac Lab 2.3 + Isaac Sim 5.1",
        "robot": "Unitree Go2" if args_cli.robot == "go2" else "fallback marker",
        "preview_note": "运动学路径预览，不代表RL真实步态",
        "lighting": args_cli.lighting,
        "configured_event": args_cli.event,
        "event_description": event["name"],
        "route_length_m": round(route_length(), 2),
        "elapsed_s": round(elapsed, 2),
        "checkpoints_total": len(SECURITY_WAYPOINTS),
        "checkpoints_reached": reached_points,
        "returned_to_gate": SECURITY_WAYPOINTS[-1].name in reached_points,
        "alert_detected": alert_detected,
        "detection_result_correct": alert_detected == expected_detection,
        "events": mission_events,
    }
    write_report(output_dir, report)
    print(f"[INFO] USD: {usd_path}")
    print(f"[INFO] Report: {output_dir / 'patrol_report.json'}")
    if args_cli.record:
        print(f"[INFO] RGB frames: {output_dir / 'frames'}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
