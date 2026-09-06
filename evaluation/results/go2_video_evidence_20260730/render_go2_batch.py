import os
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import onnxruntime as ort

ROOT = Path("/home/jovyan/go2_work/go2_rl_gym-master")
XML = Path(os.environ.get("QRC_XML", str(ROOT / "resources/robots/go2/flat.xml")))
POLICY = Path(os.environ.get("QRC_POLICY", str(ROOT / "logs/go2_baseline/exported/policies/policy.onnx")))
OUTPUT = Path(os.environ.get("QRC_OUTPUT", "/home/jovyan/go2_work/video_output.mp4"))
SUMMARY = Path(os.environ.get("QRC_SUMMARY", str(OUTPUT.with_suffix(".json"))))

DT = 0.002
DECIMATION = 10
DURATION = float(os.environ.get("QRC_DURATION", "10"))
FPS = int(os.environ.get("QRC_FPS", "50"))
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
WIDTH, HEIGHT = 640, 360

KP = np.full(12, 20.0, dtype=np.float32)
KD = np.full(12, 0.5, dtype=np.float32)
DEFAULT = np.array([
    0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
    0.1, 1.0, -1.5, -0.1, 1.0, -1.5
], dtype=np.float32)

CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ACTION_SCALE = 0.25

def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz),
    ], dtype=np.float32)

def rotate_inverse(q, v):
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qw, qv = q[0], q[1:]
    return (
        v * (2.0 * qw * qw - 1.0)
        - np.cross(qv, v) * qw * 2.0
        + qv * np.dot(qv, v) * 2.0
    )

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
model.opt.timestep = DT
mujoco.mj_forward(model, data)

session = ort.InferenceSession(
    str(POLICY),
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name
print("Policy:", POLICY)
print("ONNX input:", input_name, session.get_inputs()[0].shape)

renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.distance = 2.0
camera.azimuth = 60.0
camera.elevation = -20.0

writer = imageio_ffmpeg.write_frames(
    str(OUTPUT),
    size=(WIDTH, HEIGHT),
    fps=FPS,
    codec="libx264",
    pix_fmt_in="rgb24",
    pix_fmt_out="yuv420p",
    quality=8,
    macro_block_size=1,
)
writer.send(None)

action = np.zeros(12, dtype=np.float32)
target = DEFAULT.copy()
obs = np.zeros(45, dtype=np.float32)
start_position = data.qpos[:3].copy()
minimum_height = float(data.qpos[2])
frames = 0
total_steps = int(DURATION / DT)
render_interval = max(1, round(1.0 / (DT * FPS)))

try:
    for step in range(total_steps):
        torque = (target - data.qpos[7:]) * KP - data.qvel[6:] * KD
        data.ctrl[:] = torque
        mujoco.mj_step(model, data)

        sim_time = step * DT
        minimum_height = min(minimum_height, float(data.qpos[2]))

        if (step + 1) % DECIMATION == 0:
            command = np.array(
                [0.0, 0.0, 0.0] if sim_time < 1.0 else [
                    float(os.environ.get("QRC_VX", "0.4")),
                    float(os.environ.get("QRC_VY", "0.0")),
                    float(os.environ.get("QRC_YAW", "0.0")),
                ],
                dtype=np.float32,
            )

            joint_position = (data.qpos[7:] - DEFAULT).astype(np.float32)
            joint_velocity = (data.qvel[6:] * 0.05).astype(np.float32)
            angular_velocity = (data.qvel[3:6] * 0.25).astype(np.float32)

            obs[:3] = angular_velocity
            obs[3:6] = gravity_orientation(data.qpos[3:7])
            obs[6:9] = command * CMD_SCALE
            obs[9:21] = joint_position
            obs[21:33] = joint_velocity
            obs[33:45] = action

            result = session.run(
                None,
                {input_name: obs[None, :].astype(np.float32)},
            )
            action = np.asarray(result[0], dtype=np.float32).reshape(12)
            target = action * ACTION_SCALE + DEFAULT

        if step % render_interval == 0:
            camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.32]
            renderer.update_scene(data, camera=camera)
            writer.send(renderer.render())
            frames += 1

        if step % 1000 == 0:
            local_velocity = rotate_inverse(
                data.qpos[3:7],
                data.qvel[:3],
            )
            print(
                f"time={sim_time:5.2f}s "
                f"x={data.qpos[0]:+.3f} "
                f"height={data.qpos[2]:.3f} "
                f"vx={local_velocity[0]:+.3f}"
            )
finally:
    writer.close()
    renderer.close()

summary = {
    "status": "PASS",
    "policy": str(POLICY),
    "video": str(OUTPUT),
    "duration_seconds": DURATION,
    "frames": frames,
    "label": os.environ.get("QRC_LABEL", "unnamed"),
    "terrain": XML.name,
    "command_x": float(command[0]),
    "command_y": float(command[1]),
    "command_yaw": float(command[2]),
    "forward_distance": float(data.qpos[0] - start_position[0]),
    "minimum_base_height": minimum_height,
    "final_base_height": float(data.qpos[2]),
    "video_size_bytes": OUTPUT.stat().st_size,
}
SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print(json.dumps(summary, indent=2))
print("GO2 POLICY VIDEO PASS")
