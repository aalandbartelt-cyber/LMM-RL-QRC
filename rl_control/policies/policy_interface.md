# Policy Interface v0.1

Initial policy input:

```json
{
  "base_velocity": [0.0, 0.0, 0.0],
  "angular_velocity": [0.0, 0.0, 0.0],
  "roll_pitch_yaw": [0.0, 0.0, 0.0],
  "command_velocity": [0.3, 0.0, 0.0],
  "joint_position": [],
  "joint_velocity": [],
  "obstacle_distance": 999.0,
  "terrain_level": 0,
  "previous_action": null
}
```

Initial policy output:

```json
{
  "vx": 0.3,
  "vy": 0.0,
  "yaw_rate": 0.0,
  "speed_scale": 1.0,
  "gait_mode": "walk",
  "confidence": 0.8,
  "safety_flag": "ok"
}
```
