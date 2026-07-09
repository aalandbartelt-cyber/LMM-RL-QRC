"""Mock locomotion policy used before a trained RL policy is ready."""


class MockLocomotionPolicy:
    def __init__(self):
        self.name = "mock_policy_v0"

    def reset(self):
        return None

    def act(self, observation):
        obstacle_distance = observation.get("obstacle_distance", 999.0)
        if obstacle_distance < 0.5:
            return {
                "vx": 0.0,
                "vy": 0.0,
                "yaw_rate": 0.0,
                "speed_scale": 0.0,
                "gait_mode": "stop",
                "confidence": 1.0,
                "safety_flag": "stop_obstacle_too_close",
            }

        command = observation.get("command_velocity", [0.3, 0.0, 0.0])
        return {
            "vx": float(command[0]),
            "vy": float(command[1]) if len(command) > 1 else 0.0,
            "yaw_rate": float(command[2]) if len(command) > 2 else 0.0,
            "speed_scale": 1.0,
            "gait_mode": "walk",
            "confidence": 0.8,
            "safety_flag": "ok",
        }


if __name__ == "__main__":
    policy = MockLocomotionPolicy()
    print(policy.act({"command_velocity": [0.3, 0.0, 0.1], "obstacle_distance": 1.0}))
    print(policy.act({"command_velocity": [0.3, 0.0, 0.1], "obstacle_distance": 0.2}))
