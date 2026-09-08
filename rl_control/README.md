# RL Control

Reinforcement-learning control group workspace.

Responsibilities:

- Learn RL basics and record notes.
- Reproduce a quadruped locomotion baseline.
- Design observation, action, and reward.
- Provide a policy wrapper for simulation.
- Evaluate policies with consistent metrics.

Current integrated route:

1. Reproduce the pinned official `Unitree-Go2-Velocity` baseline.
2. Fine-tune `Unitree-Go2-Campus-Velocity` without changing its 45-observation/12-action policy interface.
3. Use `go2_foundation` to turn mission routes into bounded body-velocity commands and enforce safety checks.
4. Run `Unitree-Go2-Campus-Route-Eval` for physical stepping, collision and stall evidence.
5. Preserve at least two seeds, checkpoints, logs, metrics, videos and SHA-256 manifests.

Platform instructions: `../docs/training/SELF5000_GO2_CAMPUS.md`.
