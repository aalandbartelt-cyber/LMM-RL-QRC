# Learning References

Technical references collected for simulation, decision, and reinforcement-learning work.

## Files

| File | Recommended Readers | Why It Matters |
|---|---|---|
| `参考.md` | All groups | Collected links for RL, Go2, RoboGauge, SWAP, LeggedManip, and related open-source projects |
| `2604.08508v2.pdf` | Simulation, decision, RL | Sumo paper; useful for the layered idea: low-level learned control, high-level planning/MPC |
| `面向复杂地形的四足机器人RL-MPC分层运动控制方法研究_王留东.pdf` | RL group | RL-MPC hierarchical control, reward design, domain randomization, velocity/attitude evaluation |

## Suggested Reading Order

1. Read `参考.md` to understand the project map.
2. RL group: read the RL-MPC paper sections on hierarchical control, observation/action, domain randomization, reward design, and experiments.
3. Simulation group: read Sumo introduction and experiment sections to learn how tasks and success metrics are presented.
4. Decision group: read Sumo for the idea that high-level planning should steer a stable low-level policy rather than directly controlling joints.
