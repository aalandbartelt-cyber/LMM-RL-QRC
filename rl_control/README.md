# RL Control

Reinforcement-learning control group workspace.

Responsibilities:

- Learn RL basics and record notes.
- Reproduce a quadruped locomotion baseline.
- Design observation, action, and reward.
- Provide a policy wrapper for simulation.
- Evaluate policies with consistent metrics.

Current recommended route:

1. Start with high-level velocity commands.
2. Reproduce or call an existing Go2 locomotion baseline.
3. Add RL-MPC-like safe parameters such as speed scale, stability weight, and gait mode.
4. Avoid direct low-level joint control until simulation and safety supervisor are ready.
