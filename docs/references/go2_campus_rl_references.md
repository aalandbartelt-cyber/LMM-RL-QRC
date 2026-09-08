# Go2 Campus RL Reference Baseline

The implementation uses upstream projects as pinned contracts, not as vague inspiration.

| Project | Pinned input | How it is used |
|---|---|---|
| [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) | commit `1330cb0d4236c4abd6bf1efec05a4fce7ad79678` | Official Go2 robot asset, 45-value actor observation, 12 joint actions, velocity task and play/export flow |
| [Isaac Lab](https://github.com/isaac-sim/IsaacLab) | commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77` | Manager-based environment configuration, terrain generators, simulation launch and video wrapper pattern |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | package `2.3.1` | PPO runner used by the pinned Unitree task |
| [legged_gym](https://github.com/leggedrobotics/legged_gym) | historical local archive | Comparison with the July Isaac Gym terrain curriculum and checkpoint evidence; not checkpoint-compatible with Isaac Lab |

## Adopted Decisions

- Preserve the upstream Go2 policy interface so a campus curriculum does not silently change deployment inputs or outputs.
- Inherit the official robot task and override only terrain sampling, command ranges, selected rewards and push intensity.
- Run two independent seeds on two physical GPUs. Each simulator process sees one logical `cuda:0`, avoiding unsupported cross-process assumptions.
- Keep route feedback above locomotion. A blind 45-value proprioceptive policy cannot infer walls or a global patrol route.
- Export hashes, run metadata, exact upstream revisions and complete checkpoint directories with every final result.

## Explicit Non-Claims

- A kinematic preview that writes root pose is not RL evidence.
- Terrain benchmark success is not autonomous navigation success.
- An Isaac Gym checkpoint is not assumed loadable by the Isaac Lab task.
- Passing portable tests does not replace the paid-machine Isaac runtime smoke matrix.
