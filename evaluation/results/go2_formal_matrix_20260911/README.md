# Go2 Formal Matrix Evidence — 2026-09-11

This directory is the versioned aggregate snapshot from the completed formal
evaluation matrix stored on the team's persistent Self-5000 workspace at:

```text
/home/jovyan/go2_work/go2_formal_eval_20260911T113911Z
```

Evaluation scope:

- Policies: `baseline5001`, `curriculum_seed1_5601`, `curriculum_seed2_5601`
- Evaluation seeds: `20260911`, `20260912`, `20260913`
- Terrains: flat, slope, rough slope, stairs up, stairs down, obstacles, wave
- Total terrain evaluations: 63
- Completion markers: `WORKER_0_PASS` and `FORMAL_MATRIX_PASS`

`overall_summary.csv` contains the policy-level ranking. `terrain_summary.csv`
contains the 21 policy-by-terrain aggregates copied from the generated formal
terrain analysis. Raw per-step simulator output remains on the persistent cloud
workspace and is not duplicated here.

Evidence boundary: the metrics are Isaac Gym batch simulation results. They are
not physical-robot test results.
