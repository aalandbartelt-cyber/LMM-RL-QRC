# Go2 Formal Evaluation Matrix Design

Date: 2026-09-11
Status: Approved for specification review

## Objective

Compare the original `model_5001.pt` checkpoint and both independently trained
`model_5601.pt` checkpoints under reproducible evaluation conditions.  The
result must distinguish locomotion tracking improvements from regressions in
falls, collisions, energy use, and difficult-terrain success.

## Evaluated Policies

The launcher resolves and records exactly three checkpoints:

1. The newest `stage3_control` checkpoint named `model_5001.pt`.
2. The curriculum-v2 seed-1 checkpoint named `model_5601.pt`.
3. The curriculum-v2 seed-2 checkpoint named `model_5601.pt`.

Each path, training-run name, byte size, and SHA-256 digest is written to the
experiment manifest before evaluation starts.  The two curriculum checkpoint
digests must differ.

## Fixed Protocol

- Evaluation seeds: `20260911`, `20260912`, and `20260913`.
- Terrains: flat, slope, rough slope, stairs up, stairs down, obstacles, wave.
- Environments per terrain: 256.
- Duration per terrain: 20 seconds.
- Command: `vx=0.5 m/s`, `vy=0`, `yaw=0`.
- Randomization: disabled by the existing evaluator.
- Evidence: per-terrain JSON, per-run CSV/JSON summaries, complete logs, launch
  metadata, checkpoint hashes, repository commit, and final SHA-256 manifest.

This produces 9 policy/seed runs and 63 terrain evaluations.  All policy/seed
runs use identical terrain and command settings.

## Approaches Considered

### Selected: two persistent serial workers

One detached worker is assigned to each physical GPU.  Each worker runs its
allocated policy/seed jobs serially, while the two workers operate concurrently.
This keeps each Isaac Gym process isolated to one GPU, prevents memory overlap,
survives terminal disconnection, and completes the matrix with balanced use of
the two A100 GPUs.

### Rejected: nine manually launched jobs

Manual launch offers no implementation cost but creates a high risk of mixing
checkpoints, output directories, GPU assignments, or seeds.  It also makes
recovery and provenance harder.

### Rejected: multi-GPU execution for each evaluation

Each evaluation already fits on one A100.  Distributed evaluation adds process
coordination without improving statistical coverage and would prevent two
independent jobs from running concurrently.

## Seed Smoke Gate

Before the matrix begins, GPU 0 runs one short flat-ground evaluation with seed
`20260911`.  The gate passes only when:

- the process exits successfully;
- the log contains `Setting seed: 20260911`;
- the result JSON contains numeric seed `20260911`;
- the result contains the terrain PASS marker.

Failure stops the launcher before any formal matrix job starts.  This verifies
the evaluator fix in commit `7897e16` at runtime rather than relying only on a
source-code check.

## Scheduling and Data Flow

The launcher builds a deterministic ordered list of nine jobs, distributes them
round-robin between GPU 0 and GPU 1, and writes each job to:

```text
go2_formal_eval_<UTC timestamp>/
  manifest.txt
  seed_smoke/
  baseline5001/eval_seed_<seed>/
  curriculum_seed1_5601/eval_seed_<seed>/
  curriculum_seed2_5601/eval_seed_<seed>/
  workers/gpu0.log
  workers/gpu1.log
```

Each worker invokes the existing evaluator in `QRC_RUN_ALL=1` mode.  A job is
complete only when its log has `GO2 MULTI-TERRAIN BENCHMARK PASS` and all seven
JSON result files plus both aggregate files exist.

## Failure Handling and Resume

- A missing checkpoint, evaluator, environment, or GPU fails before launch.
- A failed job stops only its assigned worker and writes a failure marker; the
  other GPU worker may finish its current queue.
- A status command reports worker PIDs, completed job count, failure markers,
  and the latest meaningful log lines without using `watch` under the Conda
  library path.
- Relaunching against an existing output root skips only jobs that already meet
  the complete-file and PASS-marker gate.  Partial jobs are rerun in place after
  their stale per-terrain results are removed by the evaluator.

## Aggregation and Decision Rule

After both workers finish, aggregation verifies all 63 terrain JSON files and
produces a policy-level table with mean and standard deviation across evaluation
seeds.  Primary ranking is lexicographic:

1. higher mean success rate;
2. fewer total falls;
3. lower collision sample rate;
4. lower velocity-tracking RMSE;
5. lower mechanical power and mean tilt as tie-breakers.

No scalar score hides a safety regression.  Per-terrain results remain visible,
and the selected policy must be accompanied by any observed specialist strengths
or weaknesses.

## Tests and Acceptance

Portable tests verify checkpoint-resolution constraints, the three fixed seeds,
the seven fixed terrains, isolated `CUDA_VISIBLE_DEVICES` use, smoke-gate markers,
unique output paths, failure markers, and the 63-result aggregation requirement.
Shell syntax and Python compilation must pass locally.  Runtime acceptance needs
the seed smoke gate, two successful worker completion markers, nine benchmark
PASS markers, and the final aggregate plus SHA-256 manifest.

