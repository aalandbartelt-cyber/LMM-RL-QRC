# Go2 Formal Evaluation Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable two-A100 launcher and strict aggregator for the approved three-policy, three-seed, seven-terrain Go2 evaluation matrix.

**Architecture:** A Bash launcher owns platform validation, checkpoint resolution, the runtime seed smoke gate, deterministic two-worker scheduling, and completion coordination. A separate Bash status command reads artifacts without mutating the run. A pure-Python aggregator validates all 63 terrain JSON files, summarizes metrics across evaluation seeds, applies the approved lexicographic ranking, and writes a SHA-256 manifest.

**Tech Stack:** Bash 4+, Python 3.8 standard library, Isaac Gym, PyTorch, existing `evaluate_go2_terrain.py`, `unittest`, Git.

---

## File Structure

- Create `evaluation/scripts/aggregate_formal_matrix.py`: validate and summarize the complete matrix; no Isaac Gym import.
- Create `rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh`: preflight, seed smoke, queue creation, detached GPU workers, coordinator, resume.
- Create `rl_control/legacy_isaacgym/show_formal_evaluation_status.sh`: read-only progress and failure reporting without `watch`.
- Create `rl_control/tests/test_formal_evaluation_matrix.py`: behavioral aggregation tests plus launcher/status contract tests.
- Modify `evaluation/README.md`: document the formal protocol, launch, status, acceptance markers, and artifacts.

### Task 1: Strict Matrix Aggregator

**Files:**
- Create: `evaluation/scripts/aggregate_formal_matrix.py`
- Create: `rl_control/tests/test_formal_evaluation_matrix.py`

- [ ] **Step 1: Write failing aggregation tests**

Create tests that synthesize exactly 63 JSON files under the approved directory
layout.  Each JSON contains `terrain`, `seed`, `success_rate`, `fall_count`,
`collision_sample_rate`, `tracking_rmse_mps`, `mean_mechanical_power_w`, and
`mean_tilt_rad`.  Assert that:

```python
rows = module.collect_results(root)
self.assertEqual(len(rows), 63)
summary = module.summarize(rows)
self.assertEqual([row["policy"] for row in summary], [
    "baseline5001",
    "curriculum_seed1_5601",
    "curriculum_seed2_5601",
])
self.assertEqual(summary[0]["rank"], 1)
self.assertEqual(summary[0]["evaluation_seed_count"], 3)
self.assertEqual(summary[0]["terrain_evaluation_count"], 21)
```

Delete one terrain JSON and assert `collect_results` raises a `ValueError` whose
message contains `Expected 63 terrain results`.

- [ ] **Step 2: Run the tests and verify RED**

Run from `rl_control`:

```bash
PYTHONPATH=. python -m unittest tests.test_formal_evaluation_matrix -v
```

Expected: import/file failure because `aggregate_formal_matrix.py` does not exist.

- [ ] **Step 3: Implement the pure-Python aggregator**

Implement the aggregator with these constants and functions:

```python
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev

POLICIES = (
    "baseline5001",
    "curriculum_seed1_5601",
    "curriculum_seed2_5601",
)
EVALUATION_SEEDS = (20260911, 20260912, 20260913)
TERRAINS = (
    "flat", "slope", "rough_slope", "stairs_up",
    "stairs_down", "obstacles", "wave",
)
METRICS = (
    ("success_rate", "success_rate"),
    ("collision_sample_rate", "collision_sample_rate"),
    ("tracking_rmse_mps", "tracking_rmse_mps"),
    ("mean_mechanical_power_w", "mechanical_power_w"),
    ("mean_tilt_rad", "tilt_rad"),
)

def collect_results(root: Path) -> list[dict]:
    rows = []
    missing = []
    for policy in POLICIES:
        for seed in EVALUATION_SEEDS:
            for terrain in TERRAINS:
                path = root / policy / f"eval_seed_{seed}" / f"{terrain}.json"
                if not path.is_file():
                    missing.append(path.relative_to(root).as_posix())
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("terrain") != terrain or payload.get("seed") != seed:
                    raise ValueError(f"Result metadata mismatch: {path}")
                for metric in (*(source for source, _ in METRICS), "fall_count"):
                    if metric not in payload:
                        raise ValueError(f"Missing metric {metric}: {path}")
                payload["policy"] = policy
                rows.append(payload)
    if len(rows) != 63:
        raise ValueError(
            f"Expected 63 terrain results, found {len(rows)}; missing={missing}"
        )
    return rows

def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for policy in POLICIES:
        seed_rows = []
        for seed in EVALUATION_SEEDS:
            selected = [
                row for row in rows
                if row["policy"] == policy and row["seed"] == seed
            ]
            if len(selected) != 7:
                raise ValueError(f"Expected seven terrains for {policy} seed {seed}")
            stats = {
                source: mean(row[source] for row in selected)
                for source, _ in METRICS
            }
            stats["fall_count"] = sum(row["fall_count"] for row in selected)
            seed_rows.append(stats)
        result = {
            "policy": policy,
            "evaluation_seed_count": len(seed_rows),
            "terrain_evaluation_count": len(seed_rows) * len(TERRAINS),
            "total_falls": sum(row["fall_count"] for row in seed_rows),
            "mean_falls_per_seed": mean(row["fall_count"] for row in seed_rows),
            "std_falls_per_seed": pstdev(row["fall_count"] for row in seed_rows),
        }
        for source, output in METRICS:
            result[f"mean_{output}"] = mean(row[source] for row in seed_rows)
            result[f"std_{output}"] = pstdev(row[source] for row in seed_rows)
        summaries.append(result)
    summaries.sort(key=lambda row: (
        -row["mean_success_rate"],
        row["total_falls"],
        row["mean_collision_sample_rate"],
        row["mean_tracking_rmse_mps"],
        row["mean_mechanical_power_w"],
        row["mean_tilt_rad"],
    ))
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries

def write_outputs(root: Path, summary: list[dict]) -> None:
    (root / "formal_policy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    with (root / "formal_policy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

def write_sha256_manifest(root: Path) -> Path:
    manifest = root / "SHA256SUMS"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
```

`collect_results` must derive the expected path from every
policy/seed/terrain combination, load it, and reject mismatched `seed` or
`terrain` fields.  `summarize` must first average each metric across the seven
terrains within each evaluation seed, then compute policy mean and population
standard deviation across three seeds.  It must sum all falls across the 21
terrain evaluations.  Sort using:

```python
key=lambda row: (
    -row["mean_success_rate"],
    row["total_falls"],
    row["mean_collision_sample_rate"],
    row["mean_tracking_rmse_mps"],
    row["mean_mechanical_power_w"],
    row["mean_tilt_rad"],
)
```

Write `formal_policy_summary.csv`, `formal_policy_summary.json`, and
`FORMAL_MATRIX_PASS`.  Hash every regular file below the output root except
`SHA256SUMS`, using sorted POSIX relative paths, and write `SHA256SUMS` last.

- [ ] **Step 4: Run tests and verify GREEN**

```bash
cd rl_control
PYTHONPATH=. python -m unittest tests.test_formal_evaluation_matrix -v
```

Expected: aggregation and missing-result tests pass.

- [ ] **Step 5: Commit the aggregator**

```bash
git add evaluation/scripts/aggregate_formal_matrix.py \
  rl_control/tests/test_formal_evaluation_matrix.py
git commit -m "Add strict Go2 matrix aggregator"
```

### Task 2: Two-GPU Resumable Launcher

**Files:**
- Create: `rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh`
- Modify: `rl_control/tests/test_formal_evaluation_matrix.py`

- [ ] **Step 1: Write the failing launcher contract test**

Read the launcher as text and assert all fixed protocol and safety contracts:

```python
self.assertIn('EVALUATION_SEEDS=(20260911 20260912 20260913)', source)
self.assertIn('TERRAINS="flat,slope,rough_slope,stairs_up,stairs_down,obstacles,wave"', source)
self.assertIn('NUM_ENVS="256"', source)
self.assertIn('DURATION_SECONDS="20"', source)
self.assertIn('CUDA_VISIBLE_DEVICES="${GPU}"', source)
self.assertIn('Setting seed: 20260911', source)
self.assertIn('GO2 MULTI-TERRAIN BENCHMARK PASS', source)
self.assertIn('WORKER_${GPU}_PASS', source)
self.assertIn('MATRIX_FAILED', source)
self.assertNotIn('watch ', source)
```

Also assert the launcher references both `model_5001.pt` and `model_5601.pt`,
the aggregation script, and the latest-run pointer file.

- [ ] **Step 2: Run the launcher test and verify RED**

```bash
cd rl_control
PYTHONPATH=. python -m unittest \
  tests.test_formal_evaluation_matrix.FormalLauncherContractTests -v
```

Expected: file-not-found failure for the launcher.

- [ ] **Step 3: Implement validation and checkpoint resolution**

The launcher accepts `start [OUTPUT_ROOT]`, `worker GPU OUTPUT_ROOT QUEUE`, and
`coordinate OUTPUT_ROOT` modes.  In `start`, resolve:

```bash
ROOT="${QRC_WORK_ROOT:-${HOME}/go2_work}"
REPO="${ROOT}/LMM-RL-QRC"
GYM="${ROOT}/go2_rl_gym-master"
ENV_PREFIX="${QRC_LEGACY_ENV:-${HOME}/conda/envs/go2rl}"
EVAL="${REPO}/evaluation/scripts/evaluate_go2_terrain.py"
AGGREGATOR="${REPO}/evaluation/scripts/aggregate_formal_matrix.py"
EVALUATION_SEEDS=(20260911 20260912 20260913)
TERRAINS="flat,slope,rough_slope,stairs_up,stairs_down,obstacles,wave"
NUM_ENVS="256"
DURATION_SECONDS="20"
```

Require both GPU indices to pass a one-visible-device CUDA tensor test.  Resolve
the newest `stage3_control/model_5001.pt`, plus one curriculum-v2 seed-1 and one
seed-2 `model_5601.pt`.  Fail if any path is empty or either curriculum digest is
equal.  Write paths, byte sizes, SHA-256 values, repository commit, protocol,
UTC time, and environment Python path to `manifest.txt`.

- [ ] **Step 4: Implement and verify the runtime seed smoke gate**

Run baseline `model_5001.pt` on GPU 0 with one terrain, 32 environments, five
seconds, and `--seed=20260911`.  Require all three conditions:

```bash
grep -q 'Setting seed: 20260911' "${SMOKE_DIR}/eval.log"
grep -q 'GO2 TERRAIN FLAT PASS' "${SMOKE_DIR}/eval.log"
"${ENV_PREFIX}/bin/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["seed"] == 20260911 else 2)' \
  "${SMOKE_DIR}/flat.json"
```

Write `SEED_SMOKE_PASS` only after these checks.  Any failure exits before queue
creation.

- [ ] **Step 5: Implement deterministic queues and detached workers**

Write nine queue records in evaluation-seed-major, policy-minor order using the
format `policy|seed|load_run|checkpoint`.  Allocate alternating records to
`workers/gpu0.queue` and `workers/gpu1.queue`.

For every record, the worker runs the existing evaluator with one visible GPU,
the fixed protocol, and output path `${OUTPUT_ROOT}/${POLICY}/eval_seed_${SEED}`.
Skip a job only when all seven terrain JSONs, both aggregate files, and the PASS
marker in `eval.log` exist.  On success write `COMPLETE`; on failure write
`FAILED`, write `WORKER_${GPU}_FAILED`, and exit nonzero.  After its queue is
complete, write `WORKER_${GPU}_PASS`.

Launch workers with `nohup` and store `gpu0.pid` and `gpu1.pid`.  Launch a
detached coordinator that waits for either PASS or FAILED marker from each
worker.  If both pass, invoke `aggregate_formal_matrix.py`; otherwise write
`MATRIX_FAILED`.  Store the coordinator PID and write the output root to
`${ROOT}/latest_formal_eval_dir.txt`.

- [ ] **Step 6: Run tests and shell syntax checks**

```bash
cd rl_control
PYTHONPATH=. python -m unittest tests.test_formal_evaluation_matrix -v
bash -n legacy_isaacgym/run_formal_evaluation_matrix.sh
```

Expected: all tests pass and `bash -n` exits zero.

- [ ] **Step 7: Commit the launcher**

```bash
git add rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh \
  rl_control/tests/test_formal_evaluation_matrix.py
git commit -m "Add resumable two-GPU evaluation launcher"
```

### Task 3: Read-Only Status Command

**Files:**
- Create: `rl_control/legacy_isaacgym/show_formal_evaluation_status.sh`
- Modify: `rl_control/tests/test_formal_evaluation_matrix.py`

- [ ] **Step 1: Write the failing status contract test**

Assert the script reads `latest_formal_eval_dir.txt` by default, accepts an
explicit root, reports `COMPLETE`, `FAILED`, `WORKER_0_PASS`, `WORKER_1_PASS`,
`FORMAL_MATRIX_PASS`, uses `kill -0` only for PID status, and contains neither
`watch` nor `kill` without `-0`.

- [ ] **Step 2: Run the status test and verify RED**

```bash
cd rl_control
PYTHONPATH=. python -m unittest \
  tests.test_formal_evaluation_matrix.FormalStatusContractTests -v
```

Expected: file-not-found failure for the status script.

- [ ] **Step 3: Implement the status command**

The command resolves `OUTPUT_ROOT` from its first argument or the latest pointer,
prints UTC time and output root, reports each stored PID with `kill -0`, counts
job `COMPLETE` and `FAILED` files, lists worker/matrix markers, and prints the
last 60 matching lines from worker and evaluation logs:

```bash
grep -H -E \
  'BENCHMARK TERRAIN|GO2 TERRAIN .* PASS|GO2 MULTI-TERRAIN|Traceback|RuntimeError|FAILED' \
  "${OUTPUT_ROOT}"/workers/*.log \
  "${OUTPUT_ROOT}"/*/eval_seed_*/eval.log 2>/dev/null | tail -n 60 || true
```

It must never modify results or terminate a process.

- [ ] **Step 4: Verify tests and syntax**

```bash
cd rl_control
PYTHONPATH=. python -m unittest tests.test_formal_evaluation_matrix -v
bash -n legacy_isaacgym/show_formal_evaluation_status.sh
```

Expected: all tests pass and syntax check exits zero.

- [ ] **Step 5: Commit the status command**

```bash
git add rl_control/legacy_isaacgym/show_formal_evaluation_status.sh \
  rl_control/tests/test_formal_evaluation_matrix.py
git commit -m "Add formal evaluation status command"
```

### Task 4: Runbook, Full Verification, and Push

**Files:**
- Modify: `evaluation/README.md`

- [ ] **Step 1: Document exact platform workflow**

Add commands for pulling the required commit, starting the matrix, checking
status, identifying PASS/FAIL markers, resuming with an explicit existing output
root, and displaying `formal_policy_summary.csv`.  State that the diagnostic
batch with JSON seed `20260911` but runtime seed `1` is not formal evidence.

- [ ] **Step 2: Run the complete portable verification**

```bash
cd rl_control
PYTHONPATH=. python -m unittest discover -s tests -v
cd ..
python -m py_compile evaluation/scripts/aggregate_formal_matrix.py \
  evaluation/scripts/evaluate_go2_terrain.py \
  rl_control/tests/test_formal_evaluation_matrix.py
bash -n rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh
bash -n rl_control/legacy_isaacgym/show_formal_evaluation_status.sh
git diff --check
```

Expected: 0 failures, only the existing Windows symlink test may skip; both
Python compilation and Shell syntax checks exit zero; no whitespace errors.

- [ ] **Step 3: Commit documentation and push**

```bash
git add evaluation/README.md
git commit -m "Document formal Go2 evaluation workflow"
git push origin main
git status --short --branch
```

Expected: local `main` equals `origin/main` and the worktree is clean.

- [ ] **Step 4: Platform acceptance gate**

On the two-A100 host, pull the implementation commit, start the launcher, and
require these final artifacts before interpreting model rankings:

```text
seed_smoke/SEED_SMOKE_PASS
workers/WORKER_0_PASS
workers/WORKER_1_PASS
FORMAL_MATRIX_PASS
formal_policy_summary.csv
formal_policy_summary.json
SHA256SUMS
```

The matrix is incomplete if any `FAILED`, `WORKER_*_FAILED`, or `MATRIX_FAILED`
marker exists.
