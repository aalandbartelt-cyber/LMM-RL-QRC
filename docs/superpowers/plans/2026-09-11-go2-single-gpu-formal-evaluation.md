# Go2 Single-GPU Formal Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parameterize the existing formal Go2 evaluation workflow so the unchanged nine-job matrix can run safely on one explicitly selected physical GPU while retaining the default two-GPU behavior.

**Architecture:** Keep one launcher and make its physical-GPU list data-driven through `QRC_EVAL_GPUS`, with validation against `nvidia-smi`, dynamic queues, workers, markers, and coordinator acceptance. Persist the selected topology in `manifest.txt`; make the read-only status script consume that field and default to `0 1` for legacy output roots.

**Tech Stack:** Bash 4+, Python 3.8 `unittest`, Git, `nvidia-smi`

---

## File Structure

- Modify `rl_control/tests/test_formal_evaluation_matrix.py`: executable shell-contract tests for parsing, one-GPU distribution, coordinator markers, manifest persistence, and status fallback.
- Modify `rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh`: selected-GPU parsing/validation and dynamic orchestration.
- Modify `rl_control/legacy_isaacgym/show_formal_evaluation_status.sh`: selected-GPU discovery and dynamic status reporting.
- Modify `docs/superpowers/plans/2026-09-11-go2-single-gpu-formal-evaluation.md`: check off each verified step as it is completed.

### Task 1: Specify selected-GPU behavior with failing tests

**Files:**
- Modify: `rl_control/tests/test_formal_evaluation_matrix.py`

- [x] **Step 1: Add a shell runner and launcher validation tests**

Add a helper that invokes Bash functions without reaching the paid evaluation path by sourcing only launcher definitions under `QRC_SOURCE_ONLY=1`:

```python
import os
import shlex
import subprocess

def run_bash(script: str, *, env=None):
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
        exports = "; ".join(
            f"export {name}={shlex.quote(value)}" for name, value in env.items()
        )
        script = f"{exports}; {script}"
    return subprocess.run(
        ["bash", "-c", script], cwd=REPOSITORY_ROOT,
        env=merged_env, text=True, encoding="utf-8", errors="replace",
        capture_output=True, check=False,
    )

def source_launcher(command: str, *, env=None):
    return run_bash(
        'export QRC_SOURCE_ONLY=1; source '
        '"rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh"; '
        + command,
        env=env,
    )
```

Add tests that source the launcher, call `parse_gpu_selection`, and assert:

```python
def test_gpu_selection_defaults_to_two_physical_gpus(self):
    result = run_bash(
        'unset QRC_EVAL_GPUS; export QRC_SOURCE_ONLY=1; '
        'source "rl_control/legacy_isaacgym/'
        'run_formal_evaluation_matrix.sh"; parse_gpu_selection; '
        'declare -p PHYSICAL_GPUS'
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('([0]="0" [1]="1")', result.stdout)

def test_gpu_selection_accepts_one_gpu(self):
    result = source_launcher(
        "parse_gpu_selection; declare -p PHYSICAL_GPUS",
        env={"QRC_EVAL_GPUS": "1"},
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertIn('([0]="1")', result.stdout)

def test_gpu_selection_rejects_invalid_values(self):
    for value in ("", "0 0", "gpu0", "-1", "0 1 2", "0\n1 2"):
        result = source_launcher(
            "parse_gpu_selection", env={"QRC_EVAL_GPUS": value}
        )
        self.assertNotEqual(result.returncode, 0, value)

def test_gpu_selection_rejects_an_unavailable_index(self):
    result = source_launcher(
        "parse_gpu_selection; validate_gpu_selection_availability 1",
        env={"QRC_EVAL_GPUS": "1"},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unavailable", result.stderr.lower())

def test_gpu_selection_rejects_an_overflowing_index(self):
    result = source_launcher(
        "parse_gpu_selection; validate_gpu_selection_availability 1",
        env={"QRC_EVAL_GPUS": "18446744073709551616"},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unavailable", result.stderr.lower())
```

On Windows, custom values are also prefixed as shell-quoted `export`
assignments because the WSL app alias does not reliably forward `bash -c`
positional arguments or arbitrary Windows environment additions.

- [x] **Step 2: Add failing one-GPU queue and manifest tests**

Source the launcher with `QRC_SOURCE_ONLY=1`, provide synthetic run names, call
`write_queues`, and assert that `gpu1.queue` contains exactly nine records,
contains nine unique policy/seed pairs, and no `gpu0.queue` is created when
`PHYSICAL_GPUS=(1)`. Create the temporary directory under `REPOSITORY_ROOT`,
pass its repository-relative POSIX path to Bash, and set
`BASE_RUN=base`, `CURRICULUM1_RUN=curriculum1`, and
`CURRICULUM2_RUN=curriculum2` before calling the function. Add a contract
assertion that `write_manifest` emits:

```bash
physical_gpus=${PHYSICAL_GPUS[*]}
```

- [x] **Step 3: Add failing dynamic worker/coordinator and status tests**

Assert launcher source iterates `"${PHYSICAL_GPUS[@]}"`, seed smoke uses `"${PHYSICAL_GPUS[0]}"`, coordinator derives the selected list from `manifest.txt`, and hard-coded `WORKER_0_PASS && WORKER_1_PASS` logic is absent. Assert status source reads `physical_gpus`, has `PHYSICAL_GPUS=(0 1)` as its legacy fallback, and builds PID/marker names dynamically.

- [x] **Step 4: Run focused tests and confirm RED**

Run:

```bash
python -m unittest rl_control.tests.test_formal_evaluation_matrix -v
```

Expected: new tests fail because `parse_gpu_selection`, source-only mode, dynamic queues, manifest topology, and dynamic status behavior do not yet exist; existing aggregation tests still pass.

- [x] **Step 5: Commit the tests**

```bash
git add rl_control/tests/test_formal_evaluation_matrix.py
git commit -m "Test single-GPU formal evaluation orchestration"
```

### Task 2: Implement dynamic launcher orchestration

**Files:**
- Modify: `rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh`

- [x] **Step 1: Parse and validate the selected GPU list**

Add the backward-compatible default and a pure parser:

```bash
QRC_EVAL_GPUS="${QRC_EVAL_GPUS-0 1}"
PHYSICAL_GPUS=()

parse_gpu_selection() {
    [[ -n "${QRC_EVAL_GPUS//[[:space:]]/}" ]] || fail "QRC_EVAL_GPUS must select one or two physical GPU indices"
    [[ "${QRC_EVAL_GPUS}" != *$'\n'* && "${QRC_EVAL_GPUS}" != *$'\r'* ]] || fail "QRC_EVAL_GPUS must be a single space-separated line"
    read -r -a PHYSICAL_GPUS <<<"${QRC_EVAL_GPUS}"
    (( ${#PHYSICAL_GPUS[@]} >= 1 && ${#PHYSICAL_GPUS[@]} <= 2 )) || fail "QRC_EVAL_GPUS must contain one or two physical GPU indices"
    local index gpu seen=" "
    for index in "${!PHYSICAL_GPUS[@]}"; do
        gpu="${PHYSICAL_GPUS[${index}]}"
        [[ "${gpu}" =~ ^[0-9]+$ ]] || fail "Invalid physical GPU index: ${gpu}"
        while [[ "${#gpu}" -gt 1 && "${gpu}" == 0* ]]; do
            gpu="${gpu#0}"
        done
        [[ "${seen}" != *" ${gpu} "* ]] || fail "Duplicate physical GPU index: ${gpu}"
        PHYSICAL_GPUS[${index}]="${gpu}"
        seen+="${gpu} "
    done
}
```

After `nvidia-smi -L`, validate every selected index against the reported
physical count. Compare digit-string lengths before numeric comparison so a
hostile oversized input cannot overflow Bash arithmetic and alias GPU 0. Do
not silently drop a failing GPU or fall back to another topology.

- [x] **Step 2: Persist topology and parameterize smoke/queues**

Write `physical_gpus=${PHYSICAL_GPUS[*]}` to the manifest. Pass `${PHYSICAL_GPUS[0]}` into `seed_smoke_gate`, and set `CUDA_VISIBLE_DEVICES` from that argument. Rebuild queues by truncating one `gpu<index>.queue` per selected GPU, then use `index % ${#PHYSICAL_GPUS[@]}` to distribute all nine unchanged job records.

- [x] **Step 3: Parameterize worker lifecycle and coordinator acceptance**

Iterate selected physical indices when clearing markers and launching workers. Make `ensure_workers_stopped` inspect PID files associated with the current manifest selection plus any existing `workers/gpu*.pid` so a topology change cannot bypass a live recorded worker. In `coordinate_mode`, read `physical_gpus` from the manifest (fallback `0 1`) and, on every loop, fail if any selected `WORKER_<gpu>_FAILED` exists or proceed only when every selected `WORKER_<gpu>_PASS` exists.

- [x] **Step 4: Add definition-only source mode**

Guard CLI dispatch so tests can load functions without starting evaluation:

```bash
if [[ "${QRC_SOURCE_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi
```

- [x] **Step 5: Run focused tests and confirm GREEN**

Run:

```bash
python -m unittest rl_control.tests.test_formal_evaluation_matrix -v
```

Expected: launcher tests pass; status-specific tests may remain RED until Task 3.

- [x] **Step 6: Commit launcher implementation**

```bash
git add rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh
git commit -m "Support selected GPUs in formal evaluation launcher"
```

### Task 3: Make status reporting topology-aware

**Files:**
- Modify: `rl_control/legacy_isaacgym/show_formal_evaluation_status.sh`

- [x] **Step 1: Read selected topology with legacy fallback**

Initialize `PHYSICAL_GPUS=(0 1)`, then if `manifest.txt` contains a non-empty `physical_gpus=` entry, split it into the array. Treat the manifest as data only; do not `source` it.

- [x] **Step 2: Report dynamic PIDs and markers**

Loop over `"${PHYSICAL_GPUS[@]}"` for `gpu<index>.pid`, `WORKER_<index>_PASS`, and `WORKER_<index>_FAILED`, then report the coordinator and unchanged matrix markers. Preserve `kill -0` as the only process signal and keep the command read-only.

- [x] **Step 3: Run focused tests and confirm GREEN**

Run:

```bash
python -m unittest rl_control.tests.test_formal_evaluation_matrix -v
```

Expected: all formal-matrix tests pass.

- [x] **Step 4: Commit status implementation**

```bash
git add rl_control/legacy_isaacgym/show_formal_evaluation_status.sh
git commit -m "Report selected formal evaluation GPUs"
```

### Task 4: Verify, document completion, and integrate locally

**Files:**
- Modify: `docs/superpowers/plans/2026-09-11-go2-single-gpu-formal-evaluation.md`

- [x] **Step 1: Run the complete Python test suite**

Run:

```bash
cd rl_control
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v
```

Expected: all applicable tests pass; the known Windows symlink test may skip.

- [x] **Step 2: Run compilation and shell syntax checks**

Run:

```bash
python -m py_compile evaluation/scripts/aggregate_formal_matrix.py rl_control/tests/test_formal_evaluation_matrix.py
bash -n rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh
bash -n rl_control/legacy_isaacgym/show_formal_evaluation_status.sh
```

Expected: every command exits 0 with no syntax error.

- [x] **Step 3: Review requirements and diff**

Run `git diff main...HEAD --check`, inspect the full diff, and confirm: default `0 1`; explicit one-GPU support; invalid/unavailable rejection; per-GPU preflight; first-GPU smoke; nine jobs once; dynamic workers/coordinator/status; manifest persistence; legacy status fallback; no metric, policy, checkpoint, seed, terrain, ranking, or aggregation changes.

- [x] **Step 4: Commit the checked plan and merge locally to main**

Check all completed boxes, commit the plan update, switch to the primary checkout, and fast-forward `main` to the verified feature branch. Re-run focused tests and Bash syntax checks on `main` after integration.

- [x] **Step 5: Publish source without starting evaluation**

Push the verified `main` when GitHub is reachable. If the paid container cannot use Git, export a binary patch from the remote base commit and provide checksum-verified transfer/apply commands. Do not run the formal launcher locally or on the paid host during this step.
