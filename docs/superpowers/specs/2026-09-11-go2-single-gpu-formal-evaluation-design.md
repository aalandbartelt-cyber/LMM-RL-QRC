# Go2 Single-GPU Formal Evaluation Design

**Date:** 2026-09-11
**Status:** Approved for implementation

## Context

The formal Go2 evaluation launcher currently requires two physical GPUs and
splits nine policy/seed jobs between them. Two separately allocated two-A100
containers each exposed both cards through `nvidia-smi`, but only one card per
container could establish a CUDA context. The failed card changed from physical
index 1 to physical index 0 between allocations. In both cases the launcher
stopped at its CUDA preflight before the seed smoke test or evaluation workers
started.

The formal experiment itself does not require concurrent GPUs. Its scientific
unit is a policy/checkpoint and evaluation-seed pair, and all nine jobs are
independent. Running those jobs serially on one A100 changes wall-clock time but
does not change the policies, seeds, terrains, commands, metrics, acceptance
rules, or ranking.

## Goal

Allow the existing formal matrix launcher and status command to run safely on
one selected physical GPU while preserving the current two-GPU behavior and all
formal evaluation invariants.

## Considered Approaches

1. **Parameterize the existing launcher (selected).** Add an explicit physical
   GPU list, distribute the same nine jobs across that list, and make worker and
   coordinator markers dynamic. This keeps one implementation of the protocol
   and supports both one- and two-GPU execution.
2. **Create a separate single-GPU launcher.** This is initially simple but
   duplicates model discovery, seed smoke, resume, manifest, aggregation, and
   failure handling. The two launchers could silently diverge.
3. **Automatically fall back when one of two GPUs fails.** This saves a manual
   restart but hides a platform fault and makes the executed topology less
   explicit. Formal runs should fail closed unless the operator deliberately
   selects a single GPU.

## Interface

The launcher reads a space-separated physical GPU list from
`QRC_EVAL_GPUS`:

```bash
QRC_EVAL_GPUS=0 \
  bash rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh start
```

The default remains `0 1`, preserving the existing two-GPU workflow. Supported
values contain one or two unique, non-negative integer indices. Empty values,
duplicates, non-integers, more than two indices, and indices not exposed by
`nvidia-smi` fail before any evaluation work starts.

There is no automatic degradation from two GPUs to one. An operator must set
`QRC_EVAL_GPUS=0` explicitly on a one-GPU container.

## Launcher Behavior

The launcher will:

1. Parse and validate `QRC_EVAL_GPUS` before creating workers.
2. Run the existing isolated Isaac Gym/PyTorch CUDA matrix preflight once for
   every selected physical GPU.
3. Preserve the five-second seed smoke gate on the first selected GPU.
4. Write `physical_gpus=<space-separated-list>` to `manifest.txt`.
5. Distribute the unchanged nine policy/seed jobs round-robin over the selected
   GPUs. A one-GPU run places all nine jobs in one queue and executes them
   serially.
6. Start exactly one worker per selected GPU and record its PID, log, pass
   marker, or failure marker using the physical index in the filename.
7. Have the coordinator wait for exactly the selected worker markers before
   invoking the unchanged strict aggregator.

Resume remains artifact-based: complete policy/seed jobs are verified and
skipped, and partial jobs are rerun. Starting an existing output root with a
different explicit GPU list rewrites the worker queues and manifest after first
confirming no recorded worker process is still active.

## Status and Acceptance

The read-only status command obtains the active physical GPU list from
`manifest.txt`. It reports only those workers and treats their complete pass
marker set as the worker-stage success condition. For output roots created by
the earlier launcher, a missing `physical_gpus` entry falls back to `0 1`.

Final acceptance remains unchanged and still requires:

- all 63 terrain JSON results (3 policies x 3 seeds x 7 terrains);
- all selected worker pass markers and no failure marker;
- `FORMAL_MATRIX_PASS`;
- `formal_policy_summary.csv` and `formal_policy_summary.json`; and
- a valid `SHA256SUMS` file.

No metric, threshold, ranking key, model-selection rule, or seed value changes.

## Error Handling

- Any selected GPU that fails CUDA allocation or computation aborts startup.
- A failed worker writes `WORKER_<physical-index>_FAILED`; the coordinator
  writes `MATRIX_FAILED` and does not aggregate.
- A stale PID belonging to the same output root blocks restart until the prior
  worker is stopped.
- Unsupported or unavailable GPU selections fail with an actionable message
  before the seed smoke test.

## Testing

Tests will verify that:

- the default selection remains physical GPUs 0 and 1;
- explicit one-GPU selection is accepted and represented in the manifest;
- invalid or duplicate selections are rejected;
- queue creation assigns all nine jobs once when one GPU is selected;
- worker launch and coordinator acceptance use the selected GPU list rather
  than hard-coded GPU 0/1 markers;
- the status script reads the manifest and remains backward compatible; and
- the complete existing Python and shell syntax suites still pass.

Runtime acceptance on the paid host requires a one-A100 container to print one
`FORMAL EVALUATION GPU PREFLIGHT PASS`, then
`FORMAL EVALUATION SEED SMOKE PASS`, and finally
`FORMAL EVALUATION MATRIX START PASS` before detached evaluation begins.
