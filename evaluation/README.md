# Evaluation

Evaluation workspace for automated metrics, batch results, and plots.

Core metrics:

- success_rate
- completion_time
- velocity_tracking_error
- roll_error
- pitch_error
- fall_count
- collision_count
- safety_stop_count
- path_error
- energy_like_cost

## Go2 Isaac Gym benchmark

`scripts/evaluate_go2_terrain.py` evaluates one checkpoint on independent
terrain processes and aggregates the results into JSON and CSV files. The
default suite covers flat, slope, rough slope, stairs up, stairs down,
obstacles, and wave terrain.

The script expects to run from the upstream `go2_rl_gym` repository so its
`legged_gym` package and Isaac Gym installation are available. Benchmark
settings use `QRC_*` environment variables to avoid conflicting with the
upstream argument parser.

Example:

```bash
QRC_RUN_ALL=1 \
QRC_DURATION_SECONDS=20 \
QRC_COMMAND_VX=0.5 \
QRC_OUTPUT_DIR=/home/jovyan/go2_work/benchmark_model3501 \
python -u evaluate_go2_terrain.py \
  --task=go2 \
  --headless \
  --num_envs=256 \
  --seed=1 \
  --sim_device=cuda:0 \
  --rl_device=cuda:0 \
  --experiment_name=go2_baseline \
  --load_run=Jul28_12-52-34_stage2_add3000_seed1 \
  --checkpoint=3501
```

## Formal three-policy evaluation matrix

The formal protocol compares the newest `stage3_control/model_5001.pt` with the
two curriculum-v2 `model_5601.pt` checkpoints. It evaluates all three policies
on seeds `20260911`, `20260912`, and `20260913` across the seven default
terrains. Every policy/seed run uses 256 environments, 20 seconds per terrain,
and the fixed command `(vx, vy, yaw) = (0.5, 0.0, 0.0)`.

The earlier diagnostic batch whose JSON files reported seed `20260911` while
the Isaac Gym log printed `Setting seed: 1` is useful for comparison but is not
formal reproducibility evidence. Formal evaluation requires the runtime seed
smoke gate added after commit `7897e16`.

### Start on a two-A100 host

Start the paid host only when ready to run, then execute:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate /home/jovyan/conda/envs/go2rl
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd ~/go2_work/LMM-RL-QRC
git pull --ff-only
git status --short

bash rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh start
```

The foreground launch performs both GPU preflights and a five-second seed smoke
test before it starts detached workers. Do not continue unless it prints:

```text
FORMAL EVALUATION GPU PREFLIGHT PASS
FORMAL EVALUATION GPU PREFLIGHT PASS
FORMAL EVALUATION SEED SMOKE PASS
FORMAL EVALUATION MATRIX START PASS
```

The output root is saved to:

```text
~/go2_work/latest_formal_eval_dir.txt
```

### Check progress safely

Use the read-only status command. It does not call the system screen-refresh
utility, which may crash when it loads Conda's ncurses libraries:

```bash
cd ~/go2_work/LMM-RL-QRC
bash rl_control/legacy_isaacgym/show_formal_evaluation_status.sh
```

A running matrix reports zero to nine completed jobs. Final acceptance requires
all of these artifacts:

```text
seed_smoke/SEED_SMOKE_PASS
workers/WORKER_0_PASS
workers/WORKER_1_PASS
FORMAL_MATRIX_PASS
formal_policy_summary.csv
formal_policy_summary.json
SHA256SUMS
```

Any `FAILED`, `WORKER_*_FAILED`, or `MATRIX_FAILED` marker means the matrix is
not accepted.

### Resume an interrupted output root

After recreating the host, pass the existing output root explicitly. Complete
policy/seed jobs are verified and skipped; partial jobs are rerun:

```bash
cd ~/go2_work/LMM-RL-QRC
OUTPUT_ROOT=/home/jovyan/go2_work/go2_formal_eval_UTC_TIMESTAMP
bash rl_control/legacy_isaacgym/run_formal_evaluation_matrix.sh \
  start "$OUTPUT_ROOT"
```

### Read the final ranking

```bash
OUTPUT_ROOT=$(cat ~/go2_work/latest_formal_eval_dir.txt)
cat "$OUTPUT_ROOT/formal_policy_summary.csv"
(
  cd "$OUTPUT_ROOT"
  sha256sum -c SHA256SUMS
)
```

The ranking first maximizes mean success rate, then minimizes total falls,
collision sample rate, velocity-tracking RMSE, mechanical power, and mean tilt.
Per-terrain JSON files remain the authoritative evidence for specialist
strengths and regressions.
