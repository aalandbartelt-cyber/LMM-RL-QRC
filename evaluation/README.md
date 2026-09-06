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
