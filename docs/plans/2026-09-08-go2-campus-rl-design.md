# Go2 Campus RL Integration Design

Date: 2026-09-08
Status: Approved

## 1. Objective

Integrate the campus security scene and the teammate's Go2 reinforcement-learning
foundation into `LMM-RL-QRC`, then produce a reproducible training package for the
Shanghai University self-5000 platform.

The first milestone is a physically controlled Go2 policy that remains stable on
flat ground, slopes, stairs, discrete obstacles, and narrow turns. The full campus
scene is an evaluation environment, not the first large-scale training environment.

## 2. Confirmed Architecture

The project uses a split training and presentation workflow:

1. Self-5000 A100 instances perform headless physics training without cameras,
   video recording, livestreaming, or RTX rendering.
2. Vectorized procedural terrains train locomotion, turning, stopping, obstacle
   recovery, and robustness efficiently.
3. The campus scene runs as a single-environment route and mission evaluation.
4. An RTX 5070 Ti Laptop or the existing MuJoCo renderer generates videos,
   screenshots, plots, and presentation artifacts from saved checkpoints and logs.

This separation is required because A100 GPUs do not provide RTX rendering support.
Training must never depend on camera or video extensions on the platform.

## 3. Version Strategy

Use a new environment named `go2campus`. Do not delete or modify the proven
`go2rl` Isaac Gym environment.

Target stack:

- Ubuntu 22.04 or a compatible self-5000 image
- Python 3.11
- Isaac Sim 5.1.0
- Isaac Lab 2.3.0 pinned to an exact commit
- Unitree RL Lab pinned to an exact commit
- RSL-RL version matched to the pinned Unitree RL Lab checkout
- This repository installed in editable mode

The installer must fail before installing or training when disk space, GLIBC,
Python, NVIDIA driver access, CUDA allocation, or repository prerequisites are
invalid. Isaac Sim compatibility is verified with a no-window smoke test.

## 4. Repository Layout

```text
LMM-RL-QRC/
|-- rl_control/
|   |-- go2_foundation/
|   |-- campus_rl/
|   |   |-- env_cfg.py
|   |   |-- terrain_cfg.py
|   |   |-- rewards.py
|   |   `-- agents/
|   |-- scripts/
|   |   |-- install_self5000.sh
|   |   |-- preflight_self5000.sh
|   |   |-- smoke_train.sh
|   |   |-- train_baseline.sh
|   |   |-- train_campus.sh
|   |   `-- package_results.sh
|   `-- tests/
|-- simulation/scenes/campus_security/
|-- configs/campus/
`-- docs/training/
```

Upstream Isaac Lab and Unitree RL Lab repositories remain external checkouts.
Their commit hashes and installation locations are recorded in every experiment
manifest instead of copying their source trees into this repository.

## 5. Training Environment

The campus training task extends the official `Unitree-Go2-Velocity` contract and
preserves its 45-dimensional observation and 12-dimensional action interfaces.

Training terrain includes:

- flat surfaces with randomized friction;
- ascending and descending slopes;
- rough slopes and uneven height fields;
- ascending and descending stairs;
- sparse blocks and low obstacles;
- corridor and turn approximations for campus route recovery.

Command curriculum includes standing, low-speed forward motion, lateral motion,
left and right yaw commands, stop-and-go transitions, and command ramps. Harder
terrain and wider commands are introduced only after stable flat-ground tracking.

Rewards and termination rules cover:

- linear and yaw velocity tracking;
- route progress and anti-stall behavior;
- upright base orientation and suitable base height;
- action smoothness and joint acceleration;
- torque, energy, foot sliding, and undesired contact penalties;
- collision and prolonged no-progress termination;
- controlled recovery rather than repeated pushing against obstacles.

Domain randomization covers friction, payload and base mass, motor strength,
observation noise, external pushes, control delay, and initial pose.

## 6. Training Stages

Each stage has a hard pass condition. A failed stage does not automatically start
the next expensive run.

### S0: Platform preflight

- CUDA tensor allocation succeeds on every requested GPU.
- Isaac Sim starts headlessly without cameras.
- Isaac Lab and Unitree RL Lab import successfully.
- The custom Gym task is listed exactly once.

### S1: Official baseline smoke test

- Run `Unitree-Go2-Velocity` with 16 environments for 5 to 10 iterations.
- Confirm environment construction, stepping, checkpoint output, and clean exit.

### S2: Campus task smoke test

- Run the custom task with 64 environments for 10 to 20 iterations.
- Confirm all terrains spawn, observations and actions are finite, and rewards are
  logged without shape or registration errors.

### S3: Baseline training

- Train the official task to obtain a compatible locomotion checkpoint.
- Save checkpoints, resolved configs, TensorBoard logs, command line, Git commits,
  package versions, GPU inventory, and timestamps.

### S4: Campus curriculum fine-tuning

- Resume from the compatible S3 checkpoint.
- Introduce campus terrain and command curricula progressively.
- Evaluate fixed seeds regularly instead of selecting only by training reward.

### S5: Route and mission evaluation

- Connect the waypoint controller's `vx`, `vy`, and `yaw_rate` output to the policy.
- Remove all root-pose teleportation from the RL evaluation path.
- Measure completion rate, collisions, falls, no-progress events, tracking error,
  route completion time, and minimum base height.

## 7. Multi-GPU Policy

The default paid run uses two A100 GPUs only after both pass independent CUDA and
Isaac Lab smoke tests. Distributed training is optional and must demonstrate a
throughput improvement over one GPU before use.

If distributed RSL-RL is unstable or slower for this environment, the two GPUs run
independent seeds concurrently. Independent seeds are often more valuable than a
single distributed run because they provide both robustness evidence and model
selection data.

## 8. Local RTX Output Workflow

The RTX 5070 Ti Laptop is used after checkpoints and logs are downloaded. It can
produce:

- TensorBoard curves and publication-ready metric plots;
- evaluation tables, confusion-style failure summaries, and route diagrams;
- MuJoCo policy videos and frame captures;
- Isaac Sim campus videos when the local driver, VRAM, and pinned Isaac Sim build
  pass compatibility checks.

The local machine does not need to repeat the expensive policy training. It must
use the same policy interface, joint order, action scale, control frequency, and
normalization values recorded by the training manifest.

## 9. Artifacts and Reproducibility

Every accepted run must preserve:

- model checkpoints and exported policy;
- resolved environment and agent configuration;
- TensorBoard event files and plain-text logs;
- evaluation JSON/CSV files;
- package versions and upstream commit hashes;
- exact launch command and random seed;
- SHA-256 manifest for the final archive.

An archive is accepted only after remote integrity verification and a second local
SHA-256 verification after download.

## 10. Failure and Fallback Rules

If the A100 instance cannot start the pinned Isaac Lab stack in physics-only mode,
stop immediately. Do not attempt camera, video, GUI, or arbitrary package upgrades.

The fallback is to port the same terrain curriculum and reward changes to the
existing verified Isaac Gym `go2rl` environment, train on A100 there, and retain the
Isaac Lab campus scene solely for local route evaluation and presentation.

## 11. Acceptance Criteria

The integration is ready for paid platform training when:

1. All existing teammate and scene unit tests pass from the unified repository.
2. New configuration and launcher tests pass without Isaac Sim installed.
3. The installation and preflight scripts are repeatable and non-destructive.
4. Official and campus dry-run commands contain pinned task IDs and artifact paths.
5. The platform runbook contains copy-paste commands, expected outputs, stop rules,
   progress commands, packaging commands, and download verification commands.
6. No RL evaluation code writes the robot root pose to simulate locomotion.
