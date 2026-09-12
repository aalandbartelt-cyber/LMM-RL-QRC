# Simulation

Simulation group workspace.

Responsibilities:

- Select and maintain the simulator or open-source environment.
- Build scenario A: park/community patrol.
- Build scenario B: disaster reconnaissance and light delivery.
- Provide a unified runner for decision and RL modules.
- Save logs, videos, and episode metrics.

Expected subfolders:

- `docs/`: simulator setup and design notes.
- `scenes/`: scenario configs and assets.
- `runners/`: scripts that run complete tasks.
- `adapters/`: interfaces between simulator, decision module, and policy module.

## Local submission media package

Deterministic local pipeline that renders both mission scenarios with the
verified baseline5001 policy in MuJoCo and assembles the report figures.
No cloud GPU is used at any step.

### Generate the final package

```powershell
python -m simulation.demo_media.generate_submission_media --output-dir outputs/submission_media_20260912 --fps 30 --duration 140
```

Quick pipeline check (short low-fps clips):

```powershell
python -m simulation.demo_media.generate_submission_media --quick --output-dir outputs/submission_media_quick
```

Verify an existing package (SHA-256 manifest + video properties):

```powershell
python -m simulation.demo_media.generate_submission_media --verify-only --output-dir outputs/submission_media_20260912
```

### Inputs

- Policy checkpoint and Go2 MJCF assets, mirrored from the hash-verified
  runtime package `baseline5001_local_runtime_20260911T125715Z.tar.gz` to a
  pure-ASCII path (`C:/Users/mary3/go2_runtime/baseline5001_20260911/`),
  because MuJoCo's file loader rejects non-ASCII paths.
- Formal evaluation snapshot: `evaluation/results/go2_formal_matrix_20260911/`.
- Training curve source: the TensorBoard log inside the runtime package
  (parsed directly, no tensorboard dependency).

### Outputs

- `01_campus_security_simulation.mp4` / `02_disaster_response_simulation.mp4`
  — full route runs driven by the real baseline5001 policy; run results
  (checkpoints, elapsed, minimum base height) are embedded as end cards and
  stored in `route_results.json`.
- `03_hero_campus_security.png` / `04_hero_disaster_response.png` — report keyframes.
- `05_formal_evaluation_comparison.png` — three-policy formal metrics.
- `06_training_curves.png` — baseline5001 training curves.
- `07_system_architecture.png` — system architecture diagram.
- `08_evidence_contact_sheet.png` — all evidence in one labeled sheet.
- `route_results.json`, `manifest.json` (SHA-256 for every artifact), `README.md`.

### Evidence boundary (keep this wording in the report)

- The scenario videos and keyframes are **local MuJoCo task-level
  simulations** driven by the baseline5001 policy; they are **not real-robot
  experiments**.
- Quantitative results come from the 63-run formal evaluation on the
  Ziqiang-5000 platform (snapshot referenced above).
- Training curves come from the Ziqiang-5000 TensorBoard log handed over with
  hash verification.
