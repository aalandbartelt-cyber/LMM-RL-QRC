# Local Submission Media Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic local pipeline that generates two labeled simulation videos, report-ready figures, a combined pitch video, and a verifiable media manifest without cloud GPU use.

**Architecture:** Scenario configuration and deterministic mission timelines are separated from MuJoCo policy execution and OpenCV/Pillow presentation. The baseline5001 actor receives proprioceptive observations and high-level route velocity commands, while all root motion comes from MuJoCo integration. A single CLI renders both physical route runs, produces Matplotlib result figures, and validates every output before writing a SHA-256 manifest.

**Tech Stack:** Python 3.14, MuJoCo, PyTorch/ONNX Runtime, OpenCV, Pillow, NumPy, Matplotlib, JSON, pytest

---

### Task 1: Version the formal evaluation evidence

**Files:**
- Create: `evaluation/results/go2_formal_matrix_20260911/overall_summary.csv`
- Create: `evaluation/results/go2_formal_matrix_20260911/terrain_summary.csv`
- Create: `evaluation/results/go2_formal_matrix_20260911/README.md`

- [ ] **Step 1: Record the three-policy overall results**

Write a CSV with columns `rank,policy,mean_success_rate,total_falls,mean_collision_rate,mean_tracking_rmse_mps,mean_mechanical_power_w,mean_tilt_rad` and the verified values for baseline5001 and both curriculum policies.

- [ ] **Step 2: Record all 21 terrain aggregates**

Write the seven terrain rows for each policy with columns `policy,terrain,success_rate,fall_count,collision_rate,tracking_rmse_mps,mean_mechanical_power_w,mean_tilt_rad`.

- [ ] **Step 3: Verify row counts and baseline headline metrics**

Run:

```powershell
python -c "import csv,pathlib; p=pathlib.Path('evaluation/results/go2_formal_matrix_20260911/terrain_summary.csv'); r=list(csv.DictReader(p.open(encoding='utf-8'))); assert len(r)==21; b=[x for x in r if x['policy']=='baseline5001']; assert len(b)==7; print('FORMAL_EVIDENCE_PASS')"
```

Expected: `FORMAL_EVIDENCE_PASS`.

- [ ] **Step 4: Commit the evidence snapshot**

```powershell
git add evaluation/results/go2_formal_matrix_20260911
git commit -m "Record formal Go2 evaluation evidence"
```

### Task 2: Define deterministic scenario timelines

**Files:**
- Create: `simulation/demo_media/__init__.py`
- Create: `simulation/demo_media/scenarios.py`
- Create: `simulation/tests/test_demo_media.py`

- [ ] **Step 1: Write failing scenario tests**

```python
from simulation.demo_media.scenarios import build_scenarios


def test_two_submission_scenarios_have_complete_missions():
    scenarios = build_scenarios()
    assert set(scenarios) == {"campus_security", "disaster_response"}
    assert scenarios["campus_security"].events[-1].status == "任务完成"
    assert scenarios["disaster_response"].events[-1].status == "安全返航"


def test_evidence_labels_do_not_claim_real_robot():
    for scenario in build_scenarios().values():
        assert "仿真" in scenario.evidence_label
        assert "真机" not in scenario.evidence_label
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: import failure because `simulation.demo_media.scenarios` does not exist.

- [ ] **Step 3: Implement focused dataclasses and two scenarios**

Implement immutable `Point`, `MissionEvent`, and `Scenario` dataclasses. `build_scenarios()` returns the 19-point campus route plus a disaster route with safe zone, rubble, ramp, target, delivery point, blocked corridor detour, and home. Each scenario carries a Chinese title, natural-language task, safety limit, event timeline, evidence label, and physics clip names.

- [ ] **Step 4: Run the tests to verify GREEN**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: all scenario tests pass.

- [ ] **Step 5: Commit the scenario model**

```powershell
git add simulation/demo_media simulation/tests/test_demo_media.py
git commit -m "Add local dual-scene mission timelines"
```

### Task 3: Implement report-ready rendering

**Files:**
- Create: `simulation/demo_media/rendering.py`
- Modify: `simulation/tests/test_demo_media.py`

- [ ] **Step 1: Add failing frame and timeline tests**

```python
import numpy as np
from simulation.demo_media.rendering import interpolate_route, render_frame


def test_interpolate_route_starts_and_ends_at_route_bounds():
    scenario = build_scenarios()["campus_security"]
    assert interpolate_route(scenario.route, 0.0)[:2] == (
        scenario.route[0].x,
        scenario.route[0].y,
    )
    assert interpolate_route(scenario.route, 1.0)[:2] == (
        scenario.route[-1].x,
        scenario.route[-1].y,
    )


def test_rendered_frame_is_full_hd_rgb():
    frame = render_frame(build_scenarios()["campus_security"], 0.5, None)
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (1080, 1920, 3)
    assert frame.dtype == np.uint8
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: import failure for the new rendering functions.

- [ ] **Step 3: Implement the renderer**

Implement route-distance interpolation, a campus map, a disaster map, a vector Go2 marker, progress path, waypoint states, event banners, safety telemetry, task JSON panel, and a labeled physics-video inset. Use `C:/Windows/Fonts/msyh.ttc` when available and fall back to a bundled-safe font path search.

- [ ] **Step 4: Run the rendering tests**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: all tests pass and a frame is 1920×1080 RGB.

- [ ] **Step 5: Commit the renderer**

```powershell
git add simulation/demo_media/rendering.py simulation/tests/test_demo_media.py
git commit -m "Render labeled local mission simulations"
```

### Task 4: Build the media generator and validation manifest

**Files:**
- Create: `simulation/demo_media/mujoco_route.py`
- Create: `simulation/demo_media/generate_submission_media.py`
- Modify: `simulation/tests/test_demo_media.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add failing route-policy and CLI helper tests**

```python
from simulation.demo_media.generate_submission_media import validate_video, sha256_file
from simulation.demo_media.mujoco_route import compact_route, route_command


def test_sha256_file_is_stable(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"go2")
    assert sha256_file(path) == "be4fb8841ddfa79756046f79946b71a6147cb1c1d2ed13cd3430d04a197eafc3"


def test_compact_route_starts_at_origin_and_route_command_moves_forward():
    route = compact_route(build_scenarios()["campus_security"])
    assert (route[0].x, route[0].y) == (0.0, 0.0)
    command = route_command(0.0, 0.0, 0.0, route[1], 0.5)
    assert command.forward_mps > 0.0
    assert abs(command.yaw_rate_rps) <= 1.0
```

- [ ] **Step 2: Run the tests to verify RED**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: import failure for `generate_submission_media`.

- [ ] **Step 3: Implement generation and validation**

Implement `compact_route()` and the feedback route command, load the actor from the verified baseline5001 checkpoint or exported ONNX, and run the 12-joint policy in MuJoCo without writing the root pose. The CLI accepts `--output-dir`, `--policy`, `--go2-assets`, `--duration`, `--fps`, and `--quick`. It generates both physical scenario videos, hero frames, the formal comparison figure, architecture figure, evidence contact sheet, combined pitch video, README, and JSON manifest. `validate_video()` reopens each MP4 with OpenCV and verifies frame count, width, height, FPS, and nonzero duration. Add `/outputs/` to `.gitignore`.

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest simulation/tests/test_demo_media.py -v`

Expected: all tests pass.

- [ ] **Step 5: Run a quick end-to-end render**

Run:

```powershell
python -m simulation.demo_media.generate_submission_media --quick --output-dir outputs/submission_media_quick
```

Expected: `LOCAL_SUBMISSION_MEDIA_PASS` and a manifest in the output directory.

- [ ] **Step 6: Commit the media pipeline**

```powershell
git add .gitignore simulation/demo_media simulation/tests/test_demo_media.py
git commit -m "Generate local submission media package"
```

### Task 5: Generate and inspect the final package

**Files:**
- Create locally: `outputs/submission_media_20260911/*`
- Modify: `simulation/README.md`
- Modify: `docs/企业技术筛选_参赛方案大纲.md`

- [ ] **Step 1: Generate the final 1080p package**

Run:

```powershell
python -m simulation.demo_media.generate_submission_media --output-dir outputs/submission_media_20260911
```

Expected: `LOCAL_SUBMISSION_MEDIA_PASS` with three playable MP4 files and five PNG files.

- [ ] **Step 2: Visually inspect both hero images and the evidence sheet**

Open the three PNG files and verify that Chinese text is legible, route lines do not cover labels, evidence labels are visible, and no panel is clipped.

- [ ] **Step 3: Verify all manifest hashes and videos**

Run:

```powershell
python -m simulation.demo_media.generate_submission_media --verify-only --output-dir outputs/submission_media_20260911
```

Expected: `LOCAL_SUBMISSION_MEDIA_VERIFY_PASS`.

- [ ] **Step 4: Document the exact local command and report usage**

Add the generator command, output list, evidence-boundary wording, and recommended report placement to `simulation/README.md` and the contest outline.

- [ ] **Step 5: Run repository verification**

Run:

```powershell
python -m pytest -q
python -m compileall -q simulation/demo_media
git diff --check
```

Expected: tests pass, compilation succeeds, and `git diff --check` has no output.

- [ ] **Step 6: Commit the final documentation**

```powershell
git add simulation/README.md docs/企业技术筛选_参赛方案大纲.md
git commit -m "Document local simulation media workflow"
```
