# Go2 Campus RL Implementation

Approved design: `2026-09-08-go2-campus-rl-design.md`.

## Execution

- [x] Import the teammate's portable package and tests under `rl_control`.
- [x] Fix launch sequencing, CLI forwarding, resume handling and provenance.
- [x] Add curriculum terrains and conservative locomotion reward overrides.
- [x] Integrate campus geometry and feedback-driven physical policy evaluation.
- [x] Add pinned environment bootstrap, GPU preflight and platform runbook.
- [x] Add artifact integrity verification and local metric plotting.
- [x] Run portable tests, compilation, shell syntax and packaging checks.
- [x] Review runtime contracts against pinned upstream sources; record GPU gates.

## Boundaries

The main implementation owns `rl_control/campus_rl`, `go2_foundation`, launch
scripts, packaging and plotting. The platform worker owns `rl_control/platform`
and environment runbooks. The scene worker owns
`simulation/scenes/campus_security` and scene tests. GPU tests remain mandatory
after deployment and must not be represented by portable unit tests.

## Corrections To Initial Design

Isaac Sim 5.1 is pinned for the team's scene compatibility. Its supported full
runtime needs RTX hardware and a suitable host driver. A100 physics-only use is
experimental and requires an explicit opt-in plus a real startup/step test.
Reinstalling a Python environment cannot upgrade the platform's host driver.

The initial 45-observation locomotion actor is blind to walls and route geometry.
Route following and collision supervision belong to the feedback controller.
Terrain locomotion training alone does not establish navigation competence.

Policy and evaluation use the same resolved robot, observation, action and timing
configuration. Old Isaac Gym checkpoints are not assumed compatible with Lab.
