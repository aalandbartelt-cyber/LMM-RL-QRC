# LMM-RL-QRC

AI 大模型与强化学习驱动的四足机器人智能控制系统。

本仓库用于第十五届“挑战杯”揭榜挂帅擂台赛 DG-202609 项目开发。当前阶段以高可信仿真为主，目标是在九月初筛前完成：

- 两个场景的仿真闭环：园区/社区巡逻安防、灾害现场应急勘察与轻物资投送。
- 大模型任务规划：自然语言任务 -> 结构化任务 JSON -> 行为树/状态机执行。
- 强化学习运动控制：复现四足 locomotion baseline，设计 observation/action/reward，形成可评估策略接口。
- 可验证的 Sim2Real 路径：控制接口、安全 supervisor、日志、评估指标、真机准入条件。

## Current Strategy

暑期线下测试资源有限，本项目采用“仿真先行，真机准入”的路线：

```text
自然语言任务
  -> LLM / rule-based task planner
  -> task JSON
  -> behavior tree / safety supervisor
  -> simulation scene runner
  -> RL / baseline locomotion policy
  -> logs, metrics, videos, failure cases
```

九月前不盲目承诺完整真机演示，而是先证明：

1. 系统能在仿真中稳定复现。
2. 每个场景都有量化指标和失败分析。
3. RL 策略有 baseline 对比和可解释 reward 设计。
4. 真机接口、安全限制、日志和部署路径已经准备好。

## Team Structure

目前按三大技术组推进。

| Group | Main Scope | Key Deliverables |
|---|---|---|
| Simulation | 场景与仿真实现 | 两个数字场景、场景 runner、自动化评估、录屏与日志 |
| LLM Decision | 大模型决策 | 任务 JSON schema、行为树、任务状态机、安全校验、任务报告 |
| RL Control | 强化学习运动控制 | RL baseline、reward 设计、策略接口、checkpoint、策略评估 |

建议 8 人分配：

| Member | Group | Role |
|---|---|---|
| A | Simulation | 仿真框架与组间接口 |
| B | Simulation | 场景资产与任务脚本 |
| C | Simulation | 自动化评估与数据 |
| D | LLM Decision | 任务规划与行为树 |
| E | LLM Decision | 安全决策与感知接口 |
| F | RL Control | RL-MPC 理论与 reward 设计 |
| G | RL Control | Go2 RL 复现与训练 |
| H | RL Control | 策略评估、接口与部署安全 |

## Repository Layout

```text
LMM-RL-QRC/
  README.md
  .gitignore
  configs/                 # Shared YAML/JSON config files
  docs/                    # Minimal project docs and reference notes
  simulation/              # Simulation environments, scenes, runners
  llm_decision/            # Task planning, behavior tree, safety decision
  rl_control/              # RL learning notes, training, rewards, policies
  evaluation/              # Metrics, batch test results, plots
  deployment/              # Sim2Real and real-robot deployment interface
  scripts/                 # Utility scripts
  assets/                  # Images, diagrams, demo screenshots
  weekly_reports/          # Weekly deliverables and progress records
  third_party/             # Notes for external repos; do not vendor huge code
```

## Git Quickstart

Team members new to Git can read:

- `docs/git_quickstart.md`

## Group Interfaces

### LLM Decision -> Simulation

The decision module outputs a task JSON:

```json
{
  "task_id": "patrol_001",
  "scene": "park_patrol",
  "goal": "patrol_and_report",
  "waypoints": ["p1", "p2", "p3"],
  "constraints": {
    "max_speed": 0.5,
    "timeout_sec": 180
  }
}
```

### Simulation -> LLM Decision

The simulator returns task state:

```json
{
  "task_status": "running",
  "robot_pose": {"x": 1.0, "y": 2.0, "yaw": 0.2},
  "current_waypoint": "p2",
  "detected_objects": [],
  "safety_event": null
}
```

### RL Control -> Simulation

The locomotion policy returns high-level motion commands first:

```json
{
  "command_type": "velocity",
  "vx": 0.3,
  "vy": 0.0,
  "yaw_rate": 0.1,
  "speed_scale": 1.0,
  "policy_name": "mock_policy_v0",
  "safety_flag": "ok"
}
```

Low-level 12-joint actions should not be used until the policy is stable in simulation and the safety supervisor is ready.

### Simulation -> RL Control

The simulator outputs episode metrics:

```json
{
  "episode_id": "sceneA_0001",
  "success": true,
  "completion_time": 83.2,
  "fall_count": 0,
  "collision_count": 1,
  "velocity_tracking_error": 0.08,
  "roll_error": 0.02,
  "pitch_error": 0.03,
  "failure_reason": null
}
```

## Milestones

| Date | Goal | Expected Output |
|---|---|---|
| 2026-07-12 | Project bootstrapping | Scene sketches, task schema, RL notes, env logs, mock policy |
| 2026-07-19 | Minimal closed loop | Baseline sim demo, behavior-tree mock execution, initial RL baseline attempt |
| 2026-07-26 | Scenario A MVP | Park patrol simulation video and metrics |
| 2026-08-02 | Scenario B MVP | Disaster search/delivery simulation video and metrics |
| 2026-08-09 | First integration | Unified runner, RL checkpoint or baseline policy, task switching |
| 2026-08-23 | Batch evaluation | 30-50 trials per scenario, failure cases, baseline comparison |
| 2026-08-30 | Freeze v1 | Frozen interfaces, scenario configs, policy wrapper, teacher demo package |
| 2026-09-14 | Submission prep | Final code package, videos, metrics, reproducibility instructions |

## First Week Checklist

By 2026-07-12, the repository should contain at least:

```text
simulation/docs/simulator_selection.md
simulation/scenes/park_patrol/README.md
simulation/scenes/disaster_response/README.md
evaluation/docs/metrics_schema.md
llm_decision/schemas/task_schema.json
llm_decision/examples/patrol_task.json
llm_decision/examples/rescue_task.json
llm_decision/behavior_tree/minimal_bt.py
llm_decision/safety/safety_rules.md
rl_control/learning_notes/RL_notes_day1.md
rl_control/rewards/reward_design_v0.1.md
rl_control/env/env_install_log.md
rl_control/policies/policy_interface.md
rl_control/wrappers/mock_policy.py
weekly_reports/2026-07-12/
```

## Development Rules

- Keep every module runnable from a documented command.
- Every experiment must save a log or CSV result.
- Do not commit large datasets, checkpoints, or videos directly unless they are small demo assets.
- Record external repositories in `third_party/README.md` instead of copying them wholesale.
- Prefer JSON/YAML schemas for cross-group interfaces.
- Use English for code identifiers and filenames; Chinese is fine in Markdown notes.
- Every weekly deliverable should include one of: code, log, screenshot, video link, metric table, or design note.

## Branches

Suggested branch names:

```text
simulation/scene-a-mvp
simulation/evaluation-runner
llm/task-schema
llm/behavior-tree
rl/baseline-repro
rl/policy-wrapper
docs/weekly-update
```

## References

Local reference files are indexed in `docs/references/`.

- `docs/references/contest/DG-202609比赛方案.pdf`
- `docs/references/learning/参考.md`
- `docs/references/learning/2604.08508v2.pdf`
- `docs/references/learning/面向复杂地形的四足机器人RL-MPC分层运动控制方法研究_王留东.pdf`

External references recorded so far:

- MathFoundationRL, Mathematical Foundations of Reinforcement Learning
- Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation
- Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- RoboGauge
- go2_rl_gym
- unitree_cpp_deploy
