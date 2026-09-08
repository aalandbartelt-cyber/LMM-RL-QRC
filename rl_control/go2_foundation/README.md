# Go2 基础控制接口（staging）

这是项目的第一层基础代码，目标是先把仿真、策略和未来硬件适配器之间的接口固定下来。它是独立工作区里的 staging 副本，**不会修改 GitHub 仓库或 `D:\强化学习项目\campus_security_go2`**。

## 已完成的接口

- `RobotState -> ObservationBuilder -> 45 维 observation`；顺序为机身角速度、重力投影、速度命令、关节位置偏差、关节速度、上一动作。
- `PolicyRunner` 统一 TorchScript/ONNX/零策略调用，动作固定为 12 维。
- `CommandLimiter` 限制巡逻控制器产生的 `vx、vy、yaw_rate`。
- `SafetySupervisor` 对失联、摔倒、姿态过大、观测过期和非法动作执行停机判定。
- `EpisodeMetrics` / `MetricsAccumulator` 记录成功率、航点、摔倒、碰撞、安全停止、速度误差和倾角。
- `IsaacLabStateAdapter` 从 Isaac Lab 常见的批量 tensor/mapping 中抽取单个环境，转换为统一 `RobotState`。
- `Go2ControlCycle` 组合观测、策略推理和安全门，仿真与部署使用同一控制顺序。
- `RouteProgressEvaluator` 计算 P0-P7 等有序航点、路径长度、倾角、摔倒、碰撞和安全停止；结果强制携带“运动学预览 / RL 物理仿真 / 真机”证据等级。
- `PerformanceGate` 不看训练 reward 单独晋级：它要求至少三个随机种子、每种子独立 episode 数，并检查成功率、速度误差、摔倒、碰撞、安全停止和最大倾角。
- `PatrolMissionScheduler` 在低层策略外实现检查点停留、任务结束安全零速度和复位，兼容原 `WaypointVelocityController` 输出。
- `CampusMissionPipeline` 将航点控制器、检查停留与命令限幅组合成巡逻/送餐可共用的上层任务接口。
- `load_route_file`、`RouteVelocityController` / `MissionWaypoint` 提供不依赖原始园区脚本的路线控制；相同接口可换成配送路线，策略仍复用同一个低层速度跟踪 policy。
- `scripts.assess_performance` 读取训练后的 JSON 结果并执行可重复的性能晋级判断。
- `scripts.audit_policy_interface` 在加载策略前审计 observation/action 维度、缩放与控制周期，避免混用 Unitree RL Lab、MuJoCo 与真机 profile。
- `scripts.create_policy_manifest` / `scripts.verify_policy_manifest` 将 checkpoint SHA-256 与审计 metadata 绑定；模型文件被替换、扩展名不符或 I/O profile 不符都会失败，不会直接进入评估/部署。
- `scripts.launch_unitree_training` 先以 dry-run 方式检查官方 Unitree RL Lab 检出目录，再生成可审阅的 baseline/campus 训练命令；未给出 `--execute` 时不会启动训练。

## 运行验证

当前包只依赖 Python 标准库；在工作区根目录运行：

```text
python -m unittest discover -s tests -p "test_*.py" -v
```

也可以运行不依赖仿真器的端到端冒烟验证：

```text
python -m examples.foundation_smoke
```

如果要对接现有园区控制器而不改动它，可执行：

```text
python -m examples.verify_campus_bridge "D:\强化学习项目\campus_security_go2\mission_controller.py"
```

当前场景的运动学巡逻预览也可做只读路线验收；输出会明确标为非 RL 证据：

```text
python -m examples.evaluate_campus_preview "D:\强化学习项目\campus_security_go2\mission_controller.py"
```

在自强 5000 的 GPU 实例中，训练前运行只读环境预检：

```text
python -m scripts.preflight --strict
```

它只检查 Python、GPU、核心模块和可选策略文件，不会安装依赖或提交训练任务。

## 下一步接入顺序

1. 写 Isaac Lab adapter，把仿真状态转换为 `RobotState`。
2. 用 `ZeroPolicy` 先验证站立/停止/安全链路，再换成导出的 PPO policy。
3. 将 `WaypointVelocityController` 的输出经 `CommandLimiter` 送入 observation。
4. 完成单航点、低速直行、停止、转向，再接完整 P0-P7 路线。
5. 最后才写 Go2 SDK2 硬件 adapter；安全监管和急停必须独立于策略。

## 重要的配置约束

不同导出策略可能使用不同的 command scale。本包默认采用 MuJoCo 示例中的 `[2.0, 2.0, 0.25]`；如果接入真实部署配置，应显式创建 `Go2ObservationConfig(command_scale=(3.0, 2.0, 0.5))`，不能混用。也可以直接使用 `Go2ObservationConfig.mujoco()` 或 `Go2ObservationConfig.real_deployment()`。

当前官方 `unitree_rl_lab` 的 Go2 速度任务又是另一套 policy 观测：角速度 scale 为 `0.2`、命令不缩放、输入使用 `joint_pos_rel`。对它使用 `Go2ObservationConfig.unitree_rl_lab_relative_joint()`，并保证 adapter 的输入确实是相对关节位置；不可把这套配置套到旧 Gym/MuJoCo 或真机导出的策略上。
