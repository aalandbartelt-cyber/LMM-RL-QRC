# Campus Security Go2 Scene

本目录将仿真组提供的园区安防场景整理为两种用途，二者的证据等级不同。

## Files

- `kinematic_preview.py`: 原始运动学预览，只用于快速检查建筑外观和任务流程。它会直接写根位姿，不能作为 RL 策略效果证据。
- `mission_controller_legacy.py`: 队友原始任务控制器，保留用于溯源和对照。
- `scene_layout.svg`: 仿真组提供的场景平面图。
- `physical_route.json`: 修正后的 19 点物理巡检路线，绕开建筑碰撞体并校正动力机房坐标。
- `evaluate_physical_route.py`: 真实物理步进入口。它加载导出的 JIT 策略，只写速度命令，不传送机器人根位姿。

旧预览路线的 `P2 -> P3` 线段穿过办公楼，且旧 `P5` 不在动力机房旁。几何测试会明确拒绝这两类错误。

## Physical Evaluation

在已经通过 Isaac Lab 运行时门禁的机器上运行：

```bash
cd ~/go2_work/LMM-RL-QRC
source ~/conda/envs/go2campus/bin/activate

python simulation/scenes/campus_security/evaluate_physical_route.py \
  --policy /absolute/path/to/exported/policies/policy.pt \
  --route simulation/scenes/campus_security/physical_route.json \
  --output-dir ~/go2_work/campus_route_eval/seed1 \
  --device cuda:0 \
  --video --video-seconds 60
```

无窗口批量评测时附加 `--headless`。输出包括 `route_result.json`、`trajectory.csv` 和可选 MP4。`status=PASS` 仅在路线完成、未摔倒且未长时间停滞时成立。

当前 45 维 locomotion policy 不含相机、激光雷达或建筑几何观测。它负责执行速度命令，路线控制器负责反馈导航；这不能替代后续感知避障。
