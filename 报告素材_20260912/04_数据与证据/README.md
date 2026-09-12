# 本地提交媒体素材包（DG-202609）

生成时间：2026-09-12 12:10:36
生成方式：完全本地生成，未使用云端 GPU。

## 证据边界（报告引用时请保留以下措辞）

- 双场景演示视频与关键帧为**本地 MuJoCo 任务级仿真**，由正式评测第一名的 baseline5001 策略驱动，**不是真机实验**。
- 量化成绩以自强5000平台 63 组正式评测数据快照为准（`evaluation/results/go2_formal_matrix_20260911/`）。
- 训练曲线来自自强5000平台 TensorBoard 日志（随运行包哈希交接）。

## 文件清单

- `01_campus_security_simulation.mp4` — 园区安全巡检路线演示视频（本地仿真）（SHA-256 `64b26d1a7e9831f2…`）
- `02_disaster_response_simulation.mp4` — 灾害侦察与物资投送演示视频（本地仿真）（SHA-256 `b5f03e7a8e81bdda…`）
- `03_hero_campus_security.png` — 园区巡检报告关键帧（本地仿真）（SHA-256 `f5772a5314e171cf…`）
- `04_hero_disaster_response.png` — 灾害响应报告关键帧（本地仿真）（SHA-256 `c130c5e2f7d686e4…`）
- `05_formal_evaluation_comparison.png` — 三策略正式评测对比图（云端评测快照）（SHA-256 `1c8417977d52f262…`）
- `06_training_curves.png` — baseline5001 训练曲线（云端日志）（SHA-256 `47f6164bd7c39789…`）
- `07_system_architecture.png` — 系统架构图（SHA-256 `ca686d6043b70904…`）
- `08_evidence_contact_sheet.png` — 证据一览拼图（SHA-256 `489837efd868734c…`）
- `route_results.json` — 本地路线运行结构化结果（SHA-256 `834baf1c148585e1…`）

## 路线运行结果

- campus_security：PASS，检查点 18/18，用时 71.2s，最低质心高度 0.296m
- disaster_response：PASS，检查点 10/10，用时 117.3s，最低质心高度 0.311m

校验：`python -m simulation.demo_media.generate_submission_media --verify-only --output-dir <本目录>`
