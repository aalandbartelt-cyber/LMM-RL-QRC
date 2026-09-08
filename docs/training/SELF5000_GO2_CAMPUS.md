# 自强平台 Go2 Campus RL 两卡运行手册

推荐容器名：`go2-campus-2a100-s1s2-0908`

目标是让两张卡分别运行独立随机种子，而不是把单个 Isaac 仿真进程强行做分布式训练。容器计费后严格按“宿主门禁、安装、冒烟、短训、长训、评测、归档”执行。

## 1. 获取项目

```bash
mkdir -p ~/go2_work
cd ~/go2_work
git clone https://github.com/aalandbartelt-cyber/LMM-RL-QRC.git
cd LMM-RL-QRC
```

如果目录已存在：

```bash
cd ~/go2_work/LMM-RL-QRC
git pull --ff-only
```

## 2. 先做宿主门禁

A100 在 Isaac Sim 5.1 中只按实验性无界面物理路径处理，因此必须显式允许；这不会绕过驱动版本检查。

```bash
cd ~/go2_work/LMM-RL-QRC
python3 rl_control/platform/preflight_self5000.py \
  --phase host --gpus 2 --allow-unsupported-a100 \
  --output ~/go2_work/self5000_host_preflight.json
echo "exit=$?"
```

只有 JSON 中 `"ready": true` 且退出码为 `0` 才继续。出现 `driver_too_old`、`glibc_too_old`、`insufficient_gpus` 时立即停止计费安装。重装 Conda 不能升级宿主机 NVIDIA 驱动。

## 3. 安装固定环境

```bash
export QRC_GPUS=2
export QRC_ALLOW_UNSUPPORTED_A100=1
bash rl_control/platform/bootstrap_self5000.sh \
  2>&1 | tee ~/go2_work/bootstrap_self5000.log
```

固定版本为 Python 3.11、Isaac Sim 5.1.0、Isaac Lab 指定提交、Unitree RL Lab 指定提交及 RSL-RL 2.3.1。安装结束必须看到 `SELF5000 BOOTSTRAP PASS`。

## 4. 两卡冒烟

```bash
bash rl_control/platform/run_smoke_matrix.sh
```

必须看到 `TWO GPU ISAAC SMOKE MATRIX PASS`。它会在物理 GPU 0、1 上分别创建官方 Go2 环境和 Campus 环境并真实步进。任一卡失败都不进入长训。

## 5. 先做 50 轮短训

```bash
bash rl_control/platform/start_two_seed_stage.sh baseline 50 512
bash rl_control/platform/show_progress.sh 5
```

`show_progress.sh` 用 `Ctrl+C` 退出只会停止监视，不会停止训练。检查两个任务均无 `Traceback`，GPU 利用率正常，并能持续出现 `Learning iteration`。

## 6. 两种子正式基线

短训通过后启动两个独立正式实验：

```bash
bash rl_control/platform/start_two_seed_stage.sh baseline 1500 4096
bash rl_control/platform/show_progress.sh 10
```

如果显存或启动压力过大，将 `4096` 降至 `2048`，不要改变其他超参数。检查 checkpoint：

```bash
find ~/go2_work/upstream/unitree_rl_lab/logs \
  -type f -name 'model_*.pt' -printf '%T@ %p\n' | sort -nr | head -n 20
```

## 7. Campus 定向微调

选择基线中 reward、稳定性和模型文件都正常的一次运行。`QRC_LOAD_RUN` 填 RSL-RL 运行目录名，`QRC_CHECKPOINT` 填文件名。两个微调种子从同一基线 checkpoint 起步，便于比较。

```bash
export QRC_LOAD_RUN='替换为基线运行目录名'
export QRC_CHECKPOINT='model_1500.pt'
bash rl_control/platform/start_two_seed_stage.sh campus 1500 4096
bash rl_control/platform/show_progress.sh 10
```

Campus 训练包含平地、随机粗糙地、正反坡、离散障碍、上楼梯和下楼梯。它优化步态鲁棒性，不声称解决没有感知输入的全局避障。

## 8. 导出策略

对最终 checkpoint 执行无摄像头导出，参数必须使用 checkpoint 的绝对路径：

```bash
bash rl_control/platform/export_policy.sh \
  /home/jovyan/go2_work/upstream/unitree_rl_lab/logs/rsl_rl/实验名/运行名/model_1500.pt \
  Unitree-Go2-Campus-Velocity 0
```

成功标志为 `POLICY EXPORT PASS`。同目录下会得到 `exported/policy.pt`、`exported/policy.onnx` 和 `exported/SHA256SUMS`。

## 9. 物理路线评测与视频

把导出的 JIT `policy.pt` 下载后，在支持渲染的本地 RTX 机器执行：

```bash
python simulation/scenes/campus_security/evaluate_physical_route.py \
  --policy /absolute/path/to/exported/policies/policy.pt \
  --output-dir /absolute/path/to/campus_route_eval \
  --device cuda:0 --video --video-seconds 60
```

平台 A100 只做无界面训练和数值评测。曲线可以在任何本地电脑生成；Isaac 场景视频需要本地 Isaac Sim 运行时、兼容驱动和完整 Unitree 资产。

## 10. 打包每个训练结果

先找到控制日志目录和对应 RSL checkpoint 目录，然后执行：

```bash
bash rl_control/platform/package_results.sh \
  ~/go2_work/campus_training/campus_seed1_时间戳 \
  ~/go2_work/upstream/unitree_rl_lab/logs/rsl_rl/实验名/运行名 \
  ~/go2_work/campus_artifacts \
  campus_seed1_final
```

必须下载同名 `.tar.gz`、`.sha256`、`.manifest.json` 三个文件。Windows PowerShell 校验：

```powershell
$expected = (Get-Content '.\campus_seed1_final.sha256').Split()[0].ToLower()
$actual = (Get-FileHash '.\campus_seed1_final.tar.gz' -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw 'SHA256 mismatch: redownload the archive' }
tar -tzf '.\campus_seed1_final.tar.gz' | Select-Object -First 20
```

## 11. 本地绘制训练曲线

解压两个种子后，在仓库根目录运行：

```powershell
py -3.11 -m pip install -e '.\rl_control[analysis]'
py -3.11 .\rl_control\scripts\plot_training_metrics.py `
  --log 'seed1=D:\results\seed1\training_run\train.log' `
  --log 'seed2=D:\results\seed2\training_run\train.log' `
  --output '.\evaluation\plots\campus_training.png' `
  --smooth-window 25
```

脚本同时生成同名 CSV，后续可直接用于论文表格和多种子均值统计。

## 12. 旧驱动回退边界

如果平台仍是 535 系列驱动，保留 7 月已经验证的 Isaac Gym/go2rl 环境和归档，只用于复现旧 locomotion 基线或定向地形训练。不要把旧 Isaac Gym checkpoint 直接加载进本项目的 Isaac Lab 任务，两套资产、观测配置和训练栈必须分别留存。新 Campus 场景训练应更换满足门禁的节点，或改在本地兼容 RTX 主机执行。

## 13. 销毁容器前

确认每个正式种子均已下载并通过哈希校验，同时保存以下文件：训练日志、全部目标 checkpoint、导出策略、`run.env`、运行时门禁 JSON、`pip_freeze.txt`、版本锁、评测 JSON/CSV、视频和 SHA-256 清单。三件套本地校验通过后再销毁容器。
