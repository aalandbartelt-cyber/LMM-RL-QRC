"""Render the four academic diagrams used in the DG-202609 research report.

Style matches the submission media figures (dark theme, cyan accents) so the
report reads as one consistent set of evidence.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .generate_submission_media import _plt

BG = "#07111f"
PANEL = "#0d1b2d"
PANEL_ALT = "#11243a"
ACCENT = "#133449"
GREEN = "#123a2c"
AMBER = "#3a2b1a"
PURPLE = "#241a3a"
WINE = "#3a1a24"
EDGE = "#19c6e6"
TEXT = "#f2f7fb"
MUTED = "#9fb3c8"


def _canvas(title: str):
    plt = _plt()
    fig, axis = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(BG)
    axis.set_xlim(0, 19.2)
    axis.set_ylim(0, 10.8)
    axis.axis("off")
    axis.set_title(title, fontsize=22, color=TEXT, pad=16)
    return plt, fig, axis


def _box(plt, axis, x, y, w, h, label, color=PANEL_ALT, fontsize=13, edge=EDGE):
    axis.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10", facecolor=color, edgecolor=edge, linewidth=1.8)
    )
    axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color=TEXT)


def _arrow(plt, axis, x1, y1, x2, y2, label=None, lx=0.0, ly=0.14, color=EDGE):
    axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=20, color=color, linewidth=1.8))
    if label:
        axis.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=10.5, color=MUTED, ha="center")


def make_rl_network_diagram(output: Path) -> None:
    plt, fig, axis = _canvas("图 · 强化学习运动策略网络结构与控制链路")
    # Observation stack (left)
    obs_items = [
        ("本体角速度 ω（3维）×0.25", 8.6),
        ("投影重力 g（3维）", 7.6),
        ("速度指令 cmd（3维）×[2,2,0.25]", 6.6),
        ("关节位置 q−$q_0$（12维）", 5.6),
        ("关节速度 $\dot{q}$（12维）×0.05", 4.6),
        ("上一步动作 $a_{t-1}$（12维）", 3.6),
    ]
    axis.text(2.6, 9.85, "观测向量 obs（45维）", fontsize=15, color=EDGE, ha="center")
    for label, y in obs_items:
        _box(plt, axis, 0.7, y, 3.8, 0.72, label, PANEL, 11)
    # MLP (middle)
    axis.text(8.3, 9.85, "Actor 策略网络（MLP · ELU）", fontsize=15, color=EDGE, ha="center")
    layers = [("全连接 45→512", 8.3), ("全连接 512→256", 6.7), ("全连接 256→128", 5.1), ("全连接 128→12", 3.5)]
    for label, y in layers:
        _box(plt, axis, 6.9, y, 2.8, 0.95, label, ACCENT, 12)
    # Action + PD (right)
    _box(plt, axis, 11.6, 7.4, 3.0, 1.1, "动作 a（12维）\n关节位置目标偏移", GREEN, 12)
    _box(plt, axis, 11.6, 5.4, 3.0, 1.1, "×0.25 + 默认站姿 $q_0$\n目标关节角", GREEN, 12)
    _box(plt, axis, 11.6, 3.4, 3.0, 1.1, "PD 控制器\nKp=20，Kd=0.5", AMBER, 12)
    _box(plt, axis, 16.0, 5.4, 2.6, 1.1, "关节力矩 τ\n→ Go2 12 关节", PURPLE, 12)
    # Arrows
    for _, y in obs_items:
        _arrow(plt, axis, 4.5, y + 0.36, 6.9, 6.2)
    _arrow(plt, axis, 9.7, 3.97, 11.6, 7.9)
    _arrow(plt, axis, 13.1, 7.4, 13.1, 6.6)
    _arrow(plt, axis, 13.1, 5.4, 13.1, 4.6)
    _arrow(plt, axis, 14.6, 3.95, 16.4, 5.4)
    # Feedback
    _arrow(plt, axis, 17.3, 5.4, 17.3, 1.7)
    _arrow(plt, axis, 17.3, 1.7, 2.6, 1.7)
    _arrow(plt, axis, 2.6, 1.7, 2.6, 3.6)
    axis.text(10.0, 1.85, "状态反馈（IMU / 关节编码器，50 Hz）", fontsize=11, color=MUTED, ha="center")
    axis.text(
        0.7, 0.5,
        "控制参数：仿真步长 dt=0.002 s，decimation=10，策略推理频率 50 Hz；策略仅以本体感知为输入，不依赖视觉，满足赛题低算力部署要求。",
        fontsize=11, color=MUTED,
    )
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def make_llm_planning_diagram(output: Path) -> None:
    plt, fig, axis = _canvas("图 · 大模型约束式任务规划与安全仲裁流程")
    _box(plt, axis, 0.7, 7.6, 3.0, 1.3, "自然语言任务\n“巡检园区重点设施，\n发现异常立即上报”", PANEL, 11.5)
    _box(plt, axis, 4.6, 7.6, 3.0, 1.3, "大模型任务解析\n生成结构化任务 JSON", ACCENT, 12)
    _box(plt, axis, 8.6, 7.6, 3.4, 1.3, "规则校验\nJSON Schema / 白名单 / 参数边界", AMBER, 11.5)
    _box(plt, axis, 13.0, 7.6, 2.6, 1.3, "行为树 / 状态机\n任务执行", GREEN, 12)
    _box(plt, axis, 8.6, 4.9, 3.4, 1.2, "校验不通过\n拒绝执行 / 人工确认 / 规则回退", WINE, 11.5)
    _box(plt, axis, 13.0, 4.9, 2.6, 1.2, "安全监督器\n最高优先级仲裁", WINE, 12)
    _box(plt, axis, 16.3, 7.6, 2.4, 1.3, "速度指令下发\n路线控制器", GREEN, 12)
    _box(plt, axis, 16.3, 4.9, 2.4, 1.2, "限速 0.5 m/s\n禁行区 / 姿态约束 / 急停", PANEL, 10.5)
    _arrow(plt, axis, 3.7, 8.25, 4.6, 8.25)
    _arrow(plt, axis, 7.6, 8.25, 8.6, 8.25, "任务 JSON")
    _arrow(plt, axis, 12.0, 8.25, 13.0, 8.25, "校验通过")
    _arrow(plt, axis, 10.3, 7.6, 10.3, 6.2)
    _arrow(plt, axis, 15.6, 8.25, 16.3, 8.25)
    _arrow(plt, axis, 14.3, 7.6, 14.3, 6.2, "任意时刻可中断", lx=-2.2)
    _arrow(plt, axis, 15.6, 5.5, 16.3, 5.5)
    axis.text(
        0.7, 2.9,
        "设计要点：大模型只负责任务语义解析，不直接输出电机控制量；所有任务必须经规则校验后才可执行；安全监督器独立于大模型，拥有最高控制权。",
        fontsize=11.5, color=MUTED,
    )
    axis.text(
        0.7, 2.2,
        "对应赛题要求：体现“AI 大模型驱动”的任务理解能力，同时保证机器人行为的可解释性、可校验性与安全性。",
        fontsize=11.5, color=MUTED,
    )
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def make_safety_state_diagram(output: Path) -> None:
    plt, fig, axis = _canvas("图 · 双场景任务执行状态机与安全优先级")
    states = [
        (0.9, 7.9, 2.3, 1.0, "任务解析", ACCENT),
        (4.0, 7.9, 2.3, 1.0, "安全检查\n限速/路线校验", AMBER),
        (7.1, 7.9, 2.6, 1.0, "巡逻 / 侦察执行", GREEN),
        (10.6, 7.9, 2.3, 1.0, "异常告警\n入侵/受阻", WINE),
        (13.8, 7.9, 2.3, 1.0, "告警处置\n记录上报", ACCENT),
        (13.8, 5.2, 2.3, 1.0, "局部重规划", ACCENT),
        (10.6, 5.2, 2.3, 1.0, "安全返航", GREEN),
        (7.1, 5.2, 2.6, 1.0, "任务完成", GREEN),
        (4.0, 5.2, 2.3, 1.0, "安全停车\n（姿态异常/急停）", WINE),
        (0.9, 5.2, 2.3, 1.0, "人工接管", PANEL),
    ]
    for x, y, w, h, label, color in states:
        _box(plt, axis, x, y, w, h, label, color, 11.5)
    _arrow(plt, axis, 3.2, 8.4, 4.0, 8.4)
    _arrow(plt, axis, 6.3, 8.4, 7.1, 8.4)
    _arrow(plt, axis, 9.7, 8.4, 10.6, 8.4, "发现异常", ly=0.18)
    _arrow(plt, axis, 12.9, 8.4, 13.8, 8.4)
    _arrow(plt, axis, 14.95, 7.9, 14.95, 6.3, "通道受阻", lx=1.05)
    _arrow(plt, axis, 13.8, 5.7, 12.9, 5.7)
    _arrow(plt, axis, 10.6, 5.7, 9.7, 5.7)
    _arrow(plt, axis, 8.4, 7.9, 8.4, 6.3, "巡检完成", lx=-1.0)
    _arrow(plt, axis, 5.15, 7.9, 5.15, 6.3, "姿态异常/急停", lx=-1.5)
    _arrow(plt, axis, 4.0, 5.7, 3.2, 5.7)
    _arrow(plt, axis, 5.15, 6.3, 5.15, 7.9, "故障排除后复位", lx=1.6, ly=-0.2)
    axis.text(
        0.9, 3.2,
        "安全优先级（高→低）：人工接管 / 急停 ＞ 姿态异常保护 ＞ 安全监督约束 ＞ 任务执行指令。任何状态下急停或姿态越限立即进入安全停车。",
        fontsize=11.5, color=MUTED,
    )
    axis.text(
        0.9, 2.5,
        "场景一映射：巡逻执行→发现入侵→告警上报→继续巡检→安全返航；场景二映射：侦察执行→通道受阻→局部重规划→绕行→安全返航。",
        fontsize=11.5, color=MUTED,
    )
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def make_provenance_diagram(output: Path) -> None:
    plt, fig, axis = _canvas("图 · 训练评测链路与素材可追溯体系")
    # Cloud lane
    axis.text(4.4, 9.7, "自强5000平台（云端）", fontsize=15, color=EDGE, ha="center")
    axis.add_patch(FancyBboxPatch((0.6, 4.4), 7.6, 4.9, boxstyle="round,pad=0.12", facecolor="#0a1626", edgecolor=MUTED, linewidth=1.2, linestyle="--"))
    _box(plt, axis, 1.1, 7.9, 3.1, 1.0, "PPO 强化学习训练\n域随机化 + 课程学习", ACCENT, 11)
    _box(plt, axis, 5.0, 7.9, 2.9, 1.0, "63 组正式评测\n21 地形 × 3 策略", ACCENT, 11)
    _box(plt, axis, 1.1, 6.0, 3.1, 1.0, "冻结冠军策略\nbaseline5001", GREEN, 11)
    _box(plt, axis, 5.0, 6.0, 2.9, 1.0, "评测数据快照\nCSV 固化入库", GREEN, 11)
    _box(plt, axis, 2.9, 4.8, 3.4, 0.9, "ONNX / JIT / checkpoint 导出", GREEN, 11)
    _arrow(plt, axis, 4.2, 8.4, 5.0, 8.4)
    _arrow(plt, axis, 2.65, 7.9, 2.65, 7.1)
    _arrow(plt, axis, 6.45, 7.9, 6.45, 7.1)
    _arrow(plt, axis, 4.2, 6.5, 5.0, 6.5)
    _arrow(plt, axis, 4.6, 6.0, 4.6, 5.8)
    # Hash handoff
    _box(plt, axis, 8.9, 5.6, 1.9, 1.2, "SHA-256\n哈希交接", AMBER, 11.5)
    _arrow(plt, axis, 6.3, 5.25, 8.9, 6.0)
    # Local lane
    axis.text(14.6, 9.7, "本地工作站（报告证据）", fontsize=15, color=EDGE, ha="center")
    axis.add_patch(FancyBboxPatch((11.0, 1.2), 7.6, 8.1, boxstyle="round,pad=0.12", facecolor="#0a1626", edgecolor=MUTED, linewidth=1.2, linestyle="--"))
    _box(plt, axis, 11.5, 7.6, 3.2, 1.0, "策略一致性核验\n45维→12维 数值比对", GREEN, 11)
    _box(plt, axis, 15.2, 7.6, 3.0, 1.0, "MuJoCo 双场景复跑\n真实策略闭环", GREEN, 11)
    _box(plt, axis, 11.5, 5.6, 3.2, 1.0, "任务演示视频 ×2\n物理仿真视频 ×2", PURPLE, 11)
    _box(plt, axis, 15.2, 5.6, 3.0, 1.0, "图表：训练曲线 / 评测对比\n架构图 / 证据拼图", PURPLE, 10.5)
    _box(plt, axis, 13.3, 3.4, 3.2, 1.0, "素材包 manifest.json\n逐项 SHA-256 校验", AMBER, 11)
    _arrow(plt, axis, 10.8, 6.2, 11.5, 8.0)
    _arrow(plt, axis, 14.7, 8.1, 15.2, 8.1)
    _arrow(plt, axis, 13.1, 7.6, 13.1, 6.7)
    _arrow(plt, axis, 16.7, 7.6, 16.7, 6.7)
    _arrow(plt, axis, 14.3, 5.6, 14.9, 4.5)
    _arrow(plt, axis, 15.6, 5.6, 15.6, 4.5)
    axis.text(
        0.9, 1.9,
        "可追溯性设计：训练日志、评测快照、策略权重、仿真素材全部经哈希校验入库；报告中每一张图都能回溯到生成命令与原始数据。",
        fontsize=11.5, color=MUTED,
    )
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/report_diagrams")
    out_dir.mkdir(parents=True, exist_ok=True)
    make_rl_network_diagram(out_dir / "d1_rl_policy_network.png")
    make_llm_planning_diagram(out_dir / "d2_llm_task_planning.png")
    make_safety_state_diagram(out_dir / "d3_safety_state_machine.png")
    make_provenance_diagram(out_dir / "d4_training_provenance.png")
    print("REPORT_DIAGRAMS_DONE")


if __name__ == "__main__":
    import sys

    main()
