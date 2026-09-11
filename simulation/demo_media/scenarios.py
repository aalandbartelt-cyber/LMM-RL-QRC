"""Deterministic mission definitions used by the local submission demo."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    name: str
    x: float
    y: float
    kind: str = "route"


@dataclass(frozen=True)
class MissionEvent:
    progress: float
    status: str
    detail: str
    level: str = "info"


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    subtitle: str
    task_prompt: str
    evidence_label: str
    max_speed_mps: float
    route: tuple[Point, ...]
    events: tuple[MissionEvent, ...]
    physics_clips: tuple[str, ...]


def _campus_scenario() -> Scenario:
    route = (
        Point("P0 主入口起点", 0.0, -16.0, "home"),
        Point("T01 南侧主路入口", 0.0, -12.0),
        Point("T02 西侧主路南端", -24.0, -12.0),
        Point("T03 西侧主路北端", -24.0, 5.0),
        Point("P1 办公楼门禁", -16.0, 5.0, "inspect"),
        Point("P2 西侧围墙盲区", -24.0, 5.0, "alert"),
        Point("T04 办公楼西北绕行点", -24.0, 18.0),
        Point("P3 北侧电子围栏", 0.0, 18.0, "inspect"),
        Point("T05 中央通道北端", 0.0, 5.0),
        Point("P4 仓库防火门", 15.0, 5.0, "inspect"),
        Point("T06 东侧主路北端", 24.0, 5.0),
        Point("T07 东侧主路南端", 24.0, -12.0),
        Point("P6 停车区", 13.0, -12.0, "inspect"),
        Point("T08 南侧主路中央", 0.0, -12.0),
        Point("T09 动力机房绕行点", -23.0, -12.0),
        Point("P5 动力机房", -22.0, -2.0, "inspect"),
        Point("T10 返航南侧主路", -23.0, -12.0),
        Point("T11 返航主路中央", 0.0, -12.0),
        Point("P7 主入口返航", 0.0, -16.0, "home"),
    )
    events = (
        MissionEvent(0.00, "任务解析", "自然语言任务转换为结构化巡检任务"),
        MissionEvent(0.07, "安全检查", "速度上限 0.50 m/s，路线与禁行区校验通过"),
        MissionEvent(0.15, "开始巡逻", "前往办公楼门禁与西侧围墙"),
        MissionEvent(0.29, "发现异常", "西侧围墙盲区检测到可疑入侵", "alert"),
        MissionEvent(0.36, "告警上报", "记录坐标、生成事件编号并上报", "alert"),
        MissionEvent(0.49, "继续巡检", "电子围栏、仓库防火门状态正常"),
        MissionEvent(0.71, "设施检查", "停车区与动力机房检查完成"),
        MissionEvent(0.90, "安全返航", "沿校验路线返回主入口"),
        MissionEvent(1.00, "任务完成", "完成 6 个重点点位检查并生成巡检报告"),
    )
    return Scenario(
        key="campus_security",
        title="场景一｜园区安全巡检",
        subtitle="自主巡逻 · 异常识别 · 安全告警 · 任务闭环",
        task_prompt="完成园区周界与重点设施巡检，发现异常立即上报并安全返航。",
        evidence_label="本地任务级仿真演示",
        max_speed_mps=0.50,
        route=route,
        events=events,
        physics_clips=("01_targeted_flat_forward.mp4", "02_targeted_flat_turn_left.mp4"),
    )


def _disaster_scenario() -> Scenario:
    route = (
        Point("S0 安全区", -26.0, -15.0, "home"),
        Point("S1 任务入口", -21.0, -10.0),
        Point("S2 碎石通道", -12.0, -6.0, "rough"),
        Point("S3 坡道", -5.0, 1.0, "slope"),
        Point("S4 烟雾边界", 2.0, 8.0, "hazard"),
        Point("S5 目标侦察点", 12.0, 11.0, "target"),
        Point("S6 物资投送点", 18.0, 4.0, "delivery"),
        Point("S7 通道受阻", 10.0, -3.0, "blocked"),
        Point("S8 重规划绕行", 1.0, -10.0, "detour"),
        Point("S9 碎石区出口", -12.0, -12.0, "rough"),
        Point("S10 安全区返航", -26.0, -15.0, "home"),
    )
    events = (
        MissionEvent(0.00, "任务解析", "生成侦察、投送与返航行为序列"),
        MissionEvent(0.08, "风险检查", "限速 0.35 m/s，允许坡道与台阶"),
        MissionEvent(0.17, "进入灾区", "通过碎石通道并持续监测姿态"),
        MissionEvent(0.34, "地形通过", "完成坡道与台阶区通行"),
        MissionEvent(0.50, "目标确认", "确认待救援目标与可达投送区域"),
        MissionEvent(0.62, "物资投送", "急救包投送完成，记录交付状态", "success"),
        MissionEvent(0.72, "通道受阻", "原返航通道被倒塌构件阻断", "alert"),
        MissionEvent(0.79, "局部重规划", "启用安全绕行路线，降低速度"),
        MissionEvent(0.91, "返航中", "退出高风险区域并返回安全区"),
        MissionEvent(1.00, "安全返航", "侦察、投送和返航任务全部完成", "success"),
    )
    return Scenario(
        key="disaster_response",
        title="场景二｜灾害侦察与物资投送",
        subtitle="复杂地形 · 目标搜索 · 轻载投送 · 动态重规划",
        task_prompt="从安全区进入受灾区域，侦察目标点，投送急救包并返回安全区。",
        evidence_label="本地任务级仿真演示",
        max_speed_mps=0.35,
        route=route,
        events=events,
        physics_clips=(
            "03_targeted_stairs_forward.mp4",
            "05_targeted_cross_slope.mp4",
            "06_targeted_race_track.mp4",
        ),
    )


def build_scenarios() -> dict[str, Scenario]:
    """Return the two frozen submission scenarios keyed by stable identifiers."""
    campus = _campus_scenario()
    disaster = _disaster_scenario()
    return {campus.key: campus, disaster.key: disaster}
