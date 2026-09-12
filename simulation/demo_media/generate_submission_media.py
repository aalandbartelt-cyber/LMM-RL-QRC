"""Generate the local submission media package for the DG-202609 report.

All artifacts are produced locally: MuJoCo route-following videos driven by the
verified baseline5001 checkpoint, Matplotlib report figures, and a SHA-256
manifest. Nothing here contacts the cloud training platform.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
import sys
import time
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np

from .mujoco_route import RouteResult, compact_route, simulate_route
from .rendering import COLORS, font, render_frame, render_title_card
from .scenarios import Scenario, build_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("D:/上海大学__/26挑战杯擂台赛/outputs/baseline5001_runtime_20260911")
# MuJoCo's C file loader cannot open paths containing non-ASCII characters, so
# the policy and Go2 MJCF assets are mirrored to a pure-ASCII location.
ASCII_RUNTIME_ROOT = Path("C:/Users/mary3/go2_runtime/baseline5001_20260911")
DEFAULT_POLICY = ASCII_RUNTIME_ROOT / "policy" / "model_5001.pt"
DEFAULT_GO2_XML = ASCII_RUNTIME_ROOT / "resources" / "go2" / "go2.xml"
PLATFORM_SHOT = Path("D:/上海大学__/26挑战杯擂台赛/素材/上海大学自强5000平台训练截图.PNG")
FORMAL_DIR = REPO_ROOT / "evaluation" / "results" / "go2_formal_matrix_20260911"
CLIPS_DIR = REPO_ROOT / "evaluation" / "results" / "go2_video_evidence_20260730" / "video_evidence_20260730"

FRAME_W, FRAME_H = 1920, 1080
VIDEO_FILES = {
    "campus_security": "01_campus_security_simulation.mp4",
    "disaster_response": "02_disaster_response_simulation.mp4",
}
HERO_FILES = {
    "campus_security": "03_hero_campus_security.png",
    "disaster_response": "04_hero_disaster_response.png",
}
HERO_TARGET_PROGRESS = {"campus_security": 0.30, "disaster_response": 0.62}


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_video(path: str | Path, min_frames: int, width: int, height: int, fps: float) -> bool:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return False
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if frames <= 0:
        frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frames += 1
    capture.release()
    duration = frames / actual_fps if actual_fps > 1e-6 else 0.0
    return (
        frames >= min_frames
        and actual_w == width
        and actual_h == height
        and abs(actual_fps - fps) <= 1.0
        and duration > 0.0
    )


# --- TensorBoard scalar log parsing (dependency-free TFRecord/protobuf reader) ---


def _read_varint(buffer: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buffer[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _skip_field(buffer: bytes, offset: int, wire: int) -> int:
    if wire == 0:
        _, offset = _read_varint(buffer, offset)
        return offset
    if wire == 1:
        return offset + 8
    if wire == 2:
        length, offset = _read_varint(buffer, offset)
        return offset + length
    if wire == 5:
        return offset + 4
    raise ValueError(f"unsupported protobuf wire type {wire}")


def _parse_value(payload: bytes) -> tuple[str, float | None]:
    tag = ""
    simple: float | None = None
    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 2:
            length, offset = _read_varint(payload, offset)
            tag = payload[offset : offset + length].decode("utf-8", "replace")
            offset += length
        elif field == 2 and wire == 5:
            simple = struct.unpack_from("<f", payload, offset)[0]
            offset += 4
        else:
            offset = _skip_field(payload, offset, wire)
    return tag, simple


def _parse_summary(payload: bytes) -> list[tuple[str, float | None]]:
    values: list[tuple[str, float | None]] = []
    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field, wire = key >> 3, key & 7
        if field == 1 and wire == 2:
            length, offset = _read_varint(payload, offset)
            values.append(_parse_value(payload[offset : offset + length]))
            offset += length
        else:
            offset = _skip_field(payload, offset, wire)
    return values


def _parse_event(payload: bytes) -> tuple[int, list[tuple[str, float | None]]]:
    step = 0
    values: list[tuple[str, float | None]] = []
    offset = 0
    while offset < len(payload):
        key, offset = _read_varint(payload, offset)
        field, wire = key >> 3, key & 7
        if field == 2 and wire == 0:
            step, offset = _read_varint(payload, offset)
        elif field == 5 and wire == 2:
            length, offset = _read_varint(payload, offset)
            values = _parse_summary(payload[offset : offset + length])
            offset += length
        else:
            offset = _skip_field(payload, offset, wire)
    return step, values


def read_scalar_series(events_path: str | Path) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    data = Path(events_path).read_bytes()
    offset = 0
    while offset + 12 <= len(data):
        length = struct.unpack_from("<Q", data, offset)[0]
        payload = data[offset + 12 : offset + 12 + length]
        offset += 12 + length + 4
        if len(payload) != length:
            break
        step, values = _parse_event(payload)
        for tag, simple in values:
            if simple is not None:
                series.setdefault(tag, []).append((step, simple))
    return series


# --- Report figures ---


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def make_formal_figure(output: Path) -> None:
    plt = _plt()
    with open(FORMAL_DIR / "overall_summary.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = (
        ("mean_success_rate", "平均成功率", "{:.1%}"),
        ("total_falls", "总摔倒次数", "{:.0f}"),
        ("mean_collision_rate", "平均碰撞率", "{:.3f}"),
        ("mean_tracking_rmse_mps", "速度跟踪 RMSE (m/s)", "{:.3f}"),
        ("mean_mechanical_power_w", "平均机械功率 (W)", "{:.1f}"),
        ("mean_tilt_rad", "平均机身倾角 (rad)", "{:.3f}"),
    )
    policies = [row["policy"] for row in rows]
    colors = ["#19c6e6", "#ffd166", "#9fb3c8"]
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#07111f")
    fig.suptitle("三策略 63 组正式评测对比（自强5000平台 · 2026-09-11 数据快照）", fontsize=22, color="#f2f7fb")
    for axis, (key, title, fmt) in zip(axes.flat, metrics):
        axis.set_facecolor("#0d1b2d")
        values = [float(row[key]) for row in rows]
        bars = axis.bar(policies, values, color=colors)
        axis.set_title(title, fontsize=15, color="#f2f7fb")
        axis.tick_params(colors="#9fb3c8", labelsize=11)
        for spine in axis.spines.values():
            spine.set_color("#203a55")
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=12,
                color="#f2f7fb",
            )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_training_figure(output: Path, events_path: Path) -> None:
    plt = _plt()
    series = read_scalar_series(events_path)
    reward_tag = next(
        (tag for tag in series if "mean_reward" in tag.lower()),
        next((tag for tag in series if "reward" in tag.lower()), None),
    )
    if reward_tag is None:
        raise RuntimeError(f"no reward scalar found in {events_path}: {sorted(series)}")
    length_tag = next((tag for tag in series if "episode_length" in tag.lower()), None)
    fig, axes = plt.subplots(1, 2 if length_tag else 1, figsize=(19.2, 10.8), dpi=100, squeeze=False)
    fig.patch.set_facecolor("#07111f")
    panels = [(reward_tag, "训练平均回合奖励")]
    if length_tag:
        panels.append((length_tag, "平均回合长度"))
    for axis, (tag, title) in zip(axes.flat, panels):
        points = series[tag]
        steps = [p[0] for p in points]
        values = [p[1] for p in points]
        axis.set_facecolor("#0d1b2d")
        axis.plot(steps, values, color="#19c6e6", linewidth=1.2, alpha=0.35)
        window = max(5, len(values) // 40)
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        axis.plot(steps[window - 1 :], smooth, color="#24d18f", linewidth=2.4, label="滑动平均")
        axis.set_title(f"{title}（{tag}）", fontsize=15, color="#f2f7fb")
        axis.set_xlabel("训练迭代次数", fontsize=12, color="#9fb3c8")
        axis.tick_params(colors="#9fb3c8")
        axis.legend(facecolor="#0d1b2d", labelcolor="#f2f7fb")
        for spine in axis.spines.values():
            spine.set_color("#203a55")
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_color("#9fb3c8")
    fig.suptitle("baseline5001 强化学习训练曲线（自强5000平台）", fontsize=22, color="#f2f7fb")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, facecolor=fig.get_facecolor())
    plt.close(fig)


def make_architecture_figure(output: Path) -> None:
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, axis = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor("#07111f")
    axis.set_xlim(0, 19.2)
    axis.set_ylim(0, 10.8)
    axis.axis("off")
    boxes = [
        (0.6, 6.1, 3.2, 1.7, "自然语言\n任务指令", "#11243a"),
        (4.4, 6.1, 3.4, 1.7, "大模型任务规划\n结构化任务 JSON", "#133449"),
        (8.4, 6.1, 3.4, 1.7, "安全监督器\n限速 / 禁行区 / 姿态约束", "#3a2b1a"),
        (12.4, 6.1, 3.0, 1.7, "路线控制器\n航点 → 速度指令", "#11243a"),
        (16.0, 6.1, 2.7, 1.7, "RL 运动策略\nbaseline5001", "#123a2c"),
        (13.5, 2.2, 5.2, 1.7, "MuJoCo 高保真动力学\nGo2 四足机器人", "#241a3a"),
        (4.4, 2.2, 4.6, 1.7, "自强5000平台\n云端训练 + 63组正式评测", "#3a1a24"),
    ]
    for x, y, w, h, label, color in boxes:
        axis.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", facecolor=color, edgecolor="#19c6e6", linewidth=2))
        axis.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=14, color="#f2f7fb")
    arrows = [
        (3.9, 6.95, 4.3, 6.95),
        (7.9, 6.95, 8.3, 6.95),
        (11.9, 6.95, 12.3, 6.95),
        (15.5, 6.95, 15.9, 6.95),
        (17.35, 6.0, 17.35, 4.0),
        (16.2, 4.0, 16.2, 6.0),
        (13.4, 3.4, 13.9, 6.0),
        (9.1, 3.05, 13.4, 3.05),
    ]
    for x1, y1, x2, y2 in arrows:
        axis.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, color="#19c6e6", linewidth=2))
    axis.text(15.35, 7.2, "速度指令", fontsize=11, color="#9fb3c8", ha="right")
    axis.text(17.55, 5.0, "关节目标", fontsize=11, color="#9fb3c8", va="center")
    axis.text(16.0, 5.0, "状态观测", fontsize=11, color="#9fb3c8", ha="right", va="center")
    axis.text(13.15, 4.7, "状态反馈", fontsize=11, color="#9fb3c8", rotation=63, va="center")
    axis.text(11.2, 3.3, "策略权重下发（哈希交接）", fontsize=11, color="#9fb3c8", ha="center")
    axis.text(0.6, 0.7, "证据边界：本地 MuJoCo 仿真用于路线演示与报告可视化；量化成绩以自强5000平台 63 组正式评测为准。", fontsize=12, color="#9fb3c8")
    axis.set_title("系统架构｜AI大模型 + 强化学习 四足机器人智能控制", fontsize=22, color="#f2f7fb", pad=18)
    fig.savefig(output, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# --- Contact sheet ---


def _thumbnail(image, size):
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    thumb = image.copy()
    thumb.thumbnail(size, Image.Resampling.LANCZOS)
    return thumb


def make_contact_sheet(output: Path, images: list[tuple[str, object]], columns: int = 3) -> None:
    from PIL import Image, ImageDraw

    cell_w, cell_h = 600, 440
    caption_h = 54
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * cell_w + 40, rows * (cell_h + caption_h) + 90), COLORS["background"])
    draw = ImageDraw.Draw(sheet)
    title = "提交证据一览｜本地仿真演示 + 云端正式评测"
    draw.text((20, 22), title, font=font(30, True), fill=COLORS["text"])
    for index, (caption, image) in enumerate(images):
        col, row = index % columns, index // columns
        x = 20 + col * cell_w
        y = 70 + row * (cell_h + caption_h)
        thumb = _thumbnail(image, (cell_w - 16, cell_h - 16))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + (cell_h - thumb.height) // 2))
        draw.rectangle((x, y + cell_h, x + cell_w - 8, y + cell_h + caption_h - 10), fill="#0d1b2d")
        draw.text((x + 12, y + cell_h + 10), caption, font=font(19), fill=COLORS["text"])
    sheet.save(output)


# --- Physics-driven scenario video ---


def _route_distance_progress(route, segment_index: int, x: float, y: float, floor: float) -> float:
    lengths = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(route[:-1], route[1:])]
    total = sum(lengths)
    if total <= 0.0:
        return 0.0
    index = min(max(0, segment_index), len(lengths) - 1)
    a, b = route[index], route[index + 1]
    vx, vy = b.x - a.x, b.y - a.y
    seg_sq = vx * vx + vy * vy
    ratio = 0.0 if seg_sq == 0.0 else min(1.0, max(0.0, ((x - a.x) * vx + (y - a.y) * vy) / seg_sq))
    travelled = sum(lengths[:index]) + ratio * math.sqrt(seg_sq)
    return max(floor, min(1.0, travelled / total))


def _result_card(scenario: Scenario, result: RouteResult | None) -> np.ndarray:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (FRAME_W, FRAME_H), COLORS["background"])
    draw = ImageDraw.Draw(image)
    title_width = draw.textlength(scenario.title, font=font(46, True))
    draw.text(((FRAME_W - title_width) / 2, 300), scenario.title, font=font(46, True), fill=COLORS["text"])
    passed = result is not None and result.status == "PASS"
    status = "任务路线完成 · 策略运行稳定" if passed else "路线未完成 · 本次运行未达标"
    color = COLORS["green"] if passed else COLORS["red"]
    status_width = draw.textlength(status, font=font(40, True))
    draw.text(((FRAME_W - status_width) / 2, 420), status, font=font(40, True), fill=color)
    if result is not None:
        lines = (
            f"检查点完成：{result.checkpoints_reached} / {result.checkpoints_total}",
            f"路线用时：{result.elapsed_s:.1f} 秒",
            f"最低质心高度：{result.minimum_base_height_m:.3f} m（安全阈值 0.18 m）",
        )
        for index, line in enumerate(lines):
            line_width = draw.textlength(line, font=font(27))
            draw.text(((FRAME_W - line_width) / 2, 520 + index * 52), line, font=font(27), fill=COLORS["muted"])
    label_width = draw.textlength(scenario.evidence_label, font=font(23, True))
    draw.text(((FRAME_W - label_width) / 2, 800), scenario.evidence_label, font=font(23, True), fill=COLORS["cyan"])
    return np.asarray(image, dtype=np.uint8)


def render_scenario_video(
    scenario: Scenario,
    policy: Path,
    go2_xml: Path,
    output: Path,
    fps: int,
    duration_s: float,
    card_s: float,
    quick: bool,
) -> tuple[RouteResult | None, np.ndarray, int]:
    import cv2

    render_size = (960, 540) if quick else (1280, 720)
    frames, holder = simulate_route(
        scenario,
        policy,
        go2_xml,
        fps=fps,
        max_seconds=duration_s,
        render_size=render_size,
    )
    route = compact_route(scenario)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {output}")
    written = 0

    def write(rgb: np.ndarray) -> None:
        nonlocal written
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        written += 1

    title = render_title_card(scenario.title, scenario.subtitle, scenario.evidence_label)
    for _ in range(max(1, int(card_s * fps))):
        write(title)
    hero: np.ndarray | None = None
    hero_distance = float("inf")
    hero_target = HERO_TARGET_PROGRESS[scenario.key]
    floor = 0.0
    for physics in frames:
        segment = min(int(physics.progress * (len(route) - 1)), len(route) - 2)
        floor = _route_distance_progress(route, segment, physics.position[0], physics.position[1], floor)
        frame = render_frame(scenario, floor, physics.rgb)
        write(frame)
        gap = abs(floor - hero_target)
        if gap < hero_distance:
            hero_distance = gap
            hero = frame.copy()
    result = holder["result"]
    card = _result_card(scenario, result)
    for _ in range(max(1, int(card_s * fps))):
        write(card)
    writer.release()
    if hero is None:
        hero = render_frame(scenario, hero_target, None)
    return result, hero, written


# --- Manifest, README, orchestration ---


def _write_readme(output_dir: Path, entries: list[dict], results: dict[str, RouteResult | None]) -> Path:
    lines = [
        "# 本地提交媒体素材包（DG-202609）",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "生成方式：完全本地生成，未使用云端 GPU。",
        "",
        "## 证据边界（报告引用时请保留以下措辞）",
        "",
        "- 双场景演示视频与关键帧为**本地 MuJoCo 任务级仿真**，由正式评测第一名的 baseline5001 策略驱动，**不是真机实验**。",
        "- 量化成绩以自强5000平台 63 组正式评测数据快照为准（`evaluation/results/go2_formal_matrix_20260911/`）。",
        "- 训练曲线来自自强5000平台 TensorBoard 日志（随运行包哈希交接）。",
        "",
        "## 文件清单",
        "",
    ]
    for entry in entries:
        lines.append(f"- `{entry['file']}` — {entry['label']}（SHA-256 `{entry['sha256'][:16]}…`）")
    lines += ["", "## 路线运行结果", ""]
    for key, result in results.items():
        if result is None:
            lines.append(f"- {key}：无结果")
        else:
            lines.append(
                f"- {key}：{result.status}，检查点 {result.checkpoints_reached}/{result.checkpoints_total}，"
                f"用时 {result.elapsed_s:.1f}s，最低质心高度 {result.minimum_base_height_m:.3f}m"
            )
    lines += [
        "",
        "校验：`python -m simulation.demo_media.generate_submission_media --verify-only --output-dir <本目录>`",
    ]
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _clip_poster() -> tuple[str, object] | None:
    import cv2

    clip = CLIPS_DIR / "01_targeted_flat_forward.mp4"
    if not clip.exists():
        return None
    capture = cv2.VideoCapture(str(clip))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return "历史物理仿真片段（正式评测存档）", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def generate(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = args.fps or (12 if args.quick else 30)
    duration = args.duration or (14.0 if args.quick else 100.0)
    card_s = 1.0 if args.quick else 2.5
    policy = Path(args.policy)
    go2_xml = Path(args.go2_assets)
    for required in (policy, go2_xml):
        if not required.exists():
            raise FileNotFoundError(f"required input missing: {required}")

    scenarios = build_scenarios()
    results: dict[str, RouteResult | None] = {}
    heroes: dict[str, np.ndarray] = {}
    video_meta: dict[str, dict] = {}
    for key, scenario in scenarios.items():
        video_path = output_dir / VIDEO_FILES[key]
        print(f"[render] {key} -> {video_path.name} (fps={fps}, max {duration:.0f}s)", flush=True)
        result, hero, written = render_scenario_video(
            scenario, policy, go2_xml, video_path, fps, duration, card_s, args.quick
        )
        results[key] = result
        heroes[key] = hero
        status = result.status if result else "NO_RESULT"
        print(f"[render] {key} finished: {status}, frames={written}", flush=True)
        image_path = output_dir / HERO_FILES[key]
        from PIL import Image

        Image.fromarray(hero).save(image_path)
        video_meta[key] = {"min_frames": max(1, written - 2), "fps": fps}

    print("[figure] formal evaluation comparison", flush=True)
    make_formal_figure(output_dir / "05_formal_evaluation_comparison.png")
    print("[figure] training curves", flush=True)
    events = next((RUNTIME_ROOT / "training_logs").glob("events.out.tfevents.*"))
    make_training_figure(output_dir / "06_training_curves.png", events)
    print("[figure] system architecture", flush=True)
    make_architecture_figure(output_dir / "07_system_architecture.png")

    sheet_images: list[tuple[str, object]] = [
        ("园区安全巡检关键帧（本地仿真）", heroes["campus_security"]),
        ("灾害侦察与物资投送关键帧（本地仿真）", heroes["disaster_response"]),
        ("三策略正式评测对比（自强5000快照）", output_dir / "05_formal_evaluation_comparison.png"),
        ("训练曲线（自强5000 TensorBoard）", output_dir / "06_training_curves.png"),
    ]
    if PLATFORM_SHOT.exists():
        sheet_images.append(("自强5000平台正式评测与结果归档截图", PLATFORM_SHOT))
    poster = _clip_poster()
    if poster is not None:
        sheet_images.append(poster)
    from PIL import Image as PILImage

    resolved = [
        (caption, PILImage.open(img).convert("RGB") if isinstance(img, Path) else img)
        for caption, img in sheet_images
    ]
    print("[figure] evidence contact sheet", flush=True)
    make_contact_sheet(output_dir / "08_evidence_contact_sheet.png", resolved)

    results_path = output_dir / "route_results.json"
    results_path.write_text(
        json.dumps({key: asdict(result) if result else None for key, result in results.items()}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    labels = {
        VIDEO_FILES["campus_security"]: "园区安全巡检路线演示视频（本地仿真）",
        VIDEO_FILES["disaster_response"]: "灾害侦察与物资投送演示视频（本地仿真）",
        HERO_FILES["campus_security"]: "园区巡检报告关键帧（本地仿真）",
        HERO_FILES["disaster_response"]: "灾害响应报告关键帧（本地仿真）",
        "05_formal_evaluation_comparison.png": "三策略正式评测对比图（云端评测快照）",
        "06_training_curves.png": "baseline5001 训练曲线（云端日志）",
        "07_system_architecture.png": "系统架构图",
        "08_evidence_contact_sheet.png": "证据一览拼图",
        "route_results.json": "本地路线运行结构化结果",
    }
    entries = []
    for name in sorted(labels):
        path = output_dir / name
        entry = {
            "file": name,
            "label": labels[name],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if name.endswith(".mp4"):
            key = next(k for k, v in VIDEO_FILES.items() if v == name)
            entry["video"] = {"width": FRAME_W, "height": FRAME_H, **video_meta[key]}
        entries.append(entry)
    readme = _write_readme(output_dir, entries, results)
    entries.append(
        {"file": "README.md", "label": "素材包说明与证据边界", "bytes": readme.stat().st_size, "sha256": sha256_file(readme)}
    )
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator": "simulation.demo_media.generate_submission_media",
        "policy_checkpoint": str(policy),
        "go2_assets": str(go2_xml),
        "quick": bool(args.quick),
        "outputs": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("LOCAL_SUBMISSION_MEDIA_PASS", flush=True)
    return 0


def verify(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for entry in manifest["outputs"]:
        path = output_dir / entry["file"]
        if not path.exists():
            failures.append(f"missing: {entry['file']}")
            continue
        if path.stat().st_size != entry["bytes"]:
            failures.append(f"size mismatch: {entry['file']}")
        if sha256_file(path) != entry["sha256"]:
            failures.append(f"sha256 mismatch: {entry['file']}")
        video = entry.get("video")
        if video and not validate_video(
            path, video["min_frames"], video["width"], video["height"], video["fps"]
        ):
            failures.append(f"video invalid: {entry['file']}")
    for failure in failures:
        print(f"VERIFY_FAIL {failure}", flush=True)
    if failures:
        return 1
    print("LOCAL_SUBMISSION_MEDIA_VERIFY_PASS", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "submission_media"))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY))
    parser.add_argument("--go2-assets", default=str(DEFAULT_GO2_XML))
    parser.add_argument("--duration", type=float, default=None, help="max simulated seconds per scenario")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="short low-fps pipeline check")
    parser.add_argument("--verify-only", action="store_true", help="verify an existing package via manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only:
        return verify(args)
    return generate(args)


if __name__ == "__main__":
    sys.exit(main())
