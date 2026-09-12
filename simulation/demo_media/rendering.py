"""High-contrast 1080p renderer for the local task-level simulation demo."""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .scenarios import MissionEvent, Point, Scenario, build_scenarios


WIDTH = 1920
HEIGHT = 1080
MAP_BOX = (40, 150, 1300, 990)
PANEL_BOX = (1330, 150, 1880, 990)

COLORS = {
    "background": "#07111f",
    "panel": "#0d1b2d",
    "panel_alt": "#11243a",
    "line": "#203a55",
    "text": "#f2f7fb",
    "muted": "#9fb3c8",
    "cyan": "#19c6e6",
    "blue": "#2f80ed",
    "green": "#24d18f",
    "yellow": "#ffd166",
    "red": "#ff5263",
    "road": "#283849",
    "grass": "#274b3a",
    "concrete": "#415469",
    "rubble": "#665a50",
    "smoke": "#53606d",
}


def _font_path() -> str | None:
    candidates = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    return str(next((path for path in candidates if path.exists()), candidates[-1]))


@lru_cache(maxsize=16)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if bold:
        bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
        if bold_path.exists():
            return ImageFont.truetype(str(bold_path), size)
    try:
        return ImageFont.truetype(_font_path(), size)
    except (OSError, TypeError):
        return ImageFont.load_default()


def interpolate_route(route: tuple[Point, ...], progress: float) -> tuple[float, float, float]:
    """Interpolate position and yaw by travelled distance along a route."""
    if len(route) < 2:
        raise ValueError("route must contain at least two points")
    progress = min(1.0, max(0.0, float(progress)))
    lengths = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(route[:-1], route[1:])]
    total = sum(lengths)
    if total <= 0.0:
        return route[0].x, route[0].y, 0.0
    target = progress * total
    travelled = 0.0
    for index, length in enumerate(lengths):
        if target <= travelled + length or index == len(lengths) - 1:
            a, b = route[index], route[index + 1]
            ratio = 0.0 if length <= 0.0 else (target - travelled) / length
            ratio = min(1.0, max(0.0, ratio))
            x = a.x + ratio * (b.x - a.x)
            y = a.y + ratio * (b.y - a.y)
            yaw = math.atan2(b.y - a.y, b.x - a.x)
            return x, y, yaw
        travelled += length
    return route[-1].x, route[-1].y, 0.0


def _world_to_screen(x: float, y: float) -> tuple[int, int]:
    left, top, right, bottom = MAP_BOX
    margin = 45
    sx = left + margin + (x + 30.0) / 60.0 * (right - left - 2 * margin)
    sy = bottom - margin - (y + 20.0) / 40.0 * (bottom - top - 2 * margin)
    return int(sx), int(sy)


def _rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill: str, outline: str | None = None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, fill: str = COLORS["text"], bold: bool = False):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def _wrap(draw: ImageDraw.ImageDraw, value: str, max_width: int, size: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for char in value:
        trial = current + char
        if current and draw.textlength(trial, font=font(size)) > max_width:
            lines.append(current)
            current = char
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy, value: str, max_width: int, size: int, fill: str, spacing: int = 8) -> int:
    x, y = xy
    line_height = size + spacing
    for line in _wrap(draw, value, max_width, size):
        _text(draw, (x, y), line, size, fill)
        y += line_height
    return y


def _draw_campus(draw: ImageDraw.ImageDraw):
    left, top, right, bottom = MAP_BOX
    _rounded(draw, MAP_BOX, 24, "#183128", COLORS["line"], 2)
    # Roads.
    for x, y, w, h in (
        (left + 90, top + 610, 1080, 120),
        (left + 90, top + 180, 1080, 120),
        (left + 90, top + 180, 120, 550),
        (left + 1050, top + 180, 120, 550),
        (left + 585, top + 300, 110, 310),
    ):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=12, fill=COLORS["road"])
    # Buildings.
    buildings = (
        (left + 220, top + 55, left + 510, top + 225, "办公研发楼", "#315d75"),
        (left + 750, top + 55, left + 1080, top + 225, "仓储中心", "#675e49"),
        (left + 215, top + 375, left + 385, top + 495, "动力机房", "#536173"),
        (left + 770, top + 380, left + 1100, top + 545, "停车区", "#3b4b5d"),
        (left + 545, top + 740, left + 680, top + 810, "门岗", "#637789"),
    )
    for x1, y1, x2, y2, label, color in buildings:
        _rounded(draw, (x1, y1, x2, y2), 12, color, "#7890a5", 2)
        tw = draw.textlength(label, font=font(22, True))
        _text(draw, ((x1 + x2 - tw) / 2, (y1 + y2) / 2 - 14), label, 22, bold=True)
    # Trees and gate.
    for x, y in ((120, 90), (1160, 90), (1140, 750), (100, 760), (610, 100), (450, 505)):
        draw.ellipse((left + x - 15, top + y - 15, left + x + 15, top + y + 15), fill="#3f8b55")
    _text(draw, (left + 555, bottom - 38), "园区主入口", 20, COLORS["muted"])


def _draw_disaster(draw: ImageDraw.ImageDraw):
    left, top, right, bottom = MAP_BOX
    _rounded(draw, MAP_BOX, 24, "#302d2a", COLORS["line"], 2)
    # Hazard zones.
    _rounded(draw, (left + 70, top + 590, left + 340, top + 790), 26, "#233d39", "#3e756a", 2)
    _text(draw, (left + 105, top + 735), "安全区", 26, COLORS["green"], True)
    _rounded(draw, (left + 260, top + 390, left + 570, top + 675), 28, COLORS["rubble"], "#958372", 2)
    _text(draw, (left + 340, top + 470), "碎石区", 27, COLORS["yellow"], True)
    for x, y, r in ((330, 510, 28), (430, 590, 35), (500, 455, 22), (385, 650, 20), (535, 560, 26)):
        draw.polygon(
            [(left + x - r, top + y + r // 2), (left + x, top + y - r), (left + x + r, top + y + r // 2)],
            fill="#8b7a69",
        )
    _rounded(draw, (left + 525, top + 225, left + 760, top + 510), 24, "#514d48", "#9d8b71", 2)
    _text(draw, (left + 585, top + 285), "坡道 / 台阶", 24, COLORS["yellow"], True)
    for yy in range(top + 335, top + 485, 24):
        draw.line((left + 550, yy, left + 730, yy), fill="#ac9a80", width=5)
    _rounded(draw, (left + 720, top + 90, left + 1040, top + 390), 45, "#3f4952", "#697786", 2)
    for offset in (0, 55, 105):
        draw.ellipse((left + 760 + offset, top + 135, left + 930 + offset, top + 305), fill="#68727b")
    _text(draw, (left + 815, top + 320), "烟雾区", 25, COLORS["muted"], True)
    _rounded(draw, (left + 925, top + 365, left + 1165, top + 610), 24, "#443c38", "#8b7160", 2)
    _text(draw, (left + 965, top + 420), "倒塌建筑", 25, COLORS["red"], True)
    draw.line((left + 960, top + 500, left + 1130, top + 555), fill="#b06b5d", width=20)
    draw.line((left + 1110, top + 470, left + 975, top + 570), fill="#986052", width=15)
    _rounded(draw, (left + 895, top + 135, left + 1175, top + 300), 22, "#263f4a", COLORS["cyan"], 3)
    _text(draw, (left + 945, top + 185), "目标 / 投送区", 26, COLORS["cyan"], True)


def _draw_route(draw: ImageDraw.ImageDraw, scenario: Scenario, progress: float):
    points = [_world_to_screen(p.x, p.y) for p in scenario.route]
    if len(points) >= 2:
        draw.line(points, fill="#607991", width=7, joint="curve")
    # Progress route follows the same distance measure as the robot.
    segment_lengths = [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(scenario.route[:-1], scenario.route[1:])]
    target = min(1.0, max(0.0, progress)) * sum(segment_lengths)
    travelled = 0.0
    completed = [points[0]]
    for index, length in enumerate(segment_lengths):
        if target >= travelled + length:
            completed.append(points[index + 1])
            travelled += length
            continue
        ratio = 0.0 if length == 0 else (target - travelled) / length
        x = points[index][0] + ratio * (points[index + 1][0] - points[index][0])
        y = points[index][1] + ratio * (points[index + 1][1] - points[index][1])
        completed.append((int(x), int(y)))
        break
    if len(completed) >= 2:
        draw.line(completed, fill=COLORS["cyan"], width=10, joint="curve")
    for point, screen in zip(scenario.route, points):
        if point.kind == "route":
            radius, color = 5, COLORS["muted"]
        elif point.kind in {"home", "inspect", "target", "delivery"}:
            radius, color = 10, COLORS["green"]
        elif point.kind in {"alert", "hazard", "blocked"}:
            radius, color = 12, COLORS["red"]
        else:
            radius, color = 8, COLORS["yellow"]
        draw.ellipse((screen[0] - radius, screen[1] - radius, screen[0] + radius, screen[1] + radius), fill=color, outline="#eaf5ff", width=2)
        if point.kind != "route":
            _text(draw, (screen[0] + 13, screen[1] - 12), point.name, 17, COLORS["text"])


def _draw_robot(image: Image.Image, x: int, y: int, yaw: float):
    marker = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    draw = ImageDraw.Draw(marker)
    draw.ellipse((24, 36, 96, 84), fill="#0a2133", outline=COLORS["cyan"], width=5)
    draw.rounded_rectangle((38, 42, 83, 78), radius=8, fill="#e6f5fb", outline=COLORS["cyan"], width=3)
    for x1, y1, x2, y2 in ((28, 39, 12, 24), (28, 80, 12, 96), (92, 39, 108, 24), (92, 80, 108, 96)):
        draw.line((x1, y1, x2, y2), fill="#dbeaf0", width=7)
        draw.ellipse((x2 - 5, y2 - 5, x2 + 5, y2 + 5), fill=COLORS["cyan"])
    draw.polygon(((95, 60), (78, 48), (78, 72)), fill=COLORS["red"])
    marker = marker.rotate(-math.degrees(yaw), resample=Image.Resampling.BICUBIC, expand=False)
    image.alpha_composite(marker, (x - 60, y - 60))


def _current_event(events: Iterable[MissionEvent], progress: float) -> MissionEvent:
    current = next(iter(events))
    for event in events:
        if event.progress <= progress + 1e-9:
            current = event
        else:
            break
    return current


def _draw_right_panel(image: Image.Image, scenario: Scenario, progress: float, physics_frame: np.ndarray | None):
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = PANEL_BOX
    _rounded(draw, PANEL_BOX, 24, COLORS["panel"], COLORS["line"], 2)
    _text(draw, (left + 28, top + 24), "任务指令", 21, COLORS["cyan"], True)
    y = _draw_wrapped(draw, (left + 28, top + 61), scenario.task_prompt, right - left - 56, 24, COLORS["text"], 9)
    y += 18
    draw.line((left + 28, y, right - 28, y), fill=COLORS["line"], width=2)
    y += 20
    _text(draw, (left + 28, y), "系统链路", 21, COLORS["cyan"], True)
    y += 40
    modules = ("大模型任务规划", "安全监督器", "路线控制器", "RL运动策略")
    module_width = 112
    gap = 12
    for index, module in enumerate(modules):
        x = left + 28 + index * (module_width + gap)
        active = progress >= index * 0.06
        _rounded(draw, (x, y, x + module_width, y + 58), 10, COLORS["panel_alt"] if active else "#17202c", COLORS["cyan"] if active else COLORS["line"], 2)
        for line_index, line in enumerate(_wrap(draw, module, module_width - 14, 17)):
            _text(draw, (x + 8, y + 7 + line_index * 21), line, 17, COLORS["text"] if active else COLORS["muted"])
        if index < len(modules) - 1:
            _text(draw, (x + module_width + 1, y + 17), "›", 30, COLORS["cyan"], True)
    y += 83
    event = _current_event(scenario.events, progress)
    event_color = COLORS["red"] if event.level == "alert" else COLORS["green"] if event.level == "success" else COLORS["cyan"]
    _rounded(draw, (left + 28, y, right - 28, y + 104), 14, "#13283d", event_color, 3)
    _text(draw, (left + 48, y + 15), event.status, 28, event_color, True)
    _draw_wrapped(draw, (left + 48, y + 55), event.detail, right - left - 96, 19, COLORS["text"], 5)
    y += 127
    _text(draw, (left + 28, y), "安全与任务状态", 21, COLORS["cyan"], True)
    y += 40
    metrics = (
        ("任务进度", f"{progress * 100:5.1f}%"),
        ("速度上限", f"{scenario.max_speed_mps:.2f} m/s"),
        ("安全状态", "受控 / 正常" if event.level != "alert" else "告警已处置"),
    )
    for index, (label, value) in enumerate(metrics):
        yy = y + index * 40
        _text(draw, (left + 28, yy), label, 18, COLORS["muted"])
        vw = draw.textlength(value, font=font(20, True))
        _text(draw, (right - 28 - vw, yy - 1), value, 20, COLORS["text"], True)
    y += 132
    _text(draw, (left + 28, y), "Go2 checkpoint 5001 物理仿真证据", 19, COLORS["yellow"], True)
    inset = (left + 28, y + 34, right - 28, bottom - 28)
    _rounded(draw, inset, 12, "#05090e", COLORS["yellow"], 2)
    if physics_frame is not None and physics_frame.size:
        frame_image = Image.fromarray(physics_frame.astype(np.uint8), mode="RGB")
        target_w = inset[2] - inset[0] - 8
        target_h = inset[3] - inset[1] - 8
        frame_image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)
        px = inset[0] + (target_w - frame_image.width) // 2 + 4
        py = inset[1] + (target_h - frame_image.height) // 2 + 4
        image.paste(frame_image, (px, py))
    else:
        message = "策略物理仿真片段将在成片中载入"
        tw = draw.textlength(message, font=font(18))
        _text(draw, ((inset[0] + inset[2] - tw) / 2, (inset[1] + inset[3]) / 2 - 12), message, 18, COLORS["muted"])


def render_frame(scenario: Scenario, progress: float, physics_frame: np.ndarray | None) -> np.ndarray:
    """Render one 1920×1080 RGB frame for a frozen scenario state."""
    progress = min(1.0, max(0.0, float(progress)))
    image = Image.new("RGBA", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    _text(draw, (42, 28), scenario.title, 42, COLORS["text"], True)
    _text(draw, (44, 83), scenario.subtitle, 22, COLORS["muted"])
    label_width = draw.textlength(scenario.evidence_label, font=font(21, True))
    _rounded(draw, (WIDTH - label_width - 86, 35, WIDTH - 40, 82), 16, "#133449", COLORS["cyan"], 2)
    _text(draw, (WIDTH - label_width - 63, 46), scenario.evidence_label, 21, COLORS["cyan"], True)
    if scenario.key == "campus_security":
        _draw_campus(draw)
    elif scenario.key == "disaster_response":
        _draw_disaster(draw)
    else:
        raise ValueError(f"unsupported scenario: {scenario.key}")
    _draw_route(draw, scenario, progress)
    x, y, yaw = interpolate_route(scenario.route, progress)
    sx, sy = _world_to_screen(x, y)
    _draw_robot(image, sx, sy, yaw)
    # Map footer.
    draw = ImageDraw.Draw(image)
    _rounded(draw, (65, 925, 1275, 972), 12, "#0a1929", COLORS["line"], 1)
    _text(draw, (85, 936), "青色：已执行路线   灰色：规划路线   绿色：任务点   红色：风险/异常点", 19, COLORS["muted"])
    _draw_right_panel(image, scenario, progress, physics_frame)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def render_title_card(title: str, subtitle: str, badge: str) -> np.ndarray:
    """Render a full-HD title or section card for the combined video."""
    image = Image.new("RGB", (WIDTH, HEIGHT), COLORS["background"])
    draw = ImageDraw.Draw(image)
    for radius, alpha_color in ((420, "#0a253c"), (310, "#0d3350"), (190, "#12506a")):
        draw.ellipse((WIDTH // 2 - radius, HEIGHT // 2 - radius, WIDTH // 2 + radius, HEIGHT // 2 + radius), fill=alpha_color)
    tw = draw.textlength(title, font=font(58, True))
    _text(draw, ((WIDTH - tw) / 2, 390), title, 58, COLORS["text"], True)
    sw = draw.textlength(subtitle, font=font(28))
    _text(draw, ((WIDTH - sw) / 2, 485), subtitle, 28, COLORS["muted"])
    bw = draw.textlength(badge, font=font(23, True))
    _rounded(draw, ((WIDTH - bw) / 2 - 28, 575, (WIDTH + bw) / 2 + 28, 625), 18, "#16394e", COLORS["cyan"], 2)
    _text(draw, ((WIDTH - bw) / 2, 587), badge, 23, COLORS["cyan"], True)
    _text(draw, (66, HEIGHT - 68), "DG-202609｜AI大模型与强化学习驱动的四足机器人智能控制系统", 21, COLORS["muted"])
    return np.asarray(image, dtype=np.uint8)


def preview_frames() -> dict[str, np.ndarray]:
    """Convenience helper used by the generator and visual smoke tests."""
    return {key: render_frame(scenario, 0.62, None) for key, scenario in build_scenarios().items()}
