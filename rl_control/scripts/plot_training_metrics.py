#!/usr/bin/env python3
"""Turn one or more RSL-RL text logs into CSV curves and a PNG figure."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Iterable


ITERATION = re.compile(r"Learning iteration\s+(\d+)/(\d+)")
FIELDS = {
    "value_function_loss": re.compile(r"Value function loss:\s*([-+0-9.eE]+)"),
    "surrogate_loss": re.compile(r"Surrogate loss:\s*([-+0-9.eE]+)"),
    "mean_reward": re.compile(r"Mean reward:\s*([-+0-9.eE]+)"),
    "mean_episode_length": re.compile(r"Mean episode length:\s*([-+0-9.eE]+)"),
    "action_noise_std": re.compile(r"Mean action noise std:\s*([-+0-9.eE]+)"),
    "iteration_time_s": re.compile(r"Iteration time:\s*([-+0-9.eE]+)s"),
}


def parse_training_log(text: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    current: dict[str, float | int] | None = None
    for line in text.splitlines():
        iteration_match = ITERATION.search(line)
        if iteration_match:
            if current is not None:
                rows.append(current)
            current = {
                "iteration": int(iteration_match.group(1)),
                "target_iteration": int(iteration_match.group(2)),
            }
            continue
        if current is None:
            continue
        for name, pattern in FIELDS.items():
            match = pattern.search(line)
            if match:
                current[name] = float(match.group(1))
                break
    if current is not None:
        rows.append(current)
    if not rows:
        raise ValueError("No 'Learning iteration' records were found in the log")
    return rows


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    result: list[float] = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        result.append(total / min(index + 1, window))
    return result


def parse_log_argument(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError("Log label cannot be empty")
        return label, Path(raw_path)
    path = Path(value)
    return path.stem, path


def write_csv(rows_by_run: dict[str, list[dict[str, float | int]]], output: Path) -> None:
    fieldnames = ["run", "iteration", "target_iteration", *FIELDS]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for run, rows in rows_by_run.items():
            for row in rows:
                writer.writerow({"run": run, **row})


def render_plot(
    rows_by_run: dict[str, list[dict[str, float | int]]],
    output: Path,
    smooth_window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = (
        ("mean_reward", "Mean reward"),
        ("mean_episode_length", "Mean episode length"),
        ("value_function_loss", "Value function loss"),
        ("surrogate_loss", "Surrogate loss"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for axis, (field, title) in zip(axes.flat, metrics):
        for run, rows in rows_by_run.items():
            filtered = [row for row in rows if field in row]
            if not filtered:
                continue
            x = [int(row["iteration"]) for row in filtered]
            y = moving_average([float(row[field]) for row in filtered], smooth_window)
            axis.plot(x, y, linewidth=1.5, label=run)
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def load_logs(arguments: Iterable[str]) -> dict[str, list[dict[str, float | int]]]:
    rows_by_run: dict[str, list[dict[str, float | int]]] = {}
    for argument in arguments:
        label, path = parse_log_argument(argument)
        path = path.expanduser().resolve()
        if label in rows_by_run:
            raise ValueError(f"Duplicate run label: {label}")
        rows_by_run[label] = parse_training_log(path.read_text(encoding="utf-8", errors="replace"))
    return rows_by_run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, help="LABEL=PATH; repeat to compare runs")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--csv", type=Path, help="output CSV path; defaults beside the PNG")
    parser.add_argument("--smooth-window", type=int, default=25)
    args = parser.parse_args()
    if args.smooth_window <= 0:
        parser.error("--smooth-window must be positive")
    rows_by_run = load_logs(args.log)
    csv_output = args.csv or args.output.with_suffix(".csv")
    write_csv(rows_by_run, csv_output)
    render_plot(rows_by_run, args.output, args.smooth_window)
    print(f"PLOT={args.output.resolve()}")
    print(f"CSV={csv_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
