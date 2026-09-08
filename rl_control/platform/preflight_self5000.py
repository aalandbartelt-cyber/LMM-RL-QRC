#!/usr/bin/env python3
"""Fail-closed host and runtime preflight for the self-5000 platform."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
LOCKS = json.loads((SCRIPT_DIR / "versions.json").read_text(encoding="utf-8"))


def version_tuple(value: str) -> tuple[int, ...]:
    parts = re.findall(r"\d+", value)
    return tuple(int(part) for part in parts[:4])


def version_at_least(actual: str, minimum: str) -> bool:
    actual_parts = version_tuple(actual)
    minimum_parts = version_tuple(minimum)
    width = max(len(actual_parts), len(minimum_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) >= minimum_parts + (0,) * (
        width - len(minimum_parts)
    )


def collect_gpus() -> list[dict[str, str]]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return []
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if completed.returncode != 0:
        return []
    devices: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 4:
            devices.append(
                {
                    "index": fields[0],
                    "name": fields[1],
                    "driver": fields[2],
                    "memory_mib": fields[3],
                }
            )
    return devices


def collect_host_snapshot() -> dict[str, Any]:
    libc_name, libc_version = platform.libc_ver()
    free_disk = shutil.disk_usage(Path.home()).free / (1024**3)
    return {
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
        "glibc": libc_version if libc_name == "glibc" else "unknown",
        "free_disk_gib": round(free_disk, 2),
        "gpus": collect_gpus(),
    }


def assess_host(
    snapshot: dict[str, Any],
    *,
    requested_gpus: int,
    allow_unsupported_a100: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    gpus = list(snapshot.get("gpus", []))

    if snapshot.get("system") != "Linux":
        errors.append("linux_required")
    if not version_at_least(str(snapshot.get("glibc", "0")), LOCKS["minimum_glibc"]):
        errors.append("glibc_too_old")
    if float(snapshot.get("free_disk_gib", 0.0)) < float(LOCKS["minimum_free_disk_gib"]):
        errors.append("insufficient_disk")
    if len(gpus) < requested_gpus:
        errors.append("insufficient_gpus")
    if any(
        not version_at_least(str(gpu.get("driver", "0")), LOCKS["minimum_nvidia_driver"])
        for gpu in gpus[:requested_gpus]
    ):
        errors.append("driver_too_old")

    has_a100 = any("A100" in str(gpu.get("name", "")).upper() for gpu in gpus[:requested_gpus])
    if has_a100 and not allow_unsupported_a100:
        errors.append("a100_not_supported_by_isaac_sim")
    elif has_a100:
        warnings.append("a100_experimental_physics_only")

    return {
        "ready": not errors,
        "requested_gpus": requested_gpus,
        "errors": errors,
        "warnings": warnings,
        "snapshot": snapshot,
        "requirements": LOCKS,
    }


def configure_unitree_model_dir(source: Path, model_dir: Path) -> bool:
    source = source.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    text = source.read_text(encoding="utf-8")
    replacement = f"UNITREE_MODEL_DIR = {json.dumps(str(model_dir))}"
    if replacement in text:
        return False
    placeholder = "UNITREE_MODEL_DIR = MISSING"
    if text.count(placeholder) != 1:
        raise RuntimeError("Unitree asset placeholder changed; refusing an unverified source edit")
    source.write_text(text.replace(placeholder, replacement, 1), encoding="utf-8")
    return True


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_checks(requested_gpus: int) -> dict[str, Any]:
    report: dict[str, Any] = {"ready": False, "errors": [], "versions": {}}
    app = None
    try:
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True).app
        import gymnasium as gym
        import torch
        import unitree_rl_lab.tasks  # noqa: F401
        import campus_rl  # noqa: F401

        report["versions"] = {
            "python": platform.python_version(),
            "isaac_sim": package_version("isaacsim"),
            "isaac_lab": package_version("isaaclab"),
            "rsl_rl": package_version("rsl-rl-lib"),
            "go2_campus_rl": package_version("go2-campus-rl"),
        }
        if sys.version_info[:2] != (3, 11):
            report["errors"].append("python_version_mismatch")
        if report["versions"]["isaac_sim"] != LOCKS["isaac_sim"]:
            report["errors"].append("isaac_sim_version_mismatch")
        if report["versions"]["rsl_rl"] != LOCKS["rsl_rl"]:
            report["errors"].append("rsl_rl_version_mismatch")
        if not torch.cuda.is_available():
            report["errors"].append("cuda_unavailable")
        if torch.cuda.device_count() < requested_gpus:
            report["errors"].append("torch_gpu_count_mismatch")
        allocations = []
        for index in range(min(requested_gpus, torch.cuda.device_count())):
            tensor = torch.ones((1024, 1024), device=f"cuda:{index}")
            value = float((tensor @ tensor).mean().item())
            torch.cuda.synchronize(index)
            allocations.append({"index": index, "finite": value == value})
        report["cuda_allocations"] = allocations
        task_count = sum(task_id == "Unitree-Go2-Campus-Velocity" for task_id in gym.registry.keys())
        report["campus_task_count"] = task_count
        if task_count != 1:
            report["errors"].append("campus_task_registration_mismatch")
    except Exception as exc:  # simulator startup failures must become a machine-readable gate
        report["errors"].append("runtime_exception")
        report["exception"] = f"{type(exc).__name__}: {exc}"
    finally:
        if app is not None:
            app.close()
    report["ready"] = not report["errors"]
    return report


def write_report(report: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("host", "runtime", "configure-assets"), required=True)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--allow-unsupported-a100", action="store_true")
    parser.add_argument("--unitree-config", type=Path)
    parser.add_argument("--unitree-model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gpus <= 0:
        parser.error("--gpus must be positive")

    if args.phase == "configure-assets":
        if args.unitree_config is None or args.unitree_model_dir is None:
            parser.error("configure-assets requires --unitree-config and --unitree-model-dir")
        changed = configure_unitree_model_dir(args.unitree_config, args.unitree_model_dir)
        report = {"ready": True, "changed": changed, "unitree_model_dir": str(args.unitree_model_dir)}
    elif args.phase == "host":
        report = assess_host(
            collect_host_snapshot(),
            requested_gpus=args.gpus,
            allow_unsupported_a100=args.allow_unsupported_a100,
        )
    else:
        report = runtime_checks(args.gpus)
    write_report(report, args.output)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
