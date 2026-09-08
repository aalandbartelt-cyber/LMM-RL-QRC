"""Read-only runtime preflight for a local or GPU compute instance.

It never installs packages, starts a simulator, submits a job, or writes files.
Run it in the target Isaac Lab environment before attempting training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODULES = (
    "torch",
    "isaaclab",
    "isaaclab_tasks",
    "rsl_rl",
    "mujoco",
    "onnxruntime",
    "legged_gym",
    "campus_rl_overrides",
)


def module_status(name: str) -> dict[str, object]:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        found = False
    return {"available": found}


def gpu_status() -> dict[str, object]:
    executable = shutil.which("nvidia-smi")
    result: dict[str, object] = {"nvidia_smi": executable is not None}
    if executable is None:
        return result
    try:
        completed = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        result["query_returncode"] = completed.returncode
        result["devices"] = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if completed.stderr.strip():
            result["stderr"] = completed.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
    return result


def build_report(modules: tuple[str, ...], policy: Path | None) -> dict[str, object]:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "modules": {name: module_status(name) for name in modules},
        "gpu": gpu_status(),
    }
    if policy is not None:
        report["policy"] = {"path": str(policy), "exists": policy.is_file()}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, help="optional exported .pt/.onnx policy path")
    parser.add_argument("--modules", nargs="*", default=list(DEFAULT_MODULES))
    parser.add_argument("--strict", action="store_true", help="return 1 if torch or isaaclab is missing")
    args = parser.parse_args()
    report = build_report(tuple(args.modules), args.policy)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict:
        modules = report["modules"]
        if not modules["torch"]["available"] or not modules["isaaclab"]["available"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
