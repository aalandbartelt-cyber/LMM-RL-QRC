"""Inspect or explicitly execute a Unitree RL Lab Go2 training command.

Default mode is dry-run: it validates an upstream checkout and prints JSON.
Use ``--execute`` only in the GPU training environment after reviewing that
output.  This script does not install packages or alter the upstream checkout.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from go2_foundation.training_launcher import TrainingRunSpec, build_train_command, verify_unitree_rl_lab_checkout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitree-rl-lab", required=True, type=Path, help="path to an existing official checkout")
    parser.add_argument("--stage", choices=("baseline", "campus"), required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-iterations", required=True, type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--device")
    parser.add_argument("--run-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load-run")
    parser.add_argument("--checkpoint")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--execute", action="store_true", help="start the generated command (default: only inspect it)")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="extra upstream arguments; place after --")
    args = parser.parse_args()

    spec = TrainingRunSpec(
        stage=args.stage,
        seed=args.seed,
        max_iterations=args.max_iterations,
        num_envs=args.num_envs,
        device=args.device,
        headless=not args.no_headless,
        run_name=args.run_name,
        resume=args.resume,
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    report = verify_unitree_rl_lab_checkout(args.unitree_rl_lab)
    try:
        command = build_train_command(
            args.unitree_rl_lab,
            spec,
            python_executable=sys.executable,
            extra_args=tuple(args.extra),
        )
    except FileNotFoundError as exc:
        print(json.dumps({"ready": False, "checkout": report, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2

    plan = {
        "ready": True,
        "mode": "execute" if args.execute else "dry_run",
        "stage": spec.stage,
        "task_id": spec.task_id,
        "checkout": report,
        "command": list(command),
        "shell_preview": shlex.join(command),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    return subprocess.run(command, cwd=Path(args.unitree_rl_lab), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
