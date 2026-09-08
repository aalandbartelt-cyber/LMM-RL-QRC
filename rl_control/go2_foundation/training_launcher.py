"""Safe construction of reproducible Unitree RL Lab training commands.

The launcher deliberately separates command construction from execution.  A
normal invocation prints a command plan and verifies the supplied upstream
checkout.  The caller must pass ``--execute`` before a simulator or trainer can
be started.  This makes it practical to copy the project to a GPU machine and
inspect every proposed training command first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Literal


TrainingStage = Literal["baseline", "campus"]
BASELINE_TASK = "Unitree-Go2-Velocity"
CAMPUS_TASK = "Unitree-Go2-Campus-Velocity"
UNITREE_RL_LAB_COMMIT = "1330cb0d4236c4abd6bf1efec05a4fce7ad79678"


@dataclass(frozen=True)
class TrainingRunSpec:
    """The small set of changes allowed between comparable training runs."""

    stage: TrainingStage
    seed: int
    max_iterations: int
    num_envs: int | None = None
    device: str | None = None
    headless: bool = True
    run_name: str | None = None
    resume: bool = False
    load_run: str | None = None
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("baseline", "campus"):
            raise ValueError("stage must be baseline or campus")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.num_envs is not None and self.num_envs <= 0:
            raise ValueError("num_envs must be positive when supplied")
        if self.device is not None and not self.device.strip():
            raise ValueError("device cannot be blank")
        for name in ("run_name", "load_run", "checkpoint"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} cannot be blank")
        if self.resume and (self.load_run is None or self.checkpoint is None):
            raise ValueError("resume requires load_run and checkpoint")
        if not self.resume and (self.load_run is not None or self.checkpoint is not None):
            raise ValueError("load_run and checkpoint require resume=True")

    @property
    def task_id(self) -> str:
        return BASELINE_TASK if self.stage == "baseline" else CAMPUS_TASK


def _checkout_commit(checkout: Path) -> str | None:
    marker = checkout / ".qrc-upstream-commit"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip() or None
    if not (checkout / ".git").exists():
        return None
    result = subprocess.run(
        ("git", "-C", str(checkout), "rev-parse", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def verify_unitree_rl_lab_checkout(root: str | Path) -> dict[str, object]:
    """Verify required files and reject a known-but-wrong upstream revision."""
    checkout = Path(root).expanduser().resolve()
    required = {
        "train_script": checkout / "scripts" / "rsl_rl" / "train.py",
        "go2_task_registration": checkout
        / "source"
        / "unitree_rl_lab"
        / "unitree_rl_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "go2"
        / "__init__.py",
    }
    files = {name: str(path) for name, path in required.items()}
    missing = [name for name, path in required.items() if not path.is_file()]
    commit = _checkout_commit(checkout)
    problems = [f"missing:{name}" for name in missing]
    if commit is not None and commit != UNITREE_RL_LAB_COMMIT:
        problems.append("commit_mismatch")
    return {
        "checkout": str(checkout),
        "files": files,
        "ready": not problems,
        "missing": missing,
        "problems": problems,
        "commit": commit,
        "expected_commit": UNITREE_RL_LAB_COMMIT,
    }


def build_train_command(
    checkout: str | Path,
    spec: TrainingRunSpec,
    *,
    python_executable: str = "python",
    extra_args: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Build the exact command without executing it.

    The campus task is registered by importing ``campus_rl`` before
    the upstream script builds its list of Gym task IDs.  The package must be
    installed into the same interpreter as Isaac Lab on the GPU machine.
    """
    report = verify_unitree_rl_lab_checkout(checkout)
    if not report["ready"]:
        raise FileNotFoundError(f"invalid unitree_rl_lab checkout; missing: {report['missing']}")
    train_script = report["files"]["train_script"]
    if spec.headless and any(arg == "--video" or arg.startswith("--video=") for arg in extra_args):
        raise ValueError("video capture is disabled for headless GPU training")

    common = ["--task", spec.task_id, "--seed", str(spec.seed), "--max_iterations", str(spec.max_iterations)]
    if spec.headless:
        common.insert(0, "--headless")
    if spec.num_envs is not None:
        common.extend(("--num_envs", str(spec.num_envs)))
    if spec.device is not None:
        common.extend(("--device", spec.device))
    if spec.run_name is not None:
        common.extend(("--run_name", spec.run_name))
    if spec.resume:
        common.append("--resume")
        common.extend(("--load_run", spec.load_run or ""))
        common.extend(("--checkpoint", spec.checkpoint or ""))
    common.extend(extra_args)
    if spec.stage == "baseline":
        return tuple([python_executable, train_script, *common])

    # `runpy` preserves the upstream script's normal CLI behaviour.  Its only
    # extra operation is the early import needed to register the campus task.
    script_dir = str(Path(train_script).parent)
    bootstrap = (
        "import sys; sys.path.insert(0, %r); import campus_rl; import runpy; "
        "runpy.run_path(%r, run_name='__main__')"
    ) % (script_dir, train_script)
    return tuple([python_executable, "-c", bootstrap, *common])
