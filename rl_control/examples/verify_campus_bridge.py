"""Read-only validation of the bridge against the supplied campus controller.

Usage (Windows):
    python -m examples.verify_campus_bridge "D:\\强化学习项目\\campus_security_go2\\mission_controller.py"
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from go2_foundation import CommandLimiter, command_from_mission_output


def load_module(path: Path):
    spec = spec_from_file_location("campus_mission_controller_readonly", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Python module from {path}")
    module = module_from_spec(spec)
    # dataclasses and other decorators resolve the module through sys.modules
    # while executing the file.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("provide the absolute path to mission_controller.py")
    path = Path(sys.argv[1]).resolve()
    module = load_module(path)
    controller = module.WaypointVelocityController()
    mission_command = controller.update(0.0, -16.0, 0.0)
    command = CommandLimiter().apply(command_from_mission_output(mission_command))
    print(
        {
            "source": str(path),
            "target": mission_command.target_name,
            "command": command,
            "mission_complete": mission_command.mission_complete,
            "read_only": True,
        }
    )


if __name__ == "__main__":
    main()
