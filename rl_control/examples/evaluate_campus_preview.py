"""Read-only acceptance check for the current campus route preview.

This intentionally labels the output as ``kinematic_preview``.  It validates
P0-P7 ordering and mission timing, not learned locomotion.

Usage:
    python -m examples.evaluate_campus_preview "D:\\强化学习项目\\campus_security_go2\\mission_controller.py"
"""

import json
from pathlib import Path
import sys

from go2_foundation import RouteProgressEvaluator, RouteSample

from .verify_campus_bridge import load_module


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("provide the absolute path to mission_controller.py")
    source = Path(sys.argv[1]).resolve()
    module = load_module(source)
    patrol = module.KinematicPatrol()
    evaluator = RouteProgressEvaluator.from_waypoint_objects(module.SECURITY_WAYPOINTS)
    dt = 0.1
    time_s = 0.0
    max_steps = 10_000
    for _ in range(max_steps):
        state = patrol.advance(dt)
        evaluator.update(RouteSample(time_s, state["x"], state["y"]))
        time_s += dt
        if state["complete"]:
            break
    else:
        raise RuntimeError("preview did not complete before the step limit")
    result = evaluator.evaluate("kinematic_preview")
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if not result.route_complete:
        raise SystemExit("route evaluation failed")


if __name__ == "__main__":
    main()
