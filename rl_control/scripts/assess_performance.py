"""Assess an RL evaluation JSON artifact against the project performance gate.

Usage:
    python -m scripts.assess_performance path/to/evaluation.json
"""

import argparse

from go2_foundation import assess_file, decision_to_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluation_json", help="JSON file containing evidence_mode and results_by_seed")
    args = parser.parse_args()
    decision = assess_file(args.evaluation_json)
    print(decision_to_json(decision))
    return 0 if decision.promoted else 2


if __name__ == "__main__":
    raise SystemExit(main())
