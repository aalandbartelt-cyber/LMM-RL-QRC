"""Audit exported-policy metadata before simulation or hardware inference.

Usage:
    python -m scripts.audit_policy_interface policy_metadata.json
"""

import argparse
import json
from pathlib import Path

from go2_foundation.interface_audit import audit_policy_interface


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_json")
    args = parser.parse_args()
    payload = json.loads(Path(args.metadata_json).read_text(encoding="utf-8"))
    result = audit_policy_interface(payload)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.compatible else 2


if __name__ == "__main__":
    raise SystemExit(main())
