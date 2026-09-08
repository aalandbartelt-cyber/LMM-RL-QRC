"""Verify a policy artifact file matches its audited JSON manifest.

Usage:
    python -m scripts.verify_policy_manifest path/to/policy_manifest.json
"""

import argparse
import json

from go2_foundation.artifact_manifest import verify_policy_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_json")
    args = parser.parse_args()
    result = verify_policy_manifest(args.manifest_json)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0 if result.verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
