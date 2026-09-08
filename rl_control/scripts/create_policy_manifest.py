"""Create a hash-bound policy manifest next to an exported model.

The script validates the supplied interface metadata before writing.  It will
never overwrite an existing manifest; pass a copied/edited metadata JSON that
matches the policy export profile.

Usage:
    python -m scripts.create_policy_manifest policy.pt policy_metadata.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from go2_foundation.artifact_manifest import sha256_file
from go2_foundation.interface_audit import audit_policy_interface


def infer_format(model: Path) -> str:
    suffix = model.suffix.lower()
    if suffix in {".pt", ".jit", ".ts"}:
        return "torchscript"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError("cannot infer model format; use a .pt/.jit/.ts or .onnx file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("metadata_json", type=Path)
    parser.add_argument("--output", type=Path, help="default: policy_manifest.json beside the model")
    args = parser.parse_args()

    model = args.model.resolve()
    if not model.is_file():
        parser.error(f"model file does not exist: {model}")
    output = (args.output.resolve() if args.output else model.parent / "policy_manifest.json")
    if output.parent != model.parent:
        parser.error("output must be beside the model so the manifest has a safe relative model_path")
    if output.exists():
        parser.error(f"refusing to overwrite existing manifest: {output}")
    metadata = json.loads(args.metadata_json.read_text(encoding="utf-8"))
    audit = audit_policy_interface(metadata)
    if not audit.compatible:
        print(json.dumps(audit.to_dict(), ensure_ascii=False, indent=2))
        return 2
    payload = {
        "model_path": model.name,
        "model_format": infer_format(model),
        "sha256": sha256_file(model),
        "policy_interface": audit.to_dict()["metadata"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"created": str(output), "sha256": payload["sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
