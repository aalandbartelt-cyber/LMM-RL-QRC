"""Bind a policy file to its audited interface metadata.

Training checkpoints are easy to rename or copy incorrectly.  This manifest
format records the model file's SHA-256, format and observation/action contract
so an evaluation/deployment run can fail before calling an unintended policy.
It intentionally does not claim that a matching file is a good walking policy.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .interface_audit import InterfaceAuditResult, audit_policy_interface


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORMATS = {"torchscript", "onnx"}


@dataclass(frozen=True)
class PolicyArtifactManifest:
    model_path: str
    model_format: str
    sha256: str
    policy_interface: Mapping[str, object]


@dataclass(frozen=True)
class ArtifactVerification:
    verified: bool
    model_path: str
    interface: InterfaceAuditResult
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "model_path": self.model_path,
            "interface": self.interface.to_dict(),
            "reasons": list(self.reasons),
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_manifest(path: str | Path) -> PolicyArtifactManifest:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("policy manifest must be a JSON object")
    required = {"model_path", "model_format", "sha256", "policy_interface"}
    unexpected = set(payload) - required
    missing = required - set(payload)
    if unexpected or missing:
        raise ValueError(f"manifest fields unexpected={sorted(unexpected)} missing={sorted(missing)}")
    model_path = str(payload["model_path"])
    model_format = str(payload["model_format"]).lower()
    digest = str(payload["sha256"]).lower()
    interface = payload["policy_interface"]
    if not model_path or Path(model_path).is_absolute() or ".." in Path(model_path).parts:
        raise ValueError("model_path must be a non-empty relative path below the manifest")
    if model_format not in _FORMATS:
        raise ValueError(f"model_format must be one of {sorted(_FORMATS)}")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("sha256 must be a lowercase 64-character hexadecimal digest")
    if not isinstance(interface, Mapping):
        raise ValueError("policy_interface must be an object")
    return PolicyArtifactManifest(model_path, model_format, digest, interface)


def verify_policy_manifest(path: str | Path) -> ArtifactVerification:
    """Verify hash and profile; returns a structured failure rather than running a model."""
    manifest_path = Path(path).resolve()
    manifest = load_policy_manifest(manifest_path)
    interface = audit_policy_interface(manifest.policy_interface)
    model = (manifest_path.parent / manifest.model_path).resolve()
    reasons: list[str] = list(interface.reasons)
    if not model.is_file():
        reasons.append("model_file_missing")
    else:
        actual_hash = sha256_file(model)
        if actual_hash != manifest.sha256:
            reasons.append("model_sha256_mismatch")
        expected_suffixes = {"torchscript": {".pt", ".jit", ".ts"}, "onnx": {".onnx"}}[manifest.model_format]
        if model.suffix.lower() not in expected_suffixes:
            reasons.append(f"model_extension_does_not_match_{manifest.model_format}")
    return ArtifactVerification(not reasons, str(model), interface, tuple(reasons))
