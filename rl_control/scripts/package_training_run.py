#!/usr/bin/env python3
"""Create and verify a portable training-result archive."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(root: Path, archive_root: str) -> list[dict[str, Any]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in result archive: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append(
                {
                    "source": path,
                    "archive_path": f"{archive_root}/{relative}",
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return files


def checksum_path_for(archive_path: Path) -> Path:
    name = archive_path.name
    stem = name[:-7] if name.endswith(".tar.gz") else archive_path.stem
    return archive_path.with_name(f"{stem}.sha256")


def manifest_path_for(archive_path: Path) -> Path:
    name = archive_path.name
    stem = name[:-7] if name.endswith(".tar.gz") else archive_path.stem
    return archive_path.with_name(f"{stem}.manifest.json")


def create_training_archive(
    run_dir: Path,
    checkpoint_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    for source_root in (run_dir, checkpoint_dir):
        try:
            output_path.relative_to(source_root)
        except ValueError:
            continue
        raise ValueError("Archive output must be outside the source directories")

    entries = collect_files(run_dir, "training_run") + collect_files(
        checkpoint_dir, "checkpoint"
    )
    if not entries:
        raise ValueError("No result files were found")

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "files": [
            {
                "archive_path": entry["archive_path"],
                "size_bytes": entry["size_bytes"],
                "sha256": entry["sha256"],
            }
            for entry in entries
        ],
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as archive:
        for entry in entries:
            archive.add(entry["source"], arcname=entry["archive_path"], recursive=False)
        info = tarfile.TarInfo("MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(manifest_bytes))

    archive_sha256 = sha256_file(output_path)
    checksum_path = checksum_path_for(output_path)
    checksum_path.write_text(f"{archive_sha256}  {output_path.name}\n", encoding="utf-8")
    manifest_path = manifest_path_for(output_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with tarfile.open(output_path, "r:gz") as archive:
        names = set(archive.getnames())
        expected_names = {entry["archive_path"] for entry in entries} | {"MANIFEST.json"}
        verified = names == expected_names and archive.extractfile("MANIFEST.json") is not None
    if not verified:
        raise RuntimeError("Archive verification failed")

    return {
        "archive": str(output_path),
        "archive_size_bytes": output_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "checksum": str(checksum_path),
        "manifest": str(manifest_path),
        "file_count": len(entries),
        "verified": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = create_training_archive(args.run_dir, args.checkpoint_dir, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
