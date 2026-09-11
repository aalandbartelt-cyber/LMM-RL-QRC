#!/usr/bin/env python3
"""Validate and summarize the fixed Go2 formal evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


POLICIES = (
    "baseline5001",
    "curriculum_seed1_5601",
    "curriculum_seed2_5601",
)
EVALUATION_SEEDS = (20260911, 20260912, 20260913)
TERRAINS = (
    "flat",
    "slope",
    "rough_slope",
    "stairs_up",
    "stairs_down",
    "obstacles",
    "wave",
)
METRICS = (
    ("success_rate", "success_rate"),
    ("collision_sample_rate", "collision_sample_rate"),
    ("tracking_rmse_mps", "tracking_rmse_mps"),
    ("mean_mechanical_power_w", "mechanical_power_w"),
    ("mean_tilt_rad", "tilt_rad"),
)


def collect_results(root: Path) -> list[dict]:
    """Load the complete matrix and reject missing or mislabeled results."""
    rows = []
    missing = []
    for policy in POLICIES:
        for seed in EVALUATION_SEEDS:
            for terrain in TERRAINS:
                path = root / policy / f"eval_seed_{seed}" / f"{terrain}.json"
                if not path.is_file():
                    missing.append(path.relative_to(root).as_posix())
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("terrain") != terrain or payload.get("seed") != seed:
                    raise ValueError(f"Result metadata mismatch: {path}")
                for metric in (*(source for source, _ in METRICS), "fall_count"):
                    if metric not in payload:
                        raise ValueError(f"Missing metric {metric}: {path}")
                payload["policy"] = policy
                rows.append(payload)
    if len(rows) != 63:
        raise ValueError(
            f"Expected 63 terrain results, found {len(rows)}; missing={missing}"
        )
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Aggregate terrains within each seed, then seeds within each policy."""
    summaries = []
    for policy in POLICIES:
        seed_rows = []
        for seed in EVALUATION_SEEDS:
            selected = [
                row
                for row in rows
                if row["policy"] == policy and row["seed"] == seed
            ]
            if len(selected) != 7:
                raise ValueError(f"Expected seven terrains for {policy} seed {seed}")
            stats = {
                source: mean(row[source] for row in selected)
                for source, _ in METRICS
            }
            stats["fall_count"] = sum(row["fall_count"] for row in selected)
            seed_rows.append(stats)

        result = {
            "policy": policy,
            "evaluation_seed_count": len(seed_rows),
            "terrain_evaluation_count": len(seed_rows) * len(TERRAINS),
            "total_falls": sum(row["fall_count"] for row in seed_rows),
            "mean_falls_per_seed": mean(row["fall_count"] for row in seed_rows),
            "std_falls_per_seed": pstdev(row["fall_count"] for row in seed_rows),
        }
        for source, output in METRICS:
            result[f"mean_{output}"] = mean(row[source] for row in seed_rows)
            result[f"std_{output}"] = pstdev(row[source] for row in seed_rows)
        summaries.append(result)

    summaries.sort(
        key=lambda row: (
            -row["mean_success_rate"],
            row["total_falls"],
            row["mean_collision_sample_rate"],
            row["mean_tracking_rmse_mps"],
            row["mean_mechanical_power_w"],
            row["mean_tilt_rad"],
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def write_outputs(root: Path, summary: list[dict]) -> None:
    """Write human- and machine-readable policy summaries."""
    (root / "formal_policy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    with (root / "formal_policy_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def write_sha256_manifest(root: Path) -> Path:
    """Hash every result artifact except the manifest itself."""
    manifest = root / "SHA256SUMS"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def aggregate(root: Path) -> list[dict]:
    """Run the complete acceptance gate and emit final artifacts."""
    rows = collect_results(root)
    summary = summarize(rows)
    write_outputs(root, summary)
    (root / "FORMAL_MATRIX_PASS").write_text(
        f"terrain_results={len(rows)}\ntop_policy={summary[0]['policy']}\n",
        encoding="utf-8",
    )
    write_sha256_manifest(root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    summary = aggregate(args.output_root.expanduser().resolve())
    print("===== GO2 FORMAL POLICY SUMMARY =====")
    print(
        f"{'rank':>4} {'policy':<29} {'success':>9} "
        f"{'falls':>7} {'collision':>10} {'rmse':>9}"
    )
    for row in summary:
        print(
            f"{row['rank']:>4d} {row['policy']:<29} "
            f"{row['mean_success_rate']:>8.2%} {row['total_falls']:>7d} "
            f"{row['mean_collision_sample_rate']:>9.2%} "
            f"{row['mean_tracking_rmse_mps']:>9.4f}"
        )
    print(f"OUTPUT_ROOT: {args.output_root.expanduser().resolve()}")
    print("GO2 FORMAL MATRIX AGGREGATION PASS")


if __name__ == "__main__":
    main()
