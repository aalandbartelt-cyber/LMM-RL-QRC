#!/bin/bash
set -uo pipefail

WORK=/home/jovyan/go2_work
PROJECT=$WORK/go2_rl_gym-master
OUT=$WORK/video_evidence_20260730
RENDERER=$WORK/render_go2_batch.py
PYTHON=/home/jovyan/conda/envs/go2render/bin/python
POLICY=$WORK/handoff_20260728/stage3_targeted_model5001/export/policy.onnx
TERRAINS=$PROJECT/resources/robots/go2

mkdir -p "$OUT"

{
    echo "Batch start: $(date --iso-8601=seconds)"
    echo "Host: $(hostname)"
    echo "Policy: $POLICY"
    echo "Policy SHA256: $(sha256sum "$POLICY" | awk '{print $1}')"
    nvidia-smi --query-gpu=name,driver_version \
        --format=csv,noheader
} > "$OUT/run_metadata.txt"

run_case() {
    local name="$1"
    local terrain="$2"
    local duration="$3"
    local vx="$4"
    local vy="$5"
    local yaw="$6"

    local video="$OUT/${name}.mp4"
    local log="$OUT/${name}.log"

    echo
    echo "===== START: $name ====="
    echo "Terrain=$terrain Duration=$duration Command=[$vx,$vy,$yaw]"
    echo "Start=$(date --iso-8601=seconds)"

    env \
        QRC_LABEL="$name" \
        QRC_POLICY="$POLICY" \
        QRC_XML="$TERRAINS/$terrain" \
        QRC_OUTPUT="$video" \
        QRC_DURATION="$duration" \
        QRC_FPS=30 \
        QRC_VX="$vx" \
        QRC_VY="$vy" \
        QRC_YAW="$yaw" \
        LD_LIBRARY_PATH=/home/jovyan/conda/envs/go2render/lib \
        MUJOCO_GL=osmesa \
        PYOPENGL_PLATFORM=osmesa \
        LIBGL_ALWAYS_SOFTWARE=1 \
        "$PYTHON" -u "$RENDERER" \
        2>&1 | tee "$log"

    local code=${PIPESTATUS[0]}

    echo "Exit code=$code"
    echo "End=$(date --iso-8601=seconds)"
    echo "===== END: $name ====="

    return "$code"
}

run_case 01_targeted_flat_forward    flat.xml         10 0.50 0.00 0.00
run_case 02_targeted_flat_turn_left  flat.xml         10 0.25 0.00 0.50
run_case 03_targeted_stairs_forward  stairs.xml       18 0.32 0.00 0.00
run_case 04_targeted_cross_stairs    cross_stairs.xml 15 0.22 0.00 0.00
run_case 05_targeted_cross_slope     cross_slope.xml  15 0.30 0.00 0.00
run_case 06_targeted_race_track      race_track.xml   18 0.35 0.00 0.00

"$PYTHON" - <<'PY'
from pathlib import Path
import csv
import hashlib
import json

out = Path("/home/jovyan/go2_work/video_evidence_20260730")
rows = []

for path in sorted(out.glob("[0-9][0-9]_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    video = Path(data["video"])

    digest = ""
    if video.exists():
        digest = hashlib.sha256(video.read_bytes()).hexdigest()

    rows.append({
        "label": data.get("label"),
        "status": data.get("status"),
        "terrain": data.get("terrain"),
        "duration_seconds": data.get("duration_seconds"),
        "frames": data.get("frames"),
        "command_x": data.get("command_x"),
        "command_y": data.get("command_y"),
        "command_yaw": data.get("command_yaw"),
        "forward_distance": data.get("forward_distance"),
        "minimum_base_height": data.get("minimum_base_height"),
        "final_base_height": data.get("final_base_height"),
        "video_size_bytes": data.get("video_size_bytes"),
        "video_sha256": digest,
        "video": str(video),
    })

fields = list(rows[0].keys()) if rows else ["label", "status"]

with (out / "video_manifest.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"Manifest rows: {len(rows)}")
PY

cd "$OUT"
find . -maxdepth 1 -type f \
    ! -name SHA256SUMS \
    -print0 |
    sort -z |
    xargs -0 sha256sum > SHA256SUMS

echo "Batch end: $(date --iso-8601=seconds)" \
    >> "$OUT/run_metadata.txt"

echo "GO2 VIDEO MATRIX PASS"
