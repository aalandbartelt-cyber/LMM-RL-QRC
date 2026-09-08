#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${QRC_LEGACY_OUTPUT:-${HOME}/go2_work/legacy_curriculum_v2}"
INTERVAL="${1:-10}"

while true; do
    clear || true
    date -Is
    echo "===== LEGACY CURRICULUM V2 ====="
    while IFS= read -r PID_FILE; do
        RUN_DIR="$(dirname "${PID_FILE}")"
        PID="$(cat "${PID_FILE}")"
        if kill -0 "${PID}" 2>/dev/null; then
            STATE="RUNNING"
        else
            STATE="FINISHED"
        fi
        echo "${STATE} pid=${PID} run=$(basename "${RUN_DIR}")"
        grep -aE \
            "QRC_PROFILE|LOADED_ITERATION|Learning iteration|Mean reward|terrain_level_stairs_up|terrain_level_obstacles|FINAL_ITERATION|Traceback|TRAINING PASS" \
            "${RUN_DIR}/train.log" 2>/dev/null | tail -n 18 || true
        echo
    done < <(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name train.pid -type f | sort)
    echo "===== GPU ====="
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader
    sleep "${INTERVAL}"
done
