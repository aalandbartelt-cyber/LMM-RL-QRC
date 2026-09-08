#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${QRC_OUTPUT_ROOT:-${HOME}/go2_work/campus_training}"
INTERVAL="${1:-10}"

while true; do
    clear || true
    date -Is
    echo "===== TRAINING JOBS ====="
    while IFS= read -r pid_file; do
        run_dir="$(dirname "${pid_file}")"
        pid="$(cat "${pid_file}")"
        if kill -0 "${pid}" 2>/dev/null; then
            state="RUNNING"
        else
            state="FINISHED"
        fi
        echo "${state} pid=${pid} run=$(basename "${run_dir}")"
        grep -aE "Learning iteration|Mean reward|Total timesteps|ETA:|Traceback|Error|PASS" \
            "${run_dir}/train.log" 2>/dev/null | tail -n 12 || true
        echo
    done < <(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name train.pid -type f | sort)
    echo "===== GPU ====="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader
    sleep "${INTERVAL}"
done
