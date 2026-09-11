#!/usr/bin/env bash
set -euo pipefail

ROOT="${QRC_WORK_ROOT:-${HOME}/go2_work}"
LATEST_POINTER="${ROOT}/latest_formal_eval_dir.txt"
OUTPUT_ROOT="${1:-}"

if [[ -z "${OUTPUT_ROOT}" ]]; then
    [[ -f "${LATEST_POINTER}" ]] || {
        echo "No formal evaluation pointer: ${LATEST_POINTER}" >&2
        exit 2
    }
    OUTPUT_ROOT="$(cat "${LATEST_POINTER}")"
fi
[[ -d "${OUTPUT_ROOT}" ]] || {
    echo "Formal evaluation output does not exist: ${OUTPUT_ROOT}" >&2
    exit 2
}

PHYSICAL_GPUS=(0 1)
manifest_gpu_list=""
if [[ -f "${OUTPUT_ROOT}/manifest.txt" ]]; then
    manifest_gpu_list="$(
        sed -n 's/^physical_gpus=//p' "${OUTPUT_ROOT}/manifest.txt" |
            head -n 1 | tr -d '\r'
    )"
fi
if [[ -n "${manifest_gpu_list//[[:space:]]/}" ]]; then
    read -r -a PHYSICAL_GPUS <<<"${manifest_gpu_list}"
fi

echo "===== GO2 FORMAL EVALUATION STATUS ====="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "output_root=${OUTPUT_ROOT}"

for gpu in "${PHYSICAL_GPUS[@]}"; do
    name="gpu${gpu}"
    pid_file="${OUTPUT_ROOT}/workers/${name}.pid"
    if [[ ! -f "${pid_file}" ]]; then
        echo "${name}_pid=missing"
        continue
    fi
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
        echo "${name}_pid=${pid} status=running"
    else
        echo "${name}_pid=${pid} status=exited"
    fi
done

name="coordinator"
pid_file="${OUTPUT_ROOT}/workers/${name}.pid"
if [[ ! -f "${pid_file}" ]]; then
    echo "${name}_pid=missing"
else
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
        echo "${name}_pid=${pid} status=running"
    else
        echo "${name}_pid=${pid} status=exited"
    fi
fi

complete_count="$(find "${OUTPUT_ROOT}" -type f -name COMPLETE | wc -l)"
failed_count="$(find "${OUTPUT_ROOT}" -type f -name FAILED | wc -l)"
echo "jobs_complete=${complete_count}/9"
echo "jobs_failed=${failed_count}"

for gpu in "${PHYSICAL_GPUS[@]}"; do
    for marker in \
        "workers/WORKER_${gpu}_PASS" \
        "workers/WORKER_${gpu}_FAILED"; do
        if [[ -f "${OUTPUT_ROOT}/${marker}" ]]; then
            echo "marker=${marker} present"
        else
            echo "marker=${marker} absent"
        fi
    done
done

for marker in MATRIX_FAILED FORMAL_MATRIX_PASS; do
    if [[ -f "${OUTPUT_ROOT}/${marker}" ]]; then
        echo "marker=${marker} present"
    else
        echo "marker=${marker} absent"
    fi
done

echo "===== LATEST MEANINGFUL LOG LINES ====="
grep -H -E \
    'BENCHMARK TERRAIN|GO2 TERRAIN .* PASS|GO2 MULTI-TERRAIN|Traceback|RuntimeError|FAILED' \
    "${OUTPUT_ROOT}"/workers/*.log \
    "${OUTPUT_ROOT}"/*/eval_seed_*/eval.log 2>/dev/null | tail -n 60 || true

if [[ -f "${OUTPUT_ROOT}/formal_policy_summary.csv" ]]; then
    echo "===== FORMAL POLICY SUMMARY ====="
    cat "${OUTPUT_ROOT}/formal_policy_summary.csv"
fi
