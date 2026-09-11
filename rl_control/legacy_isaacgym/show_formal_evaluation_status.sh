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

echo "===== GO2 FORMAL EVALUATION STATUS ====="
echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "output_root=${OUTPUT_ROOT}"

for name in gpu0 gpu1 coordinator; do
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

complete_count="$(find "${OUTPUT_ROOT}" -type f -name COMPLETE | wc -l)"
failed_count="$(find "${OUTPUT_ROOT}" -type f -name FAILED | wc -l)"
echo "jobs_complete=${complete_count}/9"
echo "jobs_failed=${failed_count}"

for marker in \
    workers/WORKER_0_PASS \
    workers/WORKER_1_PASS \
    workers/WORKER_0_FAILED \
    workers/WORKER_1_FAILED \
    MATRIX_FAILED \
    FORMAL_MATRIX_PASS; do
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
