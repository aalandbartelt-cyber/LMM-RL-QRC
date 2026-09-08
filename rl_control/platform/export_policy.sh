#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 CHECKPOINT [TASK] [PHYSICAL_GPU]" >&2
    exit 2
fi

CHECKPOINT="$(realpath "$1")"
TASK="${2:-Unitree-Go2-Campus-Velocity}"
GPU="${3:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
ENV_PREFIX="${QRC_ENV_PREFIX:-${HOME}/conda/envs/go2campus}"
UNITREE_DIR="${QRC_UNITREE_DIR:-${HOME}/go2_work/upstream/unitree_rl_lab}"
PREFLIGHT_REPORT="${HOME}/go2_work/self5000_runtime_preflight.json"
PLAY_SCRIPT="${UNITREE_DIR}/scripts/rsl_rl/play.py"
EXPORT_DIR="$(dirname "${CHECKPOINT}")/exported"
JIT_POLICY="${EXPORT_DIR}/policy.pt"
ONNX_POLICY="${EXPORT_DIR}/policy.onnx"
LOG="$(dirname "${CHECKPOINT}")/export_policy.log"
TIMEOUT_SECONDS="${QRC_EXPORT_TIMEOUT_SECONDS:-300}"

if [[ ! -f "${CHECKPOINT}" || ! -f "${PLAY_SCRIPT}" ]]; then
    echo "Checkpoint or pinned Unitree play script is missing." >&2
    exit 2
fi
"${ENV_PREFIX}/bin/python" -c \
    'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["ready"] else 2)' \
    "${PREFLIGHT_REPORT}"

MARKER="$(mktemp)"
trap 'rm -f "${MARKER}"' EXIT
BOOTSTRAP='import os,sys,runpy; p=os.environ["QRC_PLAY_SCRIPT"]; sys.path.insert(0, os.path.dirname(p)); import campus_rl; runpy.run_path(p, run_name="__main__")'

(
    cd "${UNITREE_DIR}"
    env CUDA_VISIBLE_DEVICES="${GPU}" QRC_PLAY_SCRIPT="${PLAY_SCRIPT}" \
        "${ENV_PREFIX}/bin/python" -c "${BOOTSTRAP}" \
        --task "${TASK}" --checkpoint "${CHECKPOINT}" \
        --num_envs 1 --device cuda:0 --headless
) >"${LOG}" 2>&1 &
PID=$!

while (( SECONDS < TIMEOUT_SECONDS )); do
    if [[ -f "${JIT_POLICY}" && -f "${ONNX_POLICY}" \
        && "${JIT_POLICY}" -nt "${MARKER}" && "${ONNX_POLICY}" -nt "${MARKER}" ]]; then
        kill "${PID}" 2>/dev/null || true
        wait "${PID}" 2>/dev/null || true
        sync
        sha256sum "${JIT_POLICY}" "${ONNX_POLICY}" | tee "${EXPORT_DIR}/SHA256SUMS"
        echo "POLICY EXPORT PASS: ${EXPORT_DIR}"
        exit 0
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        wait "${PID}" || true
        echo "Policy export process ended before both files were produced." >&2
        tail -n 120 "${LOG}" >&2
        exit 2
    fi
    sleep 2
done

kill "${PID}" 2>/dev/null || true
wait "${PID}" 2>/dev/null || true
echo "Policy export timed out after ${TIMEOUT_SECONDS}s." >&2
tail -n 120 "${LOG}" >&2
exit 2
