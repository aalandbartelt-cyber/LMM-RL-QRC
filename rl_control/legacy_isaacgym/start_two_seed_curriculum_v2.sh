#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "Usage: $0 LOAD_RUN|auto CHECKPOINT ITERATIONS NUM_ENVS" >&2
    exit 2
fi

LOAD_RUN="$1"
CHECKPOINT="$2"
ITERATIONS="$3"
NUM_ENVS="$4"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="${QRC_LEGACY_PROJECT:-${HOME}/go2_work/go2_rl_gym-master}"
ENV_PREFIX="${QRC_LEGACY_ENV:-${HOME}/conda/envs/go2rl}"
OUTPUT_ROOT="${QRC_LEGACY_OUTPUT:-${HOME}/go2_work/legacy_curriculum_v2}"
LOG_ROOT="${PROJECT_ROOT}/logs/go2_baseline"

if [[ "${LOAD_RUN}" == "auto" ]]; then
    SOURCE_MODEL="$(
        find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f \
            -path '*stage3_control*' -name "model_${CHECKPOINT}.pt" \
            -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
    )"
    if [[ -z "${SOURCE_MODEL}" ]]; then
        echo "No stage3_control model_${CHECKPOINT}.pt was found under ${LOG_ROOT}." >&2
        exit 2
    fi
    LOAD_RUN="$(basename "$(dirname "${SOURCE_MODEL}")")"
fi

SOURCE_MODEL="${LOG_ROOT}/${LOAD_RUN}/model_${CHECKPOINT}.pt"
if [[ ! -x "${ENV_PREFIX}/bin/python" || ! -f "${SOURCE_MODEL}" ]]; then
    echo "Legacy environment or source checkpoint is missing." >&2
    echo "ENV=${ENV_PREFIX}" >&2
    echo "MODEL=${SOURCE_MODEL}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
PID_FILES=()

PREFLIGHT_PIDS=()
PREFLIGHT_LOGS=()
for GPU in 0 1; do
    PREFLIGHT_LOG="${OUTPUT_ROOT}/preflight_${STAMP}_gpu${GPU}.log"
    (
        CUDA_VISIBLE_DEVICES="${GPU}" "${ENV_PREFIX}/bin/python" -u - <<'PY'
from isaacgym import gymapi, gymtorch  # noqa: F401
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
x = torch.randn((1024, 1024), device="cuda:0")
y = x @ x
torch.cuda.synchronize()
if not bool(torch.isfinite(y).all()):
    raise RuntimeError("CUDA produced non-finite output")
print("CUDA DEVICE PREFLIGHT PASS", torch.cuda.get_device_name(0))
PY
    ) >"${PREFLIGHT_LOG}" 2>&1 &
    PREFLIGHT_PIDS+=("$!")
    PREFLIGHT_LOGS+=("${PREFLIGHT_LOG}")
done

PREFLIGHT_FAILED=0
for INDEX in 0 1; do
    if wait "${PREFLIGHT_PIDS[INDEX]}"; then
        tail -n 3 "${PREFLIGHT_LOGS[INDEX]}"
    else
        echo "CUDA preflight failed for physical GPU ${INDEX}." >&2
        tail -n 80 "${PREFLIGHT_LOGS[INDEX]}" >&2
        PREFLIGHT_FAILED=1
    fi
done
if [[ "${PREFLIGHT_FAILED}" -ne 0 ]]; then
    echo "Training was not started because the two-GPU CUDA gate failed." >&2
    exit 3
fi

for GPU in 0 1; do
    SEED=$((GPU + 1))
    RUN_NAME="curriculum_v2_seed${SEED}_${STAMP}"
    RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
    mkdir -p "${RUN_DIR}"
    LOG="${RUN_DIR}/train.log"
    PID_FILE="${RUN_DIR}/train.pid"
    COMMAND=(
        "${ENV_PREFIX}/bin/python" -u "${SCRIPT_DIR}/train_go2_curriculum_v2.py"
        --task=go2 --headless --num_envs="${NUM_ENVS}" --seed="${SEED}"
        --sim_device=cuda:0 --rl_device=cuda:0
        --experiment_name=go2_baseline --resume
        --load_run="${LOAD_RUN}" --checkpoint="${CHECKPOINT}"
        --max_iterations="${ITERATIONS}" --run_name="${RUN_NAME}"
    )
    printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU}" "QRC_EXPECT_SOURCE_ITERATION=${CHECKPOINT}" \
        "${COMMAND[@]}" >"${RUN_DIR}/command.sh"
    printf '\n' >>"${RUN_DIR}/command.sh"
    {
        echo "profile=legacy_curriculum_v2"
        echo "physical_gpu=${GPU}"
        echo "logical_gpu=cuda:0"
        echo "seed=${SEED}"
        echo "source_run=${LOAD_RUN}"
        echo "source_checkpoint=${CHECKPOINT}"
        echo "iterations=${ITERATIONS}"
        echo "num_envs=${NUM_ENVS}"
        echo "started_utc=${STAMP}"
    } >"${RUN_DIR}/run.env"

    (
        cd "${PROJECT_ROOT}"
        nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
            QRC_EXPECT_SOURCE_ITERATION="${CHECKPOINT}" \
            "${COMMAND[@]}" >"${LOG}" 2>&1 </dev/null &
        echo $! >"${PID_FILE}"
    )
    PID_FILES+=("${PID_FILE}")
    echo "STARTED seed=${SEED} physical_gpu=${GPU} pid=$(cat "${PID_FILE}")"
    echo "LOG=${LOG}"
    sleep 8
done

sleep 20
for PID_FILE in "${PID_FILES[@]}"; do
    PID="$(cat "${PID_FILE}")"
    LOG="$(dirname "${PID_FILE}")/train.log"
    if kill -0 "${PID}" 2>/dev/null; then
        echo "RUNNING pid=${PID} log=${LOG}"
    elif grep -q "GO2 CURRICULUM V2 TRAINING PASS" "${LOG}" 2>/dev/null; then
        echo "COMPLETED pid=${PID} log=${LOG}"
    else
        echo "EARLY FAILURE pid=${PID} log=${LOG}" >&2
        tail -n 120 "${LOG}" >&2
        exit 2
    fi
done

echo "SOURCE_RUN=${LOAD_RUN}"
echo "TRAINING PROCESS CHECK PASS"
echo "Monitor with: bash ${SCRIPT_DIR}/show_progress.sh 10"
