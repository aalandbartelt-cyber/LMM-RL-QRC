#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
    echo "Usage: $0 baseline|campus GPU SEED ITERATIONS NUM_ENVS" >&2
    exit 2
fi

STAGE="$1"
GPU="$2"
SEED="$3"
ITERATIONS="$4"
NUM_ENVS="$5"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
ENV_PREFIX="${QRC_ENV_PREFIX:-${HOME}/conda/envs/go2campus}"
UNITREE_DIR="${QRC_UNITREE_DIR:-${HOME}/go2_work/upstream/unitree_rl_lab}"
OUTPUT_ROOT="${QRC_OUTPUT_ROOT:-${HOME}/go2_work/campus_training}"
PREFLIGHT_REPORT="${HOME}/go2_work/self5000_runtime_preflight.json"

if [[ "${STAGE}" != "baseline" && "${STAGE}" != "campus" ]]; then
    echo "Stage must be baseline or campus." >&2
    exit 2
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    echo "Missing environment: ${ENV_PREFIX}" >&2
    exit 2
fi
if [[ ! -f "${PREFLIGHT_REPORT}" ]]; then
    echo "Runtime preflight report is missing: ${PREFLIGHT_REPORT}" >&2
    exit 2
fi
"${ENV_PREFIX}/bin/python" -c \
    'import json,sys; sys.exit(0 if json.load(open(sys.argv[1], encoding="utf-8"))["ready"] else 2)' \
    "${PREFLIGHT_REPORT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${STAGE}_seed${SEED}_${STAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
LOG="${RUN_DIR}/train.log"
PID_FILE="${RUN_DIR}/train.pid"

COMMAND=(
    "${ENV_PREFIX}/bin/python"
    "${REPO_ROOT}/rl_control/scripts/launch_unitree_training.py"
    --unitree-rl-lab "${UNITREE_DIR}"
    --stage "${STAGE}"
    --seed "${SEED}"
    --max-iterations "${ITERATIONS}"
    --num-envs "${NUM_ENVS}"
    --device cuda:0
    --run-name "${RUN_NAME}"
)

if [[ "${STAGE}" == "campus" ]]; then
    if [[ -z "${QRC_LOAD_RUN:-}" || -z "${QRC_CHECKPOINT:-}" ]]; then
        echo "Campus fine-tuning requires QRC_LOAD_RUN and QRC_CHECKPOINT." >&2
        exit 2
    fi
    COMMAND+=(--resume --load-run "${QRC_LOAD_RUN}" --checkpoint "${QRC_CHECKPOINT}")
fi
COMMAND+=(--execute -- --logger tensorboard --save_interval 100)

printf '%q ' env "CUDA_VISIBLE_DEVICES=${GPU}" "${COMMAND[@]}" >"${RUN_DIR}/command.sh"
printf '\n' >>"${RUN_DIR}/command.sh"
{
    echo "stage=${STAGE}"
    echo "physical_gpu=${GPU}"
    echo "logical_device=cuda:0"
    echo "seed=${SEED}"
    echo "iterations=${ITERATIONS}"
    echo "num_envs=${NUM_ENVS}"
    echo "started_utc=${STAMP}"
    git -C "${REPO_ROOT}" rev-parse HEAD | sed 's/^/project_commit=/'
    git -C "${UNITREE_DIR}" rev-parse HEAD | sed 's/^/unitree_commit=/'
} >"${RUN_DIR}/run.env"

nohup env CUDA_VISIBLE_DEVICES="${GPU}" "${COMMAND[@]}" >"${LOG}" 2>&1 </dev/null &
PID=$!
printf '%s\n' "${PID}" >"${PID_FILE}"
echo "TRAINING STARTED: stage=${STAGE} seed=${SEED} gpu=${GPU} pid=${PID}"
echo "RUN_DIR=${RUN_DIR}"
echo "LOG=${LOG}"
