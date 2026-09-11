#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
ROOT="${QRC_WORK_ROOT:-${HOME}/go2_work}"
REPO="${ROOT}/LMM-RL-QRC"
GYM="${ROOT}/go2_rl_gym-master"
LOG_ROOT="${GYM}/logs/go2_baseline"
ENV_PREFIX="${QRC_LEGACY_ENV:-${HOME}/conda/envs/go2rl}"
PYTHON="${ENV_PREFIX}/bin/python"
EVALUATOR="${REPO}/evaluation/scripts/evaluate_go2_terrain.py"
AGGREGATOR="${REPO}/evaluation/scripts/aggregate_formal_matrix.py"
LATEST_POINTER="${ROOT}/latest_formal_eval_dir.txt"
EVALUATION_SEEDS=(20260911 20260912 20260913)
TERRAINS="flat,slope,rough_slope,stairs_up,stairs_down,obstacles,wave"
NUM_ENVS="256"
DURATION_SECONDS="20"
COMMAND_VX="0.5"
COMMAND_VY="0.0"
COMMAND_YAW="0.0"
QRC_EVAL_GPUS="${QRC_EVAL_GPUS-0 1}"
PHYSICAL_GPUS=()


fail() {
    echo "ERROR: $*" >&2
    exit 2
}


parse_gpu_selection() {
    [[ -n "${QRC_EVAL_GPUS//[[:space:]]/}" ]] || \
        fail "QRC_EVAL_GPUS must select one or two physical GPU indices"
    read -r -a PHYSICAL_GPUS <<<"${QRC_EVAL_GPUS}"
    (( ${#PHYSICAL_GPUS[@]} >= 1 && ${#PHYSICAL_GPUS[@]} <= 2 )) || \
        fail "QRC_EVAL_GPUS must contain one or two physical GPU indices"

    local index
    local gpu
    local normalized
    local seen=" "
    for index in "${!PHYSICAL_GPUS[@]}"; do
        gpu="${PHYSICAL_GPUS[${index}]}"
        [[ "${gpu}" =~ ^[0-9]+$ ]] || fail "Invalid physical GPU index: ${gpu}"
        normalized="$((10#${gpu}))"
        [[ "${seen}" != *" ${normalized} "* ]] || \
            fail "Duplicate physical GPU index: ${gpu}"
        PHYSICAL_GPUS[${index}]="${normalized}"
        seen+="${normalized} "
    done
}


validate_gpu_selection_availability() {
    local gpu_count="$1"
    local gpu
    for gpu in "${PHYSICAL_GPUS[@]}"; do
        (( gpu < gpu_count )) || \
            fail "Physical GPU index ${gpu} is unavailable; nvidia-smi reports ${gpu_count} GPU(s)"
    done
}


read_manifest_gpu_list() {
    local output_root="$1"
    local value=""
    if [[ -f "${output_root}/manifest.txt" ]]; then
        value="$(
            sed -n 's/^physical_gpus=//p' "${output_root}/manifest.txt" |
                head -n 1 | tr -d '\r'
        )"
    fi
    [[ -n "${value//[[:space:]]/}" ]] || value="0 1"
    printf '%s\n' "${value}"
}


require_runtime() {
    [[ -x "${PYTHON}" ]] || fail "Python environment is missing: ${PYTHON}"
    [[ -f "${EVALUATOR}" ]] || fail "Evaluator is missing: ${EVALUATOR}"
    [[ -f "${AGGREGATOR}" ]] || fail "Aggregator is missing: ${AGGREGATOR}"
    [[ -d "${LOG_ROOT}" ]] || fail "Checkpoint directory is missing: ${LOG_ROOT}"
    command -v nvidia-smi >/dev/null || fail "nvidia-smi is unavailable"
    local gpu_count
    gpu_count="$(nvidia-smi -L | grep -c '^GPU ' || true)"
    validate_gpu_selection_availability "${gpu_count}"
}


newest_model() {
    local path_pattern="$1"
    local checkpoint_name="$2"
    find "${LOG_ROOT}" -mindepth 2 -maxdepth 2 -type f \
        -path "${path_pattern}" -name "${checkpoint_name}" \
        -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-
}


resolve_models() {
    BASE_MODEL="$(newest_model '*stage3_control*' 'model_5001.pt')"
    CURRICULUM1_MODEL="$(newest_model '*curriculum_v2_seed1_*' 'model_5601.pt')"
    CURRICULUM2_MODEL="$(newest_model '*curriculum_v2_seed2_*' 'model_5601.pt')"

    [[ -f "${BASE_MODEL}" ]] || fail "No stage3_control/model_5001.pt was found"
    [[ -f "${CURRICULUM1_MODEL}" ]] || fail "No curriculum seed-1 model_5601.pt was found"
    [[ -f "${CURRICULUM2_MODEL}" ]] || fail "No curriculum seed-2 model_5601.pt was found"

    BASE_RUN="$(basename "$(dirname "${BASE_MODEL}")")"
    CURRICULUM1_RUN="$(basename "$(dirname "${CURRICULUM1_MODEL}")")"
    CURRICULUM2_RUN="$(basename "$(dirname "${CURRICULUM2_MODEL}")")"
    BASE_HASH="$(sha256sum "${BASE_MODEL}" | awk '{print $1}')"
    CURRICULUM1_HASH="$(sha256sum "${CURRICULUM1_MODEL}" | awk '{print $1}')"
    CURRICULUM2_HASH="$(sha256sum "${CURRICULUM2_MODEL}" | awk '{print $1}')"
    [[ "${CURRICULUM1_HASH}" != "${CURRICULUM2_HASH}" ]] || \
        fail "The two curriculum checkpoints have identical SHA-256 digests"
}


gpu_preflight() {
    local gpu="$1"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u - <<'PY'
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
print("FORMAL EVALUATION GPU PREFLIGHT PASS", torch.cuda.get_device_name(0))
PY
}


write_manifest() {
    local output_root="$1"
    {
        echo "repository_commit=$(git -C "${REPO}" rev-parse HEAD)"
        echo "started_utc=$(date -u +%Y%m%dT%H%M%SZ)"
        echo "python=${PYTHON}"
        echo "physical_gpus=${PHYSICAL_GPUS[*]}"
        echo "evaluation_seeds=${EVALUATION_SEEDS[*]}"
        echo "terrains=${TERRAINS}"
        echo "num_envs=${NUM_ENVS}"
        echo "duration_seconds=${DURATION_SECONDS}"
        echo "command_vx=${COMMAND_VX}"
        echo "command_vy=${COMMAND_VY}"
        echo "command_yaw=${COMMAND_YAW}"
        echo "baseline_run=${BASE_RUN}"
        echo "baseline_model=${BASE_MODEL}"
        echo "baseline_bytes=$(stat -c %s "${BASE_MODEL}")"
        echo "baseline_sha256=${BASE_HASH}"
        echo "curriculum_seed1_run=${CURRICULUM1_RUN}"
        echo "curriculum_seed1_model=${CURRICULUM1_MODEL}"
        echo "curriculum_seed1_bytes=$(stat -c %s "${CURRICULUM1_MODEL}")"
        echo "curriculum_seed1_sha256=${CURRICULUM1_HASH}"
        echo "curriculum_seed2_run=${CURRICULUM2_RUN}"
        echo "curriculum_seed2_model=${CURRICULUM2_MODEL}"
        echo "curriculum_seed2_bytes=$(stat -c %s "${CURRICULUM2_MODEL}")"
        echo "curriculum_seed2_sha256=${CURRICULUM2_HASH}"
    } >"${output_root}/manifest.txt"
}


seed_smoke_gate() {
    local output_root="$1"
    local gpu="$2"
    local smoke_dir="${output_root}/seed_smoke"
    mkdir -p "${smoke_dir}"
    rm -f "${smoke_dir}/flat.json" "${smoke_dir}/eval.log" \
        "${smoke_dir}/SEED_SMOKE_PASS"

    (
        cd "${GYM}"
        env CUDA_VISIBLE_DEVICES="${gpu}" \
            QRC_TERRAIN=flat \
            QRC_DURATION_SECONDS=5 \
            QRC_COMMAND_VX="${COMMAND_VX}" \
            QRC_COMMAND_VY="${COMMAND_VY}" \
            QRC_COMMAND_YAW="${COMMAND_YAW}" \
            QRC_OUTPUT_DIR="${smoke_dir}" \
            "${PYTHON}" -u "${EVALUATOR}" \
                --task=go2 --headless --num_envs=32 --seed=20260911 \
                --sim_device=cuda:0 --rl_device=cuda:0 \
                --experiment_name=go2_baseline \
                --load_run="${BASE_RUN}" --checkpoint=5001
    ) >"${smoke_dir}/eval.log" 2>&1

    grep -q 'Setting seed: 20260911' "${smoke_dir}/eval.log"
    grep -q 'GO2 TERRAIN FLAT PASS' "${smoke_dir}/eval.log"
    "${PYTHON}" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d["seed"] == 20260911 else 2)' \
        "${smoke_dir}/flat.json"
    printf 'runtime_seed=20260911\n' >"${smoke_dir}/SEED_SMOKE_PASS"
    echo "FORMAL EVALUATION SEED SMOKE PASS"
}


write_queues() {
    local output_root="$1"
    rm -f "${output_root}"/workers/gpu*.queue
    local gpu
    for gpu in "${PHYSICAL_GPUS[@]}"; do
        : >"${output_root}/workers/gpu${gpu}.queue"
    done
    local index=0
    local seed
    local queue
    local queue_gpu
    local record
    for seed in "${EVALUATION_SEEDS[@]}"; do
        for record in \
            "baseline5001|${seed}|${BASE_RUN}|5001" \
            "curriculum_seed1_5601|${seed}|${CURRICULUM1_RUN}|5601" \
            "curriculum_seed2_5601|${seed}|${CURRICULUM2_RUN}|5601"; do
            queue_gpu="${PHYSICAL_GPUS[$((index % ${#PHYSICAL_GPUS[@]}))]}"
            queue="${output_root}/workers/gpu${queue_gpu}.queue"
            printf '%s\n' "${record}" >>"${queue}"
            index=$((index + 1))
        done
    done
}


job_complete() {
    local job_dir="$1"
    local seed="$2"
    [[ -f "${job_dir}/benchmark_summary.csv" ]] || return 1
    [[ -f "${job_dir}/benchmark_summary.json" ]] || return 1
    grep -q 'GO2 MULTI-TERRAIN BENCHMARK PASS' "${job_dir}/eval.log" 2>/dev/null || return 1
    local terrain
    local -a terrain_names
    IFS=',' read -ra terrain_names <<<"${TERRAINS}"
    for terrain in "${terrain_names[@]}"; do
        [[ -f "${job_dir}/${terrain}.json" ]] || return 1
    done
    "${PYTHON}" - "${job_dir}" "${seed}" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
expected_seed = int(sys.argv[2])
paths = sorted(directory.glob("*.json"))
terrain_paths = [path for path in paths if path.name != "benchmark_summary.json"]
if len(terrain_paths) != 7:
    raise SystemExit(2)
if any(json.loads(path.read_text(encoding="utf-8"))["seed"] != expected_seed for path in terrain_paths):
    raise SystemExit(2)
PY
}


run_job() {
    local GPU="$1"
    local gpu="${GPU}"
    local output_root="$2"
    local policy="$3"
    local seed="$4"
    local load_run="$5"
    local checkpoint="$6"
    local job_dir="${output_root}/${policy}/eval_seed_${seed}"
    mkdir -p "${job_dir}"

    if job_complete "${job_dir}" "${seed}"; then
        printf 'resumed_complete=1\n' >"${job_dir}/COMPLETE"
        echo "SKIP COMPLETE policy=${policy} seed=${seed}"
        return 0
    fi

    rm -f "${job_dir}/COMPLETE" "${job_dir}/FAILED" \
        "${job_dir}/benchmark_summary.csv" "${job_dir}/benchmark_summary.json"
    echo "START policy=${policy} seed=${seed} physical_gpu=${gpu}"
    if ! (
        cd "${GYM}"
        env CUDA_VISIBLE_DEVICES="${GPU}" \
            QRC_RUN_ALL=1 \
            QRC_DURATION_SECONDS="${DURATION_SECONDS}" \
            QRC_COMMAND_VX="${COMMAND_VX}" \
            QRC_COMMAND_VY="${COMMAND_VY}" \
            QRC_COMMAND_YAW="${COMMAND_YAW}" \
            QRC_TERRAINS="${TERRAINS}" \
            QRC_OUTPUT_DIR="${job_dir}" \
            "${PYTHON}" -u "${EVALUATOR}" \
                --task=go2 --headless --num_envs="${NUM_ENVS}" --seed="${seed}" \
                --sim_device=cuda:0 --rl_device=cuda:0 \
                --experiment_name=go2_baseline \
                --load_run="${load_run}" --checkpoint="${checkpoint}"
    ) >"${job_dir}/eval.log" 2>&1; then
        printf 'policy=%s\nseed=%s\nphysical_gpu=%s\n' \
            "${policy}" "${seed}" "${gpu}" >"${job_dir}/FAILED"
        return 1
    fi

    if ! job_complete "${job_dir}" "${seed}"; then
        printf 'policy=%s\nseed=%s\nphysical_gpu=%s\nreason=incomplete_artifacts\n' \
            "${policy}" "${seed}" "${gpu}" >"${job_dir}/FAILED"
        return 1
    fi
    printf 'policy=%s\nseed=%s\nphysical_gpu=%s\n' \
        "${policy}" "${seed}" "${gpu}" >"${job_dir}/COMPLETE"
    echo "PASS policy=${policy} seed=${seed} physical_gpu=${gpu}"
}


worker_mode() {
    local GPU="$1"
    local gpu="${GPU}"
    local output_root="$2"
    local queue="$3"
    local failed_marker="${output_root}/workers/WORKER_${GPU}_FAILED"
    local pass_marker="${output_root}/workers/WORKER_${GPU}_PASS"
    rm -f "${failed_marker}" "${pass_marker}"

    while IFS='|' read -r policy seed load_run checkpoint; do
        [[ -n "${policy}" ]] || continue
        if ! run_job "${gpu}" "${output_root}" "${policy}" "${seed}" \
            "${load_run}" "${checkpoint}"; then
            printf 'policy=%s\nseed=%s\n' "${policy}" "${seed}" >"${failed_marker}"
            echo "WORKER ${gpu} FAILED policy=${policy} seed=${seed}" >&2
            exit 2
        fi
    done <"${queue}"
    date -u +%Y%m%dT%H%M%SZ >"${pass_marker}"
    echo "WORKER ${gpu} PASS"
}


coordinate_mode() {
    local output_root="$1"
    local -a coordinator_gpus
    local gpu
    local all_pass
    read -r -a coordinator_gpus <<<"$(read_manifest_gpu_list "${output_root}")"
    while true; do
        for gpu in "${coordinator_gpus[@]}"; do
            if [[ -f "${output_root}/workers/WORKER_${gpu}_FAILED" ]]; then
                date -u +%Y%m%dT%H%M%SZ >"${output_root}/MATRIX_FAILED"
                exit 2
            fi
        done

        all_pass=1
        for gpu in "${coordinator_gpus[@]}"; do
            if [[ ! -f "${output_root}/workers/WORKER_${gpu}_PASS" ]]; then
                all_pass=0
            fi
        done
        if (( all_pass == 1 )); then
            break
        fi
        sleep 15
    done
    if ! "${PYTHON}" "${AGGREGATOR}" "${output_root}"; then
        date -u +%Y%m%dT%H%M%SZ >"${output_root}/MATRIX_FAILED"
        exit 2
    fi
}


ensure_workers_stopped() {
    local output_root="$1"
    local pid_file
    local pid
    for pid_file in "${output_root}"/workers/gpu*.pid \
        "${output_root}/workers/coordinator.pid"; do
        [[ -f "${pid_file}" ]] || continue
        pid="$(cat "${pid_file}")"
        [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || \
            fail "Invalid evaluation PID file: ${pid_file}"
        if kill -0 "${pid}" 2>/dev/null && \
            ps -p "${pid}" -o args= | grep -Fq "${output_root}"; then
            fail "An evaluation worker is already running: pid=${pid}"
        fi
    done
}


start_mode() {
    parse_gpu_selection
    require_runtime
    resolve_models
    export LD_LIBRARY_PATH="${ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

    local output_root="${1:-${ROOT}/go2_formal_eval_$(date -u +%Y%m%dT%H%M%SZ)}"
    mkdir -p "${output_root}/workers"
    output_root="$(cd "${output_root}" && pwd -P)"
    ensure_workers_stopped "${output_root}"
    rm -f "${output_root}"/workers/gpu*.pid \
        "${output_root}/workers/coordinator.pid"
    rm -f "${output_root}/MATRIX_FAILED" "${output_root}/FORMAL_MATRIX_PASS" \
        "${output_root}/formal_policy_summary.csv" \
        "${output_root}/formal_policy_summary.json" "${output_root}/SHA256SUMS" \
        "${output_root}"/workers/WORKER_*_FAILED \
        "${output_root}"/workers/WORKER_*_PASS

    local gpu
    for gpu in "${PHYSICAL_GPUS[@]}"; do
        gpu_preflight "${gpu}"
    done
    write_manifest "${output_root}"
    seed_smoke_gate "${output_root}" "${PHYSICAL_GPUS[0]}"
    write_queues "${output_root}"
    printf '%s\n' "${output_root}" >"${LATEST_POINTER}"

    local worker_pid
    for gpu in "${PHYSICAL_GPUS[@]}"; do
        nohup "${SCRIPT_PATH}" worker "${gpu}" "${output_root}" \
            "${output_root}/workers/gpu${gpu}.queue" \
            >"${output_root}/workers/gpu${gpu}.log" 2>&1 </dev/null &
        worker_pid=$!
        printf '%s\n' "${worker_pid}" >"${output_root}/workers/gpu${gpu}.pid"
        echo "STARTED worker_gpu=${gpu} pid=${worker_pid}"
    done

    nohup "${SCRIPT_PATH}" coordinate "${output_root}" \
        >"${output_root}.coordinator.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"${output_root}/workers/coordinator.pid"
    echo "OUTPUT_ROOT=${output_root}"
    echo "FORMAL EVALUATION MATRIX START PASS"
}


if [[ "${QRC_SOURCE_ONLY:-0}" == "1" ]]; then
    return 0 2>/dev/null || exit 0
fi


MODE="${1:-start}"
case "${MODE}" in
    start)
        shift || true
        start_mode "${1:-}"
        ;;
    worker)
        [[ $# -eq 4 ]] || fail "worker requires GPU OUTPUT_ROOT QUEUE"
        worker_mode "$2" "$3" "$4"
        ;;
    coordinate)
        [[ $# -eq 2 ]] || fail "coordinate requires OUTPUT_ROOT"
        coordinate_mode "$2"
        ;;
    *)
        fail "Usage: $0 start [OUTPUT_ROOT] | worker GPU OUTPUT_ROOT QUEUE | coordinate OUTPUT_ROOT"
        ;;
esac
