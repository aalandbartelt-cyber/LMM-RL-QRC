#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ENV_PREFIX="${QRC_ENV_PREFIX:-${HOME}/conda/envs/go2campus}"
OUTPUT_DIR="${QRC_OUTPUT_DIR:-${HOME}/go2_work/campus_smoke}"
mkdir -p "${OUTPUT_DIR}"

for gpu in 0 1; do
    echo "===== PHYSICAL GPU ${gpu}: baseline ====="
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_PREFIX}/bin/python" \
        "${SCRIPT_DIR}/smoke_isaac.py" \
        --task Unitree-Go2-Velocity --num-envs 16 --steps 8 --device cuda:0 \
        2>&1 | tee "${OUTPUT_DIR}/gpu_${gpu}_baseline.log"

    echo "===== PHYSICAL GPU ${gpu}: campus ====="
    CUDA_VISIBLE_DEVICES="${gpu}" "${ENV_PREFIX}/bin/python" \
        "${SCRIPT_DIR}/smoke_isaac.py" \
        --task Unitree-Go2-Campus-Velocity --num-envs 64 --steps 16 --device cuda:0 \
        2>&1 | tee "${OUTPUT_DIR}/gpu_${gpu}_campus.log"
done

echo "TWO GPU ISAAC SMOKE MATRIX PASS"
