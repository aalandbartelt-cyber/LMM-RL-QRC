#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
ENV_PREFIX="${QRC_ENV_PREFIX:-${HOME}/conda/envs/go2campus}"
UPSTREAM_ROOT="${QRC_UPSTREAM_ROOT:-${HOME}/go2_work/upstream}"
ISAAC_LAB_DIR="${UPSTREAM_ROOT}/IsaacLab"
UNITREE_DIR="${UPSTREAM_ROOT}/unitree_rl_lab"
UNITREE_MODEL_DIR="${UPSTREAM_ROOT}/unitree_model"
REQUESTED_GPUS="${QRC_GPUS:-2}"

PYTHON_VERSION="3.11"
ISAAC_SIM_VERSION="5.1.0"
ISAAC_LAB_COMMIT="3c6e67bb5c7ada942a6d1884ab69338f57596f77"
UNITREE_COMMIT="1330cb0d4236c4abd6bf1efec05a4fce7ad79678"
RSL_RL_VERSION="2.3.1"

ALLOW_A100=()
if [[ "${QRC_ALLOW_UNSUPPORTED_A100:-0}" == "1" ]]; then
    ALLOW_A100=(--allow-unsupported-a100)
fi

python3 "${SCRIPT_DIR}/preflight_self5000.py" --phase host \
    --gpus "${REQUESTED_GPUS}" "${ALLOW_A100[@]}" \
    --output "${HOME}/go2_work/self5000_host_preflight.json"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization script was not found." >&2
    exit 2
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    conda create --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_PREFIX}"
python -m pip install --upgrade pip setuptools wheel

clone_exact() {
    local url="$1"
    local destination="$2"
    local commit="$3"
    if [[ ! -d "${destination}/.git" ]]; then
        mkdir -p "$(dirname "${destination}")"
        git clone --filter=blob:none --no-checkout "${url}" "${destination}"
        git -C "${destination}" fetch --depth 1 origin "${commit}"
        git -C "${destination}" checkout --detach "${commit}"
    fi
    local actual
    actual="$(git -C "${destination}" rev-parse HEAD)"
    if [[ "${actual}" != "${commit}" ]]; then
        echo "Revision mismatch at ${destination}: ${actual} != ${commit}" >&2
        exit 2
    fi
    printf '%s\n' "${actual}" >"${destination}/.qrc-upstream-commit"
}

clone_exact "https://github.com/isaac-sim/IsaacLab.git" "${ISAAC_LAB_DIR}" "${ISAAC_LAB_COMMIT}"
clone_exact "https://github.com/unitreerobotics/unitree_rl_lab.git" "${UNITREE_DIR}" "${UNITREE_COMMIT}"

python -m pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
    --extra-index-url https://pypi.nvidia.com

(
    cd "${ISAAC_LAB_DIR}"
    ./isaaclab.sh -i
)
python -m pip install --force-reinstall "rsl-rl-lib==${RSL_RL_VERSION}"
python -m pip install --upgrade huggingface_hub

if [[ ! -f "${UNITREE_MODEL_DIR}/Go2/usd/go2.usd" ]]; then
    UNITREE_MODEL_DIR="${UNITREE_MODEL_DIR}" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="unitreerobotics/unitree_model",
    repo_type="dataset",
    local_dir=os.environ["UNITREE_MODEL_DIR"],
)
PY
fi

python "${SCRIPT_DIR}/preflight_self5000.py" --phase configure-assets \
    --unitree-config "${UNITREE_DIR}/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py" \
    --unitree-model-dir "${UNITREE_MODEL_DIR}"

python -m pip install --editable "${UNITREE_DIR}/source/unitree_rl_lab"
python -m pip install --editable "${REPO_ROOT}/rl_control"

python "${SCRIPT_DIR}/preflight_self5000.py" --phase runtime \
    --gpus "${REQUESTED_GPUS}" \
    --output "${HOME}/go2_work/self5000_runtime_preflight.json"

echo "SELF5000 BOOTSTRAP PASS"
