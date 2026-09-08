#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 RUN_DIR CHECKPOINT_DIR [OUTPUT_DIR] [ARCHIVE_NAME]" >&2
    exit 2
fi

RUN_DIR="$(realpath "$1")"
CHECKPOINT_DIR="$(realpath "$2")"
OUTPUT_DIR="${3:-${HOME}/go2_work/campus_artifacts}"
ARCHIVE_NAME="${4:-$(basename "${RUN_DIR}")}_result"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
ENV_PREFIX="${QRC_ENV_PREFIX:-${HOME}/conda/envs/go2campus}"
UNITREE_DIR="${QRC_UNITREE_DIR:-${HOME}/go2_work/upstream/unitree_rl_lab}"
PROVENANCE_DIR="${RUN_DIR}/provenance"
OUTPUT="${OUTPUT_DIR}/${ARCHIVE_NAME}.tar.gz"

mkdir -p "${PROVENANCE_DIR}" "${OUTPUT_DIR}"
nvidia-smi >"${PROVENANCE_DIR}/nvidia_smi.txt"
"${ENV_PREFIX}/bin/python" -m pip freeze >"${PROVENANCE_DIR}/pip_freeze.txt"
git -C "${REPO_ROOT}" status --short --branch >"${PROVENANCE_DIR}/project_git_status.txt"
git -C "${REPO_ROOT}" rev-parse HEAD >"${PROVENANCE_DIR}/project_commit.txt"
git -C "${UNITREE_DIR}" status --short --branch >"${PROVENANCE_DIR}/unitree_git_status.txt"
git -C "${UNITREE_DIR}" rev-parse HEAD >"${PROVENANCE_DIR}/unitree_commit.txt"
cp "${SCRIPT_DIR}/versions.json" "${PROVENANCE_DIR}/versions.json"

"${ENV_PREFIX}/bin/python" \
    "${REPO_ROOT}/rl_control/scripts/package_training_run.py" \
    --run-dir "${RUN_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output "${OUTPUT}"

echo "PACKAGE VERIFIED: ${OUTPUT}"
echo "Download the .tar.gz, .sha256 and .manifest.json files together."
