#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 baseline|campus ITERATIONS NUM_ENVS" >&2
    exit 2
fi

STAGE="$1"
ITERATIONS="$2"
NUM_ENVS="$3"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

"${SCRIPT_DIR}/start_training.sh" "${STAGE}" 0 1 "${ITERATIONS}" "${NUM_ENVS}"
sleep 5
"${SCRIPT_DIR}/start_training.sh" "${STAGE}" 1 2 "${ITERATIONS}" "${NUM_ENVS}"

echo "TWO INDEPENDENT SEEDS STARTED"
echo "Run ${SCRIPT_DIR}/show_progress.sh to monitor both jobs."
