#!/usr/bin/env bash
# Total-budget curves (experiments.capacity.budget_total) for the large ontologies.
# Start:  nohup setsid experiments/capacity/run_budget_total.sh < /dev/null > /dev/null 2>&1 &
# Environment: DATASETS (default "foodon go"), PROTECT (default both), TIME_LIMIT (s per curve).
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${HOME}/.venvs/mowl-ws/bin/python}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/experiments/results/raw-capacity/logs}"
DATASETS="${DATASETS:-foodon go}"
PROTECT="${PROTECT:-both}"
COST="${COST:-size}"
TIME_LIMIT="${TIME_LIMIT:-21600}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
ulimit -v 40000000
for ds in ${DATASETS}; do
    "${PYTHON}" -m experiments.capacity.budget_total --dataset "${ds}" --protect "${PROTECT}" --cost "${COST}" \
        --time-limit "${TIME_LIMIT}" --skip-existing > "${LOG_DIR}/budget_total_${ds}_${COST}.log" 2>&1
    echo "[exit ${ds}] $?" >> "${LOG_DIR}/budget_total_${ds}_${COST}.log"
done
