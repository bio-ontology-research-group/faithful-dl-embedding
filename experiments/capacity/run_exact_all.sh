#!/usr/bin/env bash
# Launcher for the long B1 (exact.py) and B2 (budget.py) runs on the large ontologies.
#
# Runs every step in sequence under a 40 GB virtual-memory limit, one dataset per
# Python process, logging to ${LOG_DIR} (default ~/tmp/gfe/capacity/logs). B1 steps whose
# JSON exists are skipped; B2 resumes (--skip-existing) and runs the budgets in
# ${BUDGETS} order (default descending: the small budgets are the expensive ones).
# Measured on this machine (2026-11-12): least_model on go_j24 and galen_j24 did not
# finish within 9.5 min; one B2 re-closure on FoodOn takes ~30 s, on GO > 6 min.
#
# Start (from the repository root):
#   nohup bash experiments/capacity/run_exact_all.sh > ~/tmp/gfe/capacity/run_exact_all.out 2>&1 &
# Optional environment: PYTHON, LOG_DIR, DATASETS (B1 list), BUDGET_DATASETS (B2 list),
# BUDGETS, BUDGET_TIME_LIMIT (seconds per dataset and order, default 21600), ORDERS
# (default "dfs"), STEP_TIMEOUT (seconds per B1 process, default 43200).

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-${HOME}/.venvs/mowl-ws/bin/python}"
LOG_DIR="${LOG_DIR:-${HOME}/tmp/gfe/capacity/logs}"
DATASETS="${DATASETS:-go foodon}"
BUDGET_DATASETS="${BUDGET_DATASETS:-foodon go}"
BUDGETS="${BUDGETS:-64 48 32 24 16 12 8 6 4 3 2 1}"
BUDGET_TIME_LIMIT="${BUDGET_TIME_LIMIT:-21600}"
STEP_TIMEOUT="${STEP_TIMEOUT:-43200}"
ORDERS="${ORDERS:-dfs}"
RAW="${REPO_DIR}/experiments/results/raw-capacity"

export TMPDIR="${TMPDIR:-${HOME}/tmp}"
case "${TMPDIR}" in /tmp|/tmp/*) export TMPDIR="${HOME}/tmp" ;; esac
mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
ulimit -v 40000000

for ds in ${DATASETS}; do
    if [[ -f "${RAW}/exact/${ds}.json" ]]; then
        echo "[skip] exact ${ds}"
        continue
    fi
    echo "[$(date +%FT%T)] exact ${ds}"
    timeout "${STEP_TIMEOUT}" "${PYTHON}" -u -m experiments.capacity.exact --dataset "${ds}" \
        > "${LOG_DIR}/exact_${ds}.log" 2>&1 \
        || echo "[FAIL] exact ${ds} (see ${LOG_DIR}/exact_${ds}.log)"
done

for order in ${ORDERS}; do
    for ds in ${BUDGET_DATASETS}; do
        echo "[$(date +%FT%T)] budget ${ds} order=${order}"
        # shellcheck disable=SC2086
        timeout "$((BUDGET_TIME_LIMIT + 3600))" "${PYTHON}" -u -m experiments.capacity.budget \
            --dataset "${ds}" --order "${order}" --budgets ${BUDGETS} --skip-existing \
            --time-limit "${BUDGET_TIME_LIMIT}" >> "${LOG_DIR}/budget_${ds}_${order}.log" 2>&1 \
            || echo "[FAIL] budget ${ds} (see ${LOG_DIR}/budget_${ds}_${order}.log)"
    done
done

"${PYTHON}" -m experiments.capacity.aggregate --notes "${REPO_DIR}/experiments/capacity/CAPACITY_NOTES.md"
echo "[$(date +%FT%T)] done"
