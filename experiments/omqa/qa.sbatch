#!/usr/bin/env bash
#SBATCH --job-name=gfe-qa
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/qa/qa_%A_%a.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/qa/qa_%A_%a.err

# GroundedEL on the query-answering benchmarks (E-OMQA LUBM/NELL, TAR YAGO4/DBpedia) on orix.
# Cell line (CELLS_FILE, one per array task): NAME then train.py arguments, e.g.
#   lubm_plain_d128 --benchmark eomqa --tables runs/qa/tables/lubm_plain/tables.pt --data-root data/qa/eomqa/lubm/data_lubm --dim 128
# The orix policy kills jobs under 75 % GPU utilisation (hourly at hh:30): keep --batch-assertions >= 8192.

set -euo pipefail
GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-${GFE_ROOT}/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-${GFE_ROOT}/mowl}"
REPO_DIR="${REPO_DIR:-${GFE_ROOT}/faithful-dl-embedding}"
OUT_DIR="${OUT_DIR:-${GFE_ROOT}/runs/qa}"
LOG_DIR="${LOG_DIR:-${GFE_ROOT}/logs/qa}"
export TMPDIR="${TMPDIR:-${GFE_ROOT}/tmp}"
export PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMPDIR}"
CELLS_FILE="${CELLS_FILE:?set CELLS_FILE}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CELL_LINE="$(sed -n "$((TASK_ID + 1))p" "${CELLS_FILE}")"
[[ -n "${CELL_LINE}" ]] || { echo "[ERROR] no cell at line $((TASK_ID + 1)) of ${CELLS_FILE}" >&2; exit 1; }
read -r NAME ARGS <<< "${CELL_LINE}"
cd "${REPO_DIR}"
echo "[RUN] task=${TASK_ID} name=${NAME} args='${ARGS}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086
# GPU utilisation record (orix cancels jobs under 75 % sm utilisation): 10-s samples, summary at the end
DMON="${LOG_DIR}/${NAME}.dmon.txt"
nvidia-smi dmon -s u -d 10 > "${DMON}" 2>&1 &
DPID=$!
"${PYTHON}" -m experiments.omqa.train --name "${NAME}" --out "${OUT_DIR}" --device cuda ${ARGS} 2>&1 | tee "${LOG_DIR}/${NAME}.log"
status=${PIPESTATUS[0]}
kill ${DPID} 2>/dev/null
grep -v '^#' "${DMON}" | awk '{s+=$2; n++; if ($2 >= 75) h++} END {if (n) printf "[dmon] mean sm%% %.1f over %d samples (10 s); %.0f%% of samples >= 75%%\n", s/n, n, 100*h/n}' | tee -a "${LOG_DIR}/${NAME}.log"
echo "[done] ${NAME} (exit ${status})"
exit ${status}
