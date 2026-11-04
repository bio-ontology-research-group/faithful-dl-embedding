#!/usr/bin/env bash
#SBATCH --job-name=gfe-grounded-go
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/grounded-e2/go_%A_%a.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/grounded-e2/go_%A_%a.err

# Slurm array job for GroundedEL E2 (GO rate-distortion sweep on ORIX).
# Grid: K in {1, 2, 4, 8} x d in {50, 200} x eps_min in {0, 0.01} (16 cells).

set -euo pipefail

GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DATA_DIR="${DATA_DIR:-/mnt/data/u/hohndor/gfe/data}"
OUT_DIR="${OUT_DIR:-/mnt/data/u/hohndor/gfe/runs/grounded-e2}"
LOG_DIR="${LOG_DIR:-/mnt/data/u/hohndor/gfe/logs/grounded-e2}"
export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
export MOWL_DATA_ROOT="${MOWL_DATA_ROOT:-${DATA_DIR}/mowl}"
export PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMPDIR}" "${MOWL_DATA_ROOT}"

GRID_FILE="${GRID_FILE:-${OUT_DIR}/go_grid.txt}"
if [[ ! -f "${GRID_FILE}" ]]; then
    tmp_grid="${GRID_FILE}.tmp.$$"
    : > "${tmp_grid}"
    for k in 1 2 4 8; do
        for d in 50 200; do
            for eps in 0 0.01; do
                echo "${k} ${d} ${eps}" >> "${tmp_grid}"
            done
        done
    done
    mv -f "${tmp_grid}" "${GRID_FILE}"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NUM=$((TASK_ID + 1))
CELL_LINE="$(sed -n "${LINE_NUM}p" "${GRID_FILE}")"
if [[ -z "${CELL_LINE}" ]]; then
    echo "[ERROR] No grid entry at line ${LINE_NUM} in ${GRID_FILE}" >&2
    exit 1
fi

read -r K D EPS <<< "${CELL_LINE}"

EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
LR="${LR:-0.01}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"

cd "${REPO_DIR}"

echo "[RUN] task=${TASK_ID} K=${K} d=${D} eps_min=${EPS} epochs=${EPOCHS} batch_size=${BATCH_SIZE} device=${DEVICE}"
"${PYTHON}" -m experiments.grounded.run_go \
    --K "${K}" \
    --d "${D}" \
    --eps-min "${EPS}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --data-root "${MOWL_DATA_ROOT}" \
    --out "${OUT_DIR}" \
    2>&1 | tee "${LOG_DIR}/go__K${K}__d${D}__eps${EPS}__s${SEED}.log"
