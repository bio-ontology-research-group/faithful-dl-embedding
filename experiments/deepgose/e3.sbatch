#!/usr/bin/env bash
#SBATCH --job-name=gfe-deepgose-e3
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-23
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_%A_%a.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_%A_%a.err

# Slurm array job for E3 (DeepGO-SE inductive protein function prediction on ORIX).
# First round (24 cells):
#   ont in {cc, mf} x seed in {0, 1, 2} x
#   {grounded d128 K1 lambda=1, grounded d128 K1 lambda=0, deepgozero_esm, mlp_esm}

set -euo pipefail

GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DEEPGO2_DIR="${DEEPGO2_DIR:-/mnt/data/u/hohndor/gfe/deepgo2}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/u/hohndor/gfe/data/deepgo2/data}"
OUT_DIR="${OUT_DIR:-/mnt/data/u/hohndor/gfe/runs/deepgose}"
LOG_DIR="${LOG_DIR:-/mnt/data/u/hohndor/gfe/logs/deepgose}"
export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
export DEEPGO2_DIR
export PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${DEEPGO2_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMPDIR}"

CELLS_FILE="${CELLS_FILE:-${OUT_DIR}/e3_cells.txt}"
if [[ ! -f "${CELLS_FILE}" ]]; then
    tmp_cells="${CELLS_FILE}.tmp.$$"
    : > "${tmp_cells}"
    for ont in cc mf; do
        for seed in 0 1 2; do
            echo "${ont} ${seed} grounded_esm 128 1 1.0" >> "${tmp_cells}"
            echo "${ont} ${seed} grounded_esm_l0 128 1 0.0" >> "${tmp_cells}"
            echo "${ont} ${seed} deepgozero_esm 0 0 0.0" >> "${tmp_cells}"
            echo "${ont} ${seed} mlp_esm 0 0 0.0" >> "${tmp_cells}"
        done
    done
    mv -f "${tmp_cells}" "${CELLS_FILE}"
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NUM=$((TASK_ID + 1))
CELL_LINE="$(sed -n "${LINE_NUM}p" "${CELLS_FILE}")"
if [[ -z "${CELL_LINE}" ]]; then
    echo "[ERROR] No cell entry at line ${LINE_NUM} in ${CELLS_FILE}" >&2
    exit 1
fi

read -r ONT SEED KIND DIM K LAMBDA_TBOX <<< "${CELL_LINE}"
DEVICE="${DEVICE:-cuda}"
TBOX_BATCH="${TBOX_BATCH:-4096}"
GROUNDED_EPOCHS="${GROUNDED_EPOCHS:-64}"

cd "${REPO_DIR}"

echo "[RUN] task=${TASK_ID} ont=${ONT} seed=${SEED} kind=${KIND} dim=${DIM} K=${K} lambda_tbox=${LAMBDA_TBOX} device=${DEVICE}"

if [[ "${KIND}" == "grounded_esm" || "${KIND}" == "grounded_esm_l0" ]]; then
    "${PYTHON}" -m experiments.deepgose.train \
        --ont "${ONT}" \
        --seed "${SEED}" \
        --model-id "${SEED}" \
        --model-name "${KIND}" \
        --dim "${DIM}" \
        --K "${K}" \
        --lambda-tbox "${LAMBDA_TBOX}" \
        --epochs "${GROUNDED_EPOCHS}" \
        --protein-batch 128 \
        --tbox-batch "${TBOX_BATCH}" \
        --lr-encoder 5e-4 \
        --lr-boxes 1e-2 \
        --lr-final 1e-4 \
        --device "${DEVICE}" \
        --data-root "${DATA_ROOT}" \
        --out "${OUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/${ONT}__${KIND}__d${DIM}__K${K}__s${SEED}.log"
elif [[ "${KIND}" == "deepgozero_esm" ]]; then
    "${PYTHON}" -m experiments.deepgose.baselines.train_deepgo \
        --data-root "${DATA_ROOT}" \
        --out "${OUT_DIR}" \
        --ont "${ONT}" \
        --model-name "deepgozero_esm" \
        --model-id "${SEED}" \
        --seed "${SEED}" \
        --batch-size 37 \
        --epochs 128 \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_DIR}/${ONT}__deepgozero_esm__s${SEED}.log"
elif [[ "${KIND}" == "mlp_esm" ]]; then
    "${PYTHON}" -m experiments.deepgose.baselines.train_mlp \
        --data-root "${DATA_ROOT}" \
        --out "${OUT_DIR}" \
        --ont "${ONT}" \
        --model-name "mlp_esm" \
        --model-id "${SEED}" \
        --seed "${SEED}" \
        --batch-size 37 \
        --epochs 128 \
        --device "${DEVICE}" \
        2>&1 | tee "${LOG_DIR}/${ONT}__mlp_esm__s${SEED}.log"
else
    echo "[ERROR] Unknown cell kind: ${KIND}" >&2
    exit 1
fi

"${PYTHON}" -m experiments.deepgose.evaluate \
    --ont "${ONT}" \
    --model-name "${KIND}" \
    --model-ids "${SEED}" \
    --n-models 1 \
    --combine avg \
    --data-root "${DATA_ROOT}" \
    --out "${OUT_DIR}" \
    --device "${DEVICE}" \
    2>&1 | tee -a "${LOG_DIR}/${ONT}__${KIND}__eval__s${SEED}.log"
