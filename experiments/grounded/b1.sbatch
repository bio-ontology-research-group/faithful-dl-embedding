#!/usr/bin/env bash
#SBATCH --job-name=gfe-grounded-b1
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/grounded-b1/b1_%A_%a.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/grounded-b1/b1_%A_%a.err

# Slurm array job for GroundedEL on ABox benchmark B1 (yeast-function) on ORIX.
# Grid: K in {1, 2, 4, 8} (grounded-k1..k8) x seeds {0, 1, 2} (12 cells).

set -euo pipefail

GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DATA_DIR="${DATA_DIR:-/mnt/data/u/hohndor/gfe/data}"
RUNS_DIR="${RUNS_DIR:-/mnt/data/u/hohndor/gfe/runs/grounded-b1}"
LOG_DIR="${LOG_DIR:-/mnt/data/u/hohndor/gfe/logs/grounded-b1}"

export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
export GFE_DATA_ROOT="${GFE_DATA_ROOT:-${DATA_DIR}}"
export YEAST_FUNCTION_DIR="${YEAST_FUNCTION_DIR:-/mnt/data/u/hohndor/gfe/data/yeast-function}"
export GFE_RUNS_DIR="${GFE_RUNS_DIR:-${RUNS_DIR}}"
export PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUNS_DIR}" "${LOG_DIR}" "${TMPDIR}"

METHODS=(grounded-k1 grounded-k2 grounded-k4 grounded-k8)
SEEDS=(0 1 2)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
N_SEEDS="${#SEEDS[@]}"
METHOD_IDX=$((TASK_ID / N_SEEDS))
SEED_IDX=$((TASK_ID % N_SEEDS))

if (( METHOD_IDX < 0 || METHOD_IDX >= ${#METHODS[@]} )); then
    echo "[ERROR] Invalid SLURM_ARRAY_TASK_ID=${TASK_ID} (expected 0..11)" >&2
    exit 1
fi

METHOD="${METHODS[${METHOD_IDX}]}"
SEED="${SEEDS[${SEED_IDX}]}"
EPOCHS="${EPOCHS:-2000}"
DEVICE="${DEVICE:-cuda}"

cd "${REPO_DIR}"

echo "[RUN] task=${TASK_ID} dataset=yeast-function method=${METHOD} seed=${SEED} epochs=${EPOCHS} device=${DEVICE} data=${YEAST_FUNCTION_DIR}"
"${PYTHON}" -m experiments.run \
    --dataset yeast-function \
    --method "${METHOD}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" \
    --device "${DEVICE}" \
    --runs-dir "${RUNS_DIR}" \
    2>&1 | tee "${LOG_DIR}/yeast-function__${METHOD}__d50__s${SEED}.log"
