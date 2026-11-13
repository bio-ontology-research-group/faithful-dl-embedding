#!/usr/bin/env bash
#SBATCH --job-name=gfe-capacity-baselines
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/capacity-baselines/baselines_%A_%a.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/capacity-baselines/baselines_%A_%a.err

# B3 single-box baselines (ELEm, ELBE, Box2EL, TransBox) on ORIX.
#
# Each array task takes CELLS_PER_JOB consecutive lines of the cells file
# (lines TASK_ID*CELLS_PER_JOB .. TASK_ID*CELLS_PER_JOB+CELLS_PER_JOB-1, one cell
# "<dataset> <method> <dim> <seed>" per line) and runs them CONCURRENTLY on the one
# GPU of the job (jobs with low GPU utilisation are cancelled on this cluster).
# The full grid (baselines_cells.txt, 1344 cells) is split: pizza and the synthetic
# theories take seconds per cell and run on a workstation CPU (run_baselines_small.sh);
# GO and FoodOn (baselines_cells_large.txt, 192 cells) run here, 6 cells per GPU,
# 32 array tasks, with 3000 epochs (the harness default of 300 under-fits):
#   sbatch --array=0-31 --export=ALL,EPOCHS=3000,\
#CELLS_FILE=$REPO_DIR/experiments/capacity/baselines_cells_large.txt \
#          experiments/capacity/baselines.sbatch
# Cells whose JSON already exists are skipped, so failed tasks can be resubmitted.
# Recommended once before the array (downloads data, builds CanonicalModel caches):
#   ${PYTHON} -m experiments.capacity.baselines --prepare pizza go foodon
# (each task also runs --prepare for its own datasets; cached models are reused).

set -euo pipefail

GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DATA_DIR="${DATA_DIR:-/mnt/data/u/hohndor/gfe/data}"
OUT_DIR="${OUT_DIR:-${REPO_DIR}/experiments/results/raw-capacity/baselines}"
LOG_DIR="${LOG_DIR:-/mnt/data/u/hohndor/gfe/logs/capacity-baselines}"
CKPT_DIR="${CKPT_DIR:-/mnt/data/u/hohndor/gfe/runs/capacity-baselines-ckpt}"
CELLS_FILE="${CELLS_FILE:-${REPO_DIR}/experiments/capacity/baselines_cells.txt}"
CELLS_PER_JOB="${CELLS_PER_JOB:-6}"
DEVICE="${DEVICE:-cuda}"
EPOCHS_ARG=()
if [[ -n "${EPOCHS:-}" ]]; then
    EPOCHS_ARG=(--epochs "${EPOCHS}")
fi

export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
case "${TMPDIR}" in
    /tmp|/tmp/*) export TMPDIR="/mnt/data/u/hohndor/gfe/tmp" ;;
esac
export GFE_DATA_ROOT="${GFE_DATA_ROOT:-${DATA_DIR}}"
export MOWL_DATA_ROOT="${MOWL_DATA_ROOT:-${DATA_DIR}/mowl}"
export GFE_RUNS_DIR="${GFE_RUNS_DIR:-${CKPT_DIR}}"
export PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
# several JVMs + torch processes share the node: keep thread pools small
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMPDIR}" "${MOWL_DATA_ROOT}" "${CKPT_DIR}"
cd "${REPO_DIR}"

# non-comment, non-empty lines of the cells file
mapfile -t CELLS < <(grep -v '^[[:space:]]*\(#\|$\)' "${CELLS_FILE}")
N_CELLS=${#CELLS[@]}
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
FIRST=$((TASK_ID * CELLS_PER_JOB))
if (( FIRST >= N_CELLS )); then
    echo "[INFO] task ${TASK_ID}: no cells (first=${FIRST}, n=${N_CELLS})"
    exit 0
fi
LAST=$((FIRST + CELLS_PER_JOB - 1))
if (( LAST >= N_CELLS )); then
    LAST=$((N_CELLS - 1))
fi

# write this task's slice to its own cells file (indices are 0-based in the slice)
SLICE="${LOG_DIR}/cells_${SLURM_ARRAY_JOB_ID:-local}_${TASK_ID}.txt"
printf '%s\n' "${CELLS[@]:FIRST:LAST-FIRST+1}" > "${SLICE}"
DATASETS=$(awk '{print $1}' "${SLICE}" | sort -u | tr '\n' ' ')
echo "[RUN] task=${TASK_ID} cells ${FIRST}..${LAST} of ${N_CELLS} datasets: ${DATASETS} device=${DEVICE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086
"${PYTHON}" -m experiments.capacity.baselines --prepare ${DATASETS}

PIDS=()
I=0
while read -r DS METHOD DIM SEED; do
    TAG="${DS}__${METHOD}__d${DIM}__s${SEED}"
    "${PYTHON}" -m experiments.capacity.baselines \
        --cells "${SLICE}" --start "${I}" --count 1 --skip-existing \
        --device "${DEVICE}" --out "${OUT_DIR}" --runs-dir "${CKPT_DIR}" ${EPOCHS_ARG[@]+"${EPOCHS_ARG[@]}"} \
        > "${LOG_DIR}/${TAG}.log" 2>&1 &
    PIDS+=("$!")
    echo "[RUN] started ${TAG} pid ${PIDS[-1]}"
    I=$((I + 1))
done < "${SLICE}"

STATUS=0
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        echo "[ERROR] cell process ${PID} failed" >&2
        STATUS=1
    fi
done
echo "[DONE] task=${TASK_ID} status=${STATUS}"
exit "${STATUS}"
