#!/usr/bin/env bash
#SBATCH --job-name=gfe-deepgose-utilprobe
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_utilprobe_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_utilprobe_%j.err
# Short real training run (default 2 epochs, cc, lambda=1) with nvidia-smi utilisation sampling,
# to check the >75 % GPU-utilisation requirement of orix before long runs. Writes to a scratch OUT_DIR.
set -uo pipefail
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DEEPGO2_DIR="${DEEPGO2_DIR:-/mnt/data/u/hohndor/gfe/deepgo2}"
DATA_ROOT="${DATA_ROOT:-/mnt/data/u/hohndor/gfe/data/deepgo2/data}"
OUT_DIR="${OUT_DIR:-/mnt/data/u/hohndor/gfe/runs/deepgose-probe}"
export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
export DEEPGO2_DIR PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${DEEPGO2_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"
ONT="${ONT:-cc}"; EPOCHS="${EPOCHS:-2}"; TBOX_BATCH="${TBOX_BATCH:-1024}"; LAMBDA="${LAMBDA:-1.0}"
DMON="${OUT_DIR}/dmon_${ONT}_${SLURM_JOB_ID}.txt"
nvidia-smi dmon -s u -d 5 > "${DMON}" 2>&1 &
DPID=$!
"${PYTHON}" -m experiments.deepgose.train --ont "${ONT}" --seed 0 --model-id 0 --model-name probe_grounded \
    --dim 128 --K 1 --lambda-tbox "${LAMBDA}" --epochs "${EPOCHS}" --protein-batch 128 --tbox-batch "${TBOX_BATCH}" \
    --data-root "${DATA_ROOT}" --out "${OUT_DIR}" --device cuda
status=$?
kill ${DPID} 2>/dev/null
echo "[dmon] sm% samples (5 s each), full run:"
grep -v '^#' "${DMON}" | awk '{print $2}' | tr '\n' ' '; echo
echo "[dmon] sm% over the last 40 samples (training phase):"
grep -v '^#' "${DMON}" | tail -40 | awk '{s+=$2; n++} END {if (n) printf "%.1f%% over %d samples\n", s/n, n}'
exit ${status}
