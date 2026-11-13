#!/usr/bin/env bash
#SBATCH --job-name=gfe-qa-prep
#SBATCH --partition=pi-hohndor
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/qa/prep_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/qa/prep_%j.err
# Build the E-OMQA prepared-split caches (<case>/<part>.prepared.pt) on a CPU-only allocation so that the GPU
# training jobs start scoring within seconds (orix cancels GPU jobs under 75 % utilisation).
set -euo pipefail
GFE_ROOT="${GFE_ROOT:-/mnt/data/u/hohndor/gfe}"
PYTHON="${PYTHON:-${GFE_ROOT}/venv/bin/python}"
export TMPDIR="${TMPDIR:-${GFE_ROOT}/tmp}"
export PYTHONPATH="${GFE_ROOT}/mowl:${GFE_ROOT}/faithful-dl-embedding:${PYTHONPATH:-}"
cd "${GFE_ROOT}/faithful-dl-embedding"
for root in ${DATA_ROOTS:?set DATA_ROOTS (space-separated data_<kb> dirs)}; do
  echo "[prep] ${root}"
  "${PYTHON}" -m experiments.omqa.protocol --data-root "${root}"
done
echo "[done] prepare caches"
