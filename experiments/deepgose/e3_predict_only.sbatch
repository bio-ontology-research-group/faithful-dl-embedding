#!/usr/bin/env bash
#SBATCH --job-name=gfe-deepgose-predict
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_predict_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_predict_%j.err

# Recover GroundedEL cells whose job died after the best checkpoint was saved but before
# predictions were written: CELLS="ont:seed[:model_name] ..." (model_name default grounded_esm).
# Usage: CELLS="cc:0 cc:2 mf:2" sbatch e3_predict_only.sbatch

set -uo pipefail
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
cd "${REPO_DIR}"
: "${CELLS:?set CELLS=\"ont:seed[:model] ...\"}"
DIM="${DIM:-128}"; K="${K:-1}"

status=0
for cell in ${CELLS}; do
  IFS=: read -r ont seed model <<< "${cell}"
  model="${model:-grounded_esm}"
  lambda=1.0; [[ "${model}" == *_l0 ]] && lambda=0.0
  echo "[predict-only] ${ont} ${model} seed ${seed}"
  "${PYTHON}" -m experiments.deepgose.train --ont "${ont}" --seed "${seed}" --model-id "${seed}" \
      --model-name "${model}" --dim "${DIM}" --K "${K}" --lambda-tbox "${lambda}" \
      --data-root "${DATA_ROOT}" --out "${OUT_DIR}" --device cuda --predict-only \
      2>&1 | tee -a "${LOG_DIR}/${ont}__${model}__d${DIM}__K${K}__s${seed}__predict_only.log" || status=1
done
echo "[done] e3_predict_only status=${status}"
exit ${status}
