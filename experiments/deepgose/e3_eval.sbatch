#!/usr/bin/env bash
#SBATCH --job-name=gfe-deepgose-eval
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_eval_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_eval_%j.err

# Post-hoc evaluation for E3: every single-seed model (eval_{model}_avg_1_ids{s}.json)
# and the S-seed ensembles (avg and min over all seeds present), per ontology and model.
# Usage: sbatch e3_eval.sbatch            (ONTS="cc mf", MODELS=all four, SEEDS="0 1 2")
#        ONTS=cc MODELS=grounded_esm sbatch e3_eval.sbatch

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
mkdir -p "${LOG_DIR}" "${TMPDIR}"
cd "${REPO_DIR}"

ONTS="${ONTS:-cc mf}"
MODELS="${MODELS:-grounded_esm grounded_esm_l0 deepgozero_esm mlp_esm}"
SEEDS="${SEEDS:-0 1 2}"
DEVICE="${DEVICE:-cuda}"

for ont in ${ONTS}; do
  for model in ${MODELS}; do
    present=()
    for s in ${SEEDS}; do
      [[ -f "${OUT_DIR}/${ont}/test_predictions_${model}_${s}.pkl" ]] && present+=("${s}")
    done
    if [[ ${#present[@]} -eq 0 ]]; then
      echo "[skip] ${ont} ${model}: no predictions"; continue
    fi
    for s in "${present[@]}"; do
      if [[ -f "${OUT_DIR}/${ont}/eval_${model}_avg_1_ids${s}.json" ]]; then
        echo "[have] ${ont} ${model} seed ${s}"; continue
      fi
      echo "[eval] ${ont} ${model} seed ${s}"
      "${PYTHON}" -m experiments.deepgose.evaluate --ont "${ont}" --model-name "${model}" \
        --model-ids "${s}" --n-models 1 --combine avg \
        --data-root "${DATA_ROOT}" --out "${OUT_DIR}" --device "${DEVICE}" \
        2>&1 | tee -a "${LOG_DIR}/${ont}__${model}__eval__s${s}.log"
    done
    n=${#present[@]}
    if [[ ${n} -ge 2 ]]; then
      for comb in avg min; do
        echo "[ensemble] ${ont} ${model} ${comb} over ${present[*]}"
        "${PYTHON}" -m experiments.deepgose.evaluate --ont "${ont}" --model-name "${model}" \
          --model-ids "${present[@]}" --n-models "${n}" --combine "${comb}" \
          --data-root "${DATA_ROOT}" --out "${OUT_DIR}" --device "${DEVICE}" \
          2>&1 | tee -a "${LOG_DIR}/${ont}__${model}__eval__ens_${comb}.log"
      done
    fi
  done
done
echo "[done] e3_eval"
