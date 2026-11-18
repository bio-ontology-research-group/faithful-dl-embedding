#!/usr/bin/env bash
#SBATCH --job-name=gfe-ens
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/qa/ens_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/qa/ens_%j.err
# Model combination and missed-answer diagnosis (experiments/omqa/ensemble.py).
# usage: sbatch --export=ALL,OUT=name,RUNS="run1 run2 ...",EXTRA="--diagnose" experiments/omqa/ensemble.sbatch
set -euo pipefail
G=/mnt/data/u/hohndor/gfe
export TMPDIR=$G/tmp PYTHONPATH=$G/mowl:$G/faithful-dl-embedding
cd $G/faithful-dl-embedding
DIRS=(); for r in ${RUNS:?}; do DIRS+=("$G/runs/qa/$r"); done
# shellcheck disable=SC2086
$G/venv/bin/python -m experiments.omqa.ensemble --runs "${DIRS[@]}" --out $G/runs/qa/ensembles/${OUT:?}.json ${EXTRA:-}
