#!/usr/bin/env bash
#SBATCH --job-name=gfe-qtocladj
#SBATCH --partition=pi-hohndor
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/baselines/qtoclosedadj_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/baselines/qtoclosedadj_%j.err
# CPU-only step 1 of the "deduce from predicted facts" pilot (H1, part A): rule closure of a QTO neural adjacency
# under the mined + verified E-OMQA TBox. No GPU (the closure is scipy on CPU). Run this first for NELL (load + closure
# + save of the 8-11 GB adjacency is expected to take well over 10 minutes), then qto_closed.sbatch with the same
# variables, e.g.
#   J=$(sbatch --parsable --export=ALL,KB=nell,SRC=nell qto_closed_adj.sbatch)
#   sbatch --dependency=afterok:$J --export=ALL,KB=nell,SRC=nell qto_closed.sbatch
# Variables: KB (lubm|nell), SRC (source adjacency key, e.g. lubm, lubm_onto, nell, nell_onto), DECAY (default 1.0),
# MINSCORE (default 0). Output: neural_adj/${SRC}C<tag>_10_0.0002.pt, tag = d<DECAY>m<MINSCORE> with '.' -> 'p'.
set -euo pipefail
: "${KB:?set KB=lubm|nell}"
: "${SRC:?set SRC (source adjacency key, e.g. ${KB} or ${KB}_onto)}"
DECAY=${DECAY:-1.0}
MINSCORE=${MINSCORE:-0}
G=/mnt/data/u/hohndor/gfe
B=$G/baselines
REPO=$G/faithful-dl-embedding
PY=$G/venv/bin/python
H=$G/runs/qa/hybrid
export TMPDIR=$G/tmp
export OMP_NUM_THREADS=8
TAG="d${DECAY}m${MINSCORE}"; TAG=${TAG//./p}
KEY="${SRC}C${TAG}"
[[ "$KEY" =~ ^[A-Za-z0-9_]+$ ]] || { echo "bad key $KEY (no '-', '/' or '.')"; exit 1; }
case "$SRC" in *_onto) SUF=-onto ;; *) SUF= ;; esac
IN=$B/src/QTO/neural_adj/${SRC}_10_0.0002.pt
OUT=$B/src/QTO/neural_adj/${KEY}_10_0.0002.pt
mkdir -p $H $B/logs
if [ -e "$OUT" ]; then echo "exists: $OUT (skip)"; exit 0; fi
ls -la "$IN" $H/tbox_${KB}.json
cd $REPO
/usr/bin/time -v $PY -m experiments.omqa.hybrid.closure_adj --adj "$IN" --tbox $H/tbox_${KB}.json \
  --data-dir $B/data/${KB}-ded${SUF} --out "$OUT" --decay $DECAY --min-score $MINSCORE --max-iter 10 \
  --report $H/closure_${KEY}.json 2>&1 | grep -vE "UserWarning|_validate_loaded_sparse|sparse_coo_tensor\(" | tee $B/logs/qto_closed_adj_${KEY}.log
ls -la "$OUT"
