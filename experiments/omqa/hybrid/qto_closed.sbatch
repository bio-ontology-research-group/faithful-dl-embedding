#!/usr/bin/env bash
#SBATCH --job-name=gfe-qtoclosed
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/baselines/qtoclosed_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/baselines/qtoclosed_%j.err
# "Deduce from predicted facts" pilot (H1, part A): QTO inference (all nine shapes, cases ded and ind_ded) on the rule
# closure of a QTO neural adjacency, unmodified QTO main.py. Variables: KB (lubm|nell), SRC (source adjacency key,
# e.g. lubm, lubm_onto, nell, nell_onto), DECAY (default 1.0), MINSCORE (default 0).
#   sbatch --export=ALL,KB=lubm,SRC=lubm qto_closed.sbatch
# Step 1 (closure, CPU) runs inline only for LUBM (~1.5 min on the workstation). For NELL run qto_closed_adj.sbatch
# (CPU-only, no GPU) first and submit this job with --dependency=afterok:<that job>, so the GPU is not idle for the
# closure; this job stops if the NELL closure is missing.
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
TAG="d${DECAY}m${MINSCORE}"; TAG=${TAG//./p}
KEY="${SRC}C${TAG}"
[[ "$KEY" =~ ^[A-Za-z0-9_]+$ ]] || { echo "bad key $KEY (no '-', '/' or '.')"; exit 1; }
case "$SRC" in *_onto) SUF=-onto ;; *) SUF= ;; esac
OUT=$B/src/QTO/neural_adj/${KEY}_10_0.0002.pt
mkdir -p $H $B/logs $B/runs/kbc
# (1) closure on CPU (skip if it exists)
if [ ! -e "$OUT" ]; then
  if [ "$KB" != "lubm" ]; then echo "missing $OUT: run qto_closed_adj.sbatch (CPU) first"; exit 1; fi
  ( cd $REPO && OMP_NUM_THREADS=8 $PY -m experiments.omqa.hybrid.closure_adj --adj $B/src/QTO/neural_adj/${SRC}_10_0.0002.pt \
      --tbox $H/tbox_${KB}.json --data-dir $B/data/${KB}-ded${SUF} --out "$OUT" --decay $DECAY --min-score $MINSCORE \
      --max-iter 10 --report $H/closure_${KEY}.json 2>&1 | grep -vE "UserWarning|_validate_loaded_sparse|sparse_coo_tensor\(" \
      | tee $B/logs/qto_closed_adj_${KEY}.log )
fi
ls -la "$OUT"
cd $B/src/QTO
export PYTHONPATH=$B/src/QTO/kbc:$B/src/QTO
mkdir -p results neural_adj data
nvidia-smi dmon -s u -d 10 > $B/runs/kbc/${KEY}.closed.dmon.txt 2>&1 &
DPID=$!
for CASE in ded ind_ded; do
  # (2) QTO derives the adjacency key from data/<key>-<case>: point it at the data dirs of the source key
  [ -e data/${KEY}-${CASE} ] || ln -s $B/data/${KB}-${CASE}${SUF} data/${KEY}-${CASE}
  # (3) inference, all nine shapes
  $PY main.py --data_path data/${KEY}-${CASE} --kbc_path kbc/${SRC}/best_valid.model --fraction 10 --thrshd 0.0002 \
    --neg_scale 6 --tasks 1p.2p.3p.2i.3i.ip.pi.2u.up --cpu_num 4 > $B/logs/qto_closed_${KEY}_${CASE}.log 2>&1
done
kill $DPID 2>/dev/null || true
for CASE in ded ind_ded; do
  echo "== $KEY $CASE"; grep -E "^Test [0-9a-zA-Z-]+ HITS3_hard" $B/logs/qto_closed_${KEY}_${CASE}.log || true
done
grep -v '^#' $B/runs/kbc/${KEY}.closed.dmon.txt | awk '{s+=$2; n++} END {if (n) printf "[dmon] mean sm%% %.1f over %d samples\n", s/n, n}'
