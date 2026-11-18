#!/usr/bin/env bash
#SBATCH --job-name=gfe-predtab
#SBATCH --partition=pi-hohndor
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/baselines/predtab_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/baselines/predtab_%j.err
# CPU-only (H1, part B): predicted facts from the QTO adjacency trained on the enriched facts, then GroundedEL tables
# = enriched facts + predicted facts, one table set per threshold. Variable: KB (lubm|nell), THRS (default "0.5 0.3").
set -euo pipefail
: "${KB:?set KB}"
THRS=${THRS:-"0.5 0.3"}
G=/mnt/data/u/hohndor/gfe
B=$G/baselines
H=$G/runs/qa/hybrid
PY=$G/venv/bin/python
export TMPDIR=$G/tmp OMP_NUM_THREADS=8
cd $G/faithful-dl-embedding
D=$H/pred_${KB}_onto
mkdir -p $D
$PY -m experiments.omqa.hybrid.predicted_facts --adj $B/src/QTO/neural_adj/${KB}_onto_10_0.0002.pt \
  --data-dir $B/data/${KB}-ded-onto --tbox $H/tbox_${KB}.json --thresholds $THRS --max-per-head-rel 3 --out-dir $D \
  2>&1 | grep -vE "UserWarning|_validate_loaded_sparse|sparse_coo_tensor\("
for T in $THRS; do
  O=$G/runs/qa/tables/${KB}_onto_pred${T/./}
  $PY -m experiments.omqa.build_eomqa --data-root $G/data/qa/eomqa/$KB/data_$KB --setting onto --inverse-method matching \
    --extra-triples $D/pred_t${T}.tsv --out $O | tail -5
  ls -la $O
done
