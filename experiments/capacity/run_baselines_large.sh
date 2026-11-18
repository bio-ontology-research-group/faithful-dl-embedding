#!/usr/bin/env bash
# B3 baselines for GO and FoodOn on CPU (fallback when no GPU cluster is available).
# Splits baselines_cells_large.txt into WORKERS interleaved slices, one process per
# slice; cells whose JSON exists are skipped, so the script can be restarted.
# Start detached:  nohup setsid experiments/capacity/run_baselines_large.sh < /dev/null &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/.venvs/mowl-ws/bin/python}"
MOWL_DIR="${MOWL_DIR:-$HOME/Public/software/mowl-wt/abox}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-3000}"
OUT="${OUT:-$REPO/experiments/results/raw-capacity/baselines}"
LOGS="$REPO/experiments/results/raw-capacity/logs"
CKPT="${CKPT:-$HOME/tmp/gfe-runs/capacity}"
export PYTHONPATH="$MOWL_DIR:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$OUT" "$LOGS" "$CKPT"
cd "$REPO"
ulimit -v "${VMEM_KB:-80000000}"
PIDS=()
for ((w = 0; w < WORKERS; w++)); do
    SLICE="$LOGS/baselines_large_cells_$w.txt"
    awk -v w="$w" -v n="$WORKERS" 'NF == 4 && $1 !~ /^#/ {if (i++ % n == w) print}' \
        experiments/capacity/baselines_cells_large.txt > "$SLICE"
    "$PYTHON" -m experiments.capacity.baselines --cells "$SLICE" --skip-existing --device cpu \
        --epochs "$EPOCHS" --out "$OUT" --runs-dir "$CKPT" > "$LOGS/baselines_large_$w.log" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
echo "[DONE] status=$STATUS" >> "$LOGS/baselines_large_0.log"
exit "$STATUS"
