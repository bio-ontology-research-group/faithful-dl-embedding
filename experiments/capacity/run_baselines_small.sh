#!/usr/bin/env bash
# B3 baselines for the small datasets (pizza + synthetic theories) on CPU.
# Splits the non-GO/FoodOn cells of baselines_cells.txt into WORKERS interleaved
# slices and runs one process per slice (cells whose JSON exists are skipped).
# Start detached:  nohup setsid experiments/capacity/run_baselines_small.sh < /dev/null &
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$HOME/.venvs/mowl-ws/bin/python}"
MOWL_DIR="${MOWL_DIR:-$HOME/Public/software/mowl-wt/abox}"
WORKERS="${WORKERS:-4}"
OUT="${OUT:-$REPO/experiments/results/raw-capacity/baselines}"
LOGS="$REPO/experiments/results/raw-capacity/logs"
CKPT="${CKPT:-$HOME/tmp/gfe-runs/capacity}"
export PYTHONPATH="$MOWL_DIR:$REPO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
mkdir -p "$OUT" "$LOGS" "$CKPT"
cd "$REPO"
ulimit -v 40000000
PIDS=()
for ((w = 0; w < WORKERS; w++)); do
    SLICE="$LOGS/baselines_small_cells_$w.txt"
    grep -v '^\(go\|foodon\) ' experiments/capacity/baselines_cells.txt \
        | awk -v w="$w" -v n="$WORKERS" 'NF == 4 && (NR - 1) % n == w' > "$SLICE"
    "$PYTHON" -m experiments.capacity.baselines --cells "$SLICE" --skip-existing --device cpu \
        --out "$OUT" --runs-dir "$CKPT" > "$LOGS/baselines_small_$w.log" 2>&1 &
    PIDS+=("$!")
done
STATUS=0
for p in "${PIDS[@]}"; do wait "$p" || STATUS=1; done
echo "[DONE] status=$STATUS" >> "$LOGS/baselines_small_0.log"
exit "$STATUS"
