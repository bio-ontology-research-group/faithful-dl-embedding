#!/usr/bin/env bash
# Run the retrieval-benchmark pipeline for ONE knowledge base in the foreground.
#
# Usage:  ./run_pipeline.sh <kb> [extra ground_truth.py options, e.g. --budget-minutes 90]
# Inputs: $GFE_RETRIEVAL_DATA/kbs/<kb>/<kb>.owl (default data dir ~/tmp/gfe/retrieval).
# Outputs: <kb>_el.owl (if missing), queries.json, ground_truth.json, predictions_abox.json,
#          predictions_told.json in the KB directory; a log under $GFE_RETRIEVAL_DATA/logs/.
# Limits: virtual memory 40,000,000 KiB (ulimit -v); JVM heap from common.choose_heap_gb
#         (<= 20 GiB and <= MemAvailable - 4 GiB, so the process stays under 24 GB); glibc arenas capped.
# Wrap the call in `timeout` to bound wall time, e.g. `timeout 3h ./run_pipeline.sh mutagenesis`.
set -euo pipefail
kb=${1:?usage: run_pipeline.sh <kb> [ground_truth options]}
shift
export TMPDIR=${TMPDIR:-$HOME/tmp}
export MALLOC_ARENA_MAX=4
data=${GFE_RETRIEVAL_DATA:-$HOME/tmp/gfe/retrieval}
py=${PYTHON:-$HOME/.venvs/mowl-ws/bin/python}
cd "$(dirname "$0")"
ulimit -v 40000000
avail=$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "== $kb $(date +%FT%T) MemAvailable=${avail}GiB"
if [ "$avail" -lt 6 ]; then echo "less than 6 GiB available; not starting" >&2; exit 1; fi
mkdir -p "$data/logs"
log="$data/logs/pipeline_${kb}.log"
{
  [ -s "$data/kbs/$kb/${kb}_el.owl" ] || "$py" make_el_projection.py --kb "$kb" --data-dir "$data"
  "$py" sample_queries.py --kb "$kb" --data-dir "$data"
  "$py" ground_truth.py --kb "$kb" --data-dir "$data" "$@"
  "$py" abox_lookup.py --kb "$kb" --data-dir "$data"
  "$py" abox_lookup.py --kb "$kb" --data-dir "$data" --told-hierarchy
} 2>&1 | tee "$log"
echo "== $kb done $(date +%FT%T)"
