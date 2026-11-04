#!/usr/bin/env bash
# Parallel synthetic sweep for GroundedEL Experiment E1 (design §9 E1).
#
# Full grid:
#   theories in {tk3, tk4, tk5, tk6, tk7, box2el_t1, box2el_c4,
#                role_functional, role_cycle, role_chain, unsat_mix}
#   K in {1, 2, 4, 8, full} (with Kr = K)
#   dim in {2, 10}
#   seed in {0, 1, 2}
#   epochs = 3000, lr = 0.01
#
# Runs up to 12 cells in parallel with `xargs -P 12`, writing cell JSONs to
# `/mnt/data1/gfe/runs/grounded-e1/` and per-cell logs to
# `/mnt/data1/gfe/logs/grounded-e1/`. Skips cells whose output JSON already
# exists on disk.
#
# Environment overrides (for subset testing):
#   THEORIES, KS, DIMS, SEEDS, CELLS, EPOCHS, LR, JOBS, OUT_DIR, LOG_DIR, PYTHON

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
if [[ "${TMPDIR}" == "/tmp" || "${TMPDIR}" == /tmp/* ]]; then
    export TMPDIR="$HOME/tmp"
fi
mkdir -p "${TMPDIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

PYTHON="${PYTHON:-$HOME/.venvs/mowl-ws/bin/python}"
OUT_DIR="${OUT_DIR:-/mnt/data1/gfe/runs/grounded-e1}"
LOG_DIR="${LOG_DIR:-/mnt/data1/gfe/logs/grounded-e1}"
EPOCHS="${EPOCHS:-3000}"
LR="${LR:-0.01}"
JOBS="${JOBS:-12}"

THEORIES="${THEORIES:-tk3 tk4 tk5 tk6 tk7 box2el_t1 box2el_c4 role_functional role_cycle role_chain unsat_mix}"
KS="${KS:-1 2 4 8 full}"
DIMS="${DIMS:-2 10}"
SEEDS="${SEEDS:-0 1 2}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

# Pre-generate all theory OWL files once before launching parallel workers.
"${PYTHON}" -m experiments.grounded.theories --no-verify >/dev/null

emit_cells() {
    if [[ -n "${CELLS:-}" ]]; then
        printf '%s\n' "${CELLS}" | awk 'NF == 4 { print $1, $2, $3, $4 }'
        return
    fi
    local th k d s
    for th in ${THEORIES}; do
        for k in ${KS}; do
            for d in ${DIMS}; do
                for s in ${SEEDS}; do
                    printf '%s %s %s %s\n' "${th}" "${k}" "${d}" "${s}"
                done
            done
        done
    done
}

filter_missing_cells() {
    local th k d s json_path
    while read -r th k d s; do
        [[ -z "${th}" ]] && continue
        json_path="${OUT_DIR}/${th}__K${k}__d${d}__s${s}.json"
        if [[ -s "${json_path}" ]]; then
            continue
        fi
        printf '%s %s %s %s\n' "${th}" "${k}" "${d}" "${s}"
    done
}

export REPO_ROOT PYTHON OUT_DIR LOG_DIR EPOCHS LR TMPDIR OMP_NUM_THREADS MKL_NUM_THREADS

run_one_cell() {
    local th="$1"
    local k="$2"
    local d="$3"
    local s="$4"
    local json_path="${OUT_DIR}/${th}__K${k}__d${d}__s${s}.json"
    local log_path="${LOG_DIR}/${th}__K${k}__d${d}__s${s}.log"
    if [[ -s "${json_path}" ]]; then
        return 0
    fi
    "${PYTHON}" -m experiments.grounded.run_synthetic \
        --theory "${th}" \
        --K "${k}" \
        --Kr "${k}" \
        --dim "${d}" \
        --seed "${s}" \
        --epochs "${EPOCHS}" \
        --lr "${LR}" \
        --out "${OUT_DIR}" \
        > "${log_path}" 2>&1
}
export -f run_one_cell

emit_cells | filter_missing_cells | xargs -r -P "${JOBS}" -n 4 bash -c 'run_one_cell "$1" "$2" "$3" "$4"' _
