#!/usr/bin/env bash
#SBATCH --job-name=gfe-deepgose-profile
#SBATCH --partition=pi-hohndor
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_profile_%j.out
#SBATCH --error=/mnt/data/u/hohndor/gfe/logs/deepgose/e3_profile_%j.err
# Profile the GroundedEL TBox step on a GPU and sample nvidia-smi utilisation meanwhile.
set -uo pipefail
PYTHON="${PYTHON:-/mnt/data/u/hohndor/gfe/venv/bin/python}"
MOWL_DIR="${MOWL_DIR:-/mnt/data/u/hohndor/gfe/mowl}"
REPO_DIR="${REPO_DIR:-/mnt/data/u/hohndor/gfe/faithful-dl-embedding}"
DEEPGO2_DIR="${DEEPGO2_DIR:-/mnt/data/u/hohndor/gfe/deepgo2}"
OUT_DIR="${OUT_DIR:-/mnt/data/u/hohndor/gfe/runs/deepgose}"
export TMPDIR="${TMPDIR:-/mnt/data/u/hohndor/gfe/tmp}"
export DEEPGO2_DIR PYTHONPATH="${MOWL_DIR}:${REPO_DIR}:${DEEPGO2_DIR}:${PYTHONPATH:-}"
cd "${REPO_DIR}"
ONT="${ONT:-cc}"; TBOX_BATCH="${TBOX_BATCH:-1024}"; TAG="${TAG:-base}"
nvidia-smi dmon -s u -d 2 > "${OUT_DIR}/profile_${ONT}_${TAG}_dmon.txt" 2>&1 &
DMON=$!
"${PYTHON}" -m experiments.deepgose.profile_tbox --ont "${ONT}" --tbox-batch "${TBOX_BATCH}" --device cuda \
    --cache-dir "${OUT_DIR}/${ONT}" --backward --out "${OUT_DIR}/profile_${ONT}_${TAG}.json" ${EXTRA_ARGS:-}
status=$?
kill ${DMON} 2>/dev/null
echo "[dmon] mean sm% over samples (excluding load phase = last 60 samples):"
grep -v '^#' "${OUT_DIR}/profile_${ONT}_${TAG}_dmon.txt" | tail -60 | awk '{s+=$2; n++} END {if (n) printf "%.1f%% over %d samples\n", s/n, n}'
exit ${status}
