#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
JOBS="${JOBS:-3}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
source ../../_runner.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUTF8=1
mkdir -p results/raw results/logs

run_one() {
    local seed="$1" her="$2"
    local tag
    if [ "$her" = "1" ]; then
        tag="her_r0.8_future_s${seed}"
    else
        tag="noher_s${seed}"
    fi
    "$PY" run_one.py --seed "$seed" --her "$her" \
        --out "results/raw/${tag}.json" > "results/logs/${tag}.log" 2>&1 || return "$?"
}
export -f run_one
export PY

for seed in $SEEDS; do
    for her in 1 0; do
        submit_job run_one "$seed" "$her"
    done
done
wait_all
echo "ALL DONE"
