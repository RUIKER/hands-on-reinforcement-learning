#!/usr/bin/env bash
# 共享批任务控制：使用当前虚拟环境，限制并发，并保留每个子进程的退出状态。
# 在设置 JOBS 后 source 本文件；用 submit_job 提交，用 wait_all 收尾。
PY="${PY:-python}"
JOBS="${JOBS:-3}"
if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "JOBS must be a positive integer" >&2
    exit 2
fi
command -v "$PY" >/dev/null || { echo "Python executable not found: $PY" >&2; exit 2; }
_runner_pids=()
_runner_failed=0

_wait_oldest() {
    local pid="${_runner_pids[0]}"
    if ! wait "$pid"; then
        echo "[FAIL] worker $pid; inspect results/logs/" >&2
        _runner_failed=1
    fi
    _runner_pids=("${_runner_pids[@]:1}")
}

submit_job() {
    if [ "${#_runner_pids[@]}" -ge "$JOBS" ]; then
        _wait_oldest
    fi
    "$@" &
    _runner_pids+=("$!")
}

wait_all() {
    while [ "${#_runner_pids[@]}" -gt 0 ]; do
        _wait_oldest
    done
    return "$_runner_failed"
}
