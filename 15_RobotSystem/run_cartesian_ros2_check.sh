#!/usr/bin/env bash
# Build the ROS interfaces and evaluate the 24 + 48 frozen goal coordinates.
set -eo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s SCENE_XML NEW_OUTPUT_DIR [PYTHON]\n' "$0" >&2
  exit 2
fi

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scene="$(realpath -e -- "$1")"
work="$(realpath -m -- "$2")"
python="$(realpath -s -- "$(command -v "${3:-python3}")")"
source "${ROS_SETUP:-/opt/ros/lyrical/setup.bash}"

mkdir -p -- "$(dirname -- "$work")"
mkdir -- "$work"
(
  cd -- "$work"
  colcon build --base-paths "$project/ros2_ws/src/panda_policy_interfaces" \
    --packages-select panda_policy_interfaces --cmake-args "-DPython3_EXECUTABLE=${ROS_PYTHON:-/usr/bin/python3}"
)
source "$work/install/local_setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-85}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

"$python" "$project/panda_cartesian_ros2.py" serve "$scene" >"$work/server.log" 2>&1 &
server=$!
cleanup() {
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
}
trap cleanup EXIT
timeout 120s "$python" "$project/panda_cartesian_ros2.py" evaluate "$scene" \
  "$project/results/20261122_cartesian_reach" "$work/evidence"
