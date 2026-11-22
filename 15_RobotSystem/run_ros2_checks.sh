#!/usr/bin/env bash
set -eo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s SCENE_XML NEW_OUTPUT_DIR [PYTHON]\n' "$0" >&2
  exit 2
fi

project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
scene="$(realpath -e -- "$1")"
output="$(realpath -m -- "$2")"
python="$(command -v "${3:-python3}")"
ros_setup="${ROS_SETUP:-/opt/ros/lyrical/setup.bash}"
evidence="$project/results/20261119_panda_policy"

source "$ros_setup"
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
build="$output/build"
mkdir -- "$build"
(
  cd -- "$build"
  colcon build --base-paths "$project/ros2_ws/src" \
    --packages-select panda_policy_interfaces panda_sim panda_trajectory_cmd
)
source "$build/install/local_setup.bash"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-120}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
"$python" "$project/panda_ros2_policy.py" serve "$scene" >"$output/server.log" 2>&1 &
server=$!
cleanup() {
  kill "$server" 2>/dev/null || true
  wait "$server" 2>/dev/null || true
}
trap cleanup EXIT

timeout 40s "$python" "$project/panda_ros2_policy.py" evaluate "$scene" \
  "$evidence/actor.npz" "$evidence/protocol.json" "$output/replay" \
  --case 30104 --case 30124
