"""Inspect existing trajectories without updating a policy."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np


def analyze(log, env, train_goals, pos_limit, speed_limit):
    episodes = defaultdict(list)
    for line in log.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        episodes[row["seed"]].append(row)
    details = []
    candidates_all = []
    for seed, rows in episodes.items():
        assert [row["step"] for row in rows] == list(range(1, len(rows) + 1))
        assert all(a["next_observation"] == b["observation"] for a, b in zip(rows, rows[1:]))
        assert all(not (row["terminated"] or row["truncated"]) for row in rows[:-1])
        assert rows[-1]["terminated"] or rows[-1]["truncated"]
        states = np.array([row["observation"] for row in rows], dtype=float)
        after = np.array([row["next_observation"] for row in rows], dtype=float)
        failed = np.array([row["outcome"] == "failure" for row in rows])
        low_speed = np.max(np.abs(after[:, 2:4]), axis=1) < speed_limit
        q = after[:, :2]
        candidate_idx = np.flatnonzero(low_speed & ~failed & (q >= env.lower).all(1) & (q <= env.upper).all(1))
        candidates = q[candidate_idx]
        candidates_all.extend(candidates.tolist())
        future = candidate_idx[None, :] >= np.arange(len(rows))[:, None]
        current_at_goal = ((np.abs(states[:, None, :2] - candidates) < pos_limit).all(2)
                           & (np.abs(states[:, None, 2:4]) < speed_limit).all(2))
        valid = future & ~current_at_goal & ~failed[:, None]
        success = valid & low_speed[:, None] & (np.abs(q[:, None, :] - candidates) < pos_limit).all(2)
        future_counts = future.sum(axis=1)
        # Each transition is equally likely. Within it, draw one eligible future goal uniformly.
        probabilities = np.divide(success.sum(axis=1), future_counts,
                                  out=np.zeros(len(rows)), where=future_counts > 0)
        details.append({
            "seed": seed, "steps": len(rows), "original_goal_rad": states[0, 4:6].tolist(),
            "low_speed_steps": int(low_speed.sum()), "candidate_steps": (candidate_idx + 1).tolist(),
            "candidate_positions_rad": candidates.tolist(),
            "future_goal_pairs": int(future.sum()), "valid_pairs": int(valid.sum()),
            "virtual_success_pairs": int(success.sum()),
            "expected_success_fraction_per_relabel_attempt": float(probabilities.mean()),
        })
    all_q = np.asarray(candidates_all).reshape(-1, 2)
    covered = (np.abs(np.asarray(train_goals)[:, None, :] - all_q) < pos_limit).all(2).any(1)
    steps = sum(item["steps"] for item in details)
    return {
        "source": str(log), "steps": steps, "low_speed_steps": sum(i["low_speed_steps"] for i in details),
        "candidate_count": len(all_q),
        "candidate_position_min_rad": all_q.min(0).tolist() if len(all_q) else None,
        "candidate_position_max_rad": all_q.max(0).tolist() if len(all_q) else None,
        "candidate_grid_cells_0_01_rad": len(np.unique(np.floor(all_q / .01), axis=0)),
        "train_goals_within_tolerance": np.asarray(train_goals)[covered].tolist(),
        "virtual_success_pairs": sum(i["virtual_success_pairs"] for i in details),
        "expected_success_fraction_per_relabel_attempt": sum(
            i["steps"] * i["expected_success_fraction_per_relabel_attempt"] for i in details) / steps,
        "episodes": details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("project", "scene", "workspace", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    from panda_reach import TRAIN_GOALS
    from panda_two_joint_env import POSITION_ERROR_THRESHOLD_RAD, SPEED_THRESHOLD_RAD_S, PandaTwoJointEnv

    env = PandaTwoJointEnv(args.scene)
    historical = args.workspace / "panda-sac-50k-20261112-154543"
    audit = args.workspace / "panda-sac-audit-20261112"
    smoke_episodes = [json.loads(line) for line in
                      (audit / "logging-on/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    legacy_episodes = [json.loads(line) for line in
                       (historical / "train/episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {
        "criteria": {"position_error_rad_lt": POSITION_ERROR_THRESHOLD_RAD,
                     "speed_rad_s_lt": SPEED_THRESHOLD_RAD_S},
        "training_50k": {"episode_count": len(legacy_episodes),
                         "has_per_step_trajectory": (historical / "train/steps.jsonl").exists()},
        "training_320_audit": {"steps": sum(row["episode_steps"] for row in smoke_episodes),
                               "low_speed_steps_per_episode": [row["speed_threshold_steps"] for row in smoke_episodes],
                               "full_trajectory_saved": False},
        "validation_50k": {
            controller: analyze(historical / "validation" / f"{controller}.jsonl",
                                env, TRAIN_GOALS, POSITION_ERROR_THRESHOLD_RAD, SPEED_THRESHOLD_RAD_S)
            for controller in ("sac", "random", "position_target", "home_hold")
        },
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for controller, stats in report["validation_50k"].items():
        print(controller, {key: value for key, value in stats.items() if key != "episodes"})
    print(report["training_50k"], report["training_320_audit"])


if __name__ == "__main__":
    main()
