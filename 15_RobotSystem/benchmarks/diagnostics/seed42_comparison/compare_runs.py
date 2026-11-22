"""Compare two Panda SAC runs and reconstruct HER sampling from complete rollout logs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def coverage(positions, goals, threshold):
    if not len(positions):
        return {"samples": 0, "within_any_training_goal": 0, "per_training_goal": [0] * len(goals)}
    near = (np.abs(positions[:, None, :] - goals) < threshold).all(axis=2)
    return {"samples": len(positions), "within_any_training_goal": int(near.any(axis=1).sum()),
            "per_training_goal": near.sum(axis=0).tolist(),
            "position_min_rad": positions.min(axis=0).tolist(),
            "position_max_rad": positions.max(axis=0).tolist()}


def trajectory_stats(rows, position_limit, speed_limit):
    after = np.array([row["next_observation"] for row in rows])
    error = np.max(np.abs(after[:, :2] - after[:, 4:6]), axis=1)
    speed = np.max(np.abs(after[:, 2:4]), axis=1)
    reached, slow = error < position_limit, speed < speed_limit
    best = int(error.argmin())
    return {
        "steps": len(rows), "first_action": rows[0]["action_normalized"], "best_step": rows[best]["step"],
        "best_max_abs_error_rad": float(error[best]), "speed_at_best_rad_s": float(speed[best]),
        "final_q_rad": after[-1, :2].tolist(), "final_abs_error_rad": np.abs(after[-1, :2] - after[-1, 4:6]).tolist(),
        "final_abs_speed_rad_s": np.abs(after[-1, 2:4]).tolist(), "position_steps": int(reached.sum()),
        "low_speed_steps": int(slow.sum()), "both_steps": int((reached & slow).sum()),
        "command_clip_steps": sum(row["command_clipped"] for row in rows),
        "warnings": sum(row["physics_warning"] for row in rows),
        "limit_violations": sum(row["actual_limit_violation"] for row in rows),
        "outcome": rows[-1]["outcome"],
    }


def reconstruction(rows, config, episode_rows, expected, env, goals, position_limit, speed_limit):
    n = len(rows)
    assert n == config["steps"] <= config["replay_capacity"]
    assert [row["step"] for row in rows] == list(range(1, n + 1))
    before = np.array([row["observation"] for row in rows])
    after = np.array([row["next_observation"] for row in rows])
    failed = np.array([row["outcome"] == "failure" for row in rows])
    slow_before = (np.abs(before[:, 2:4]) < speed_limit).all(axis=1)
    slow_after = (np.abs(after[:, 2:4]) < speed_limit).all(axis=1)
    position = after[:, :2]
    within = (position >= env.lower).all(axis=1) & (position <= env.upper).all(axis=1)
    candidates = np.flatnonzero(slow_after & within & ~failed)
    reached = (np.abs(position - after[:, 4:6]) < position_limit).all(axis=1)
    episodes = np.array([row["episode"] for row in rows])
    ep_end_by_step = np.array([row["step"] - 1 for row in episode_rows])[episodes - 1]
    first_candidate = np.searchsorted(candidates, np.arange(n))
    episode_end_candidate = np.searchsorted(candidates, ep_end_by_step, side="right")
    observed_end_candidate = np.searchsorted(candidates, np.arange(n), side="right")

    rng = random.Random(config["seed"])
    goal_rng = random.Random(config["her"]["goal_rng_seed"])
    batch = config["batch"]
    attempts = batch // 2
    successful_candidate_indices = []
    successful_original_goals = []
    counts = Counter(dict.fromkeys(expected, 0))
    by_window = {}
    for step in range(config["warmup"], n):
        sources = np.array(rng.sample(range(step + 1), batch)[:attempts])
        counts["attempts"] += attempts
        end = np.minimum(episode_end_candidate[sources], observed_end_candidate[step])
        first = first_candidate[sources]
        eligible = (first < end) & ~failed[sources]
        counts["source_failed"] += int(failed[sources].sum())
        counts["no_future_goal"] += int((~eligible & ~failed[sources]).sum())
        source = sources[eligible]
        goal_index = candidates[[goal_rng.randrange(int(lo), int(hi)) for lo, hi in zip(first[eligible], end[eligible])]]
        virtual_goal = position[goal_index]
        already = slow_before[source] & (np.abs(before[source, :2] - virtual_goal) < position_limit).all(axis=1)
        success = ~already & slow_after[source] & (np.abs(position[source] - virtual_goal) < position_limit).all(axis=1)
        relabeled = int((~already).sum())
        counts["already_at_goal"] += int(already.sum())
        counts["relabeled"] += relabeled
        counts["relabeled_success"] += int(success.sum())
        counts["original"] += batch - relabeled
        successful_candidate_indices.extend(goal_index[success])
        successful_original_goals.extend(before[source[success], 4:6])
        window = by_window.setdefault((step // 10_000 + 1) * 10_000, Counter())
        window.update(attempts=attempts, relabeled=relabeled, relabeled_success=int(success.sum()))
    assert dict(counts) == expected, (dict(counts), expected)

    success_positions = position[np.asarray(successful_candidate_indices, dtype=int)]
    matching_original = (np.abs(success_positions - np.asarray(successful_original_goals).reshape(-1, 2))
                         < position_limit).all(axis=1)
    windows = []
    for start in range(0, n, 10_000):
        end = min(start + 10_000, n)
        span = slice(start, end)
        windows.append({
            "steps": [start + 1, end], "real_position_steps": int(reached[span].sum()),
            "real_success_steps": int((reached[span] & slow_after[span] & ~failed[span]).sum()),
            "low_speed_steps": int(slow_after[span].sum()),
            "candidate_goal_coverage": coverage(position[candidates[(candidates >= start) & (candidates < end)]],
                                                goals, position_limit),
            "replay": by_window.get(end, {}),
        })
    return {
        "counts_match_training_summary": True, "counts": dict(counts),
        "actual_training_position_steps": int(reached.sum()),
        "actual_training_low_speed_steps": int(slow_after.sum()),
        "eligible_positions": coverage(position[candidates], goals, position_limit),
        "successful_virtual_targets": coverage(success_positions, goals, position_limit),
        "successful_virtual_targets_near_original_goal": int(matching_original.sum()),
        "windows": windows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("run", "old_run", "project", "scene"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    from panda_two_joint_env import PandaTwoJointEnv
    env = PandaTwoJointEnv(args.scene)
    runs = {"sac": args.old_run, "sac_her": args.run}
    train_configs = {key: read_json(path / "train/config.json") for key, path in runs.items()}
    validation_configs = {key: read_json(path / "validation/config.json") for key, path in runs.items()}
    for key, value in train_configs["sac"].items():
        assert train_configs["sac_her"][key] == value, key
    for key, value in validation_configs["sac"].items():
        if key not in ("checkpoint", "checkpoint_sha256"):
            assert validation_configs["sac_her"][key] == value, key
    config = train_configs["sac_her"]
    pos_limit, speed_limit = config["position_error_threshold_rad"], config["speed_threshold_rad_s"]
    goals = np.asarray(config["goals_rad"])
    report = {
        "matching_previous_training_config": True, "matching_validation_protocol": True,
        "train_goals_rad": goals.tolist(), "validation": {},
    }
    for name, root in runs.items():
        episodes = read_jsonl(root / "train/episodes.jsonl")
        summary = read_json(root / "validation/summary.json")
        rows = read_jsonl(root / "validation/sac.jsonl")
        report["validation"][name] = {
            "outcomes_training": dict(Counter(row["outcome"] for row in episodes)),
            "train_goal_count": len({tuple(row["goal_rad"]) for row in episodes}),
            "successes": summary["sac"]["successes"], "baseline_successes": summary["position_target"]["successes"],
            "cases": [{"seed": seed, "goal_rad": goal, **trajectory_stats(
                [row for row in rows if row["seed"] == seed], pos_limit, speed_limit
            )} for seed, goal in validation_configs[name]["cases"]],
        }
    summary = read_json(args.run / "train/summary.json")
    report["her_replay"] = reconstruction(read_jsonl(args.run / "train/steps.jsonl"), config,
                                          read_jsonl(args.run / "train/episodes.jsonl"), summary["her"],
                                          env, goals, pos_limit, speed_limit)
    updates = read_jsonl(args.run / "train/updates.jsonl")
    report["update_telemetry"] = {
        "alpha_initial": updates[0]["alpha"], "alpha_min": min(row["alpha"] for row in updates),
        "alpha_max": max(row["alpha"] for row in updates), "alpha_final": updates[-1]["alpha"],
        "entropy_final": updates[-1]["actor_entropy_estimate"],
        "max_sampled_critic_loss": max(max(row["critic_1_loss"], row["critic_2_loss"]) for row in updates),
        "last_update": updates[-1],
    }
    with (args.run / "comparison.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
