"""Compare seed 43's actor with a position controller handoff at a stall."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from statistics import median
import sys

import numpy as np


def grouped_trace(path):
    grouped = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[row["seed"]].append(row)
    return grouped


def diagnose(traces):
    stats = []
    for seed, rows in traces.items():
        if rows[-1]["outcome"] != "timeout":
            continue
        after = np.array([row["next_observation"] for row in rows])
        q, goal, ctrl = after[-1, :2], after[-1, 4:6], after[-1, 6:8]
        axis = int(np.argmax(np.abs(goal - q)))
        assert not any(row["command_clipped"] or row["physics_warning"] or row["actual_limit_violation"]
                       for row in rows)
        stats.append({
            "case_id": seed, "goal_rad": goal.tolist(), "limiting_axis": axis + 1,
            "best_error_rad": float(np.max(np.abs(after[:, :2] - goal), axis=1).min()),
            "goal_minus_q_rad": (goal - q).tolist(),
            "goal_minus_ctrl_rad": (goal - ctrl).tolist(),
            "ctrl_minus_q_rad": (ctrl - q).tolist(),
            "final_max_speed_rad_s": float(np.max(np.abs(after[-1, 2:4]))),
            "last_50_ctrl_peak_to_peak_rad": np.ptp(after[-50:, 6:8], axis=0).tolist(),
            "last_50_max_abs_action": float(np.max(np.abs([row["action_normalized"] for row in rows[-50:]]))),
        })
    return stats


def fresh_cases(previous, forbidden, env, seed):
    used = [tuple(round(angle * 1000) for angle in goal) for goal in forbidden]
    used.extend(tuple(round(angle * 1000) for angle in goal) for _, goal in previous)
    rng = random.Random(seed)
    cases = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for _ in range(6):
            for _ in range(10000):
                x, y = sx * rng.randint(40, 160), sy * rng.randint(40, 160)
                if all(max(abs(x - px), abs(y - py)) >= 21 for px, py in used):
                    break
            else:
                raise RuntimeError("Could not construct a new goal set")
            used.append((x, y))
            goal = [x / 1000, y / 1000]
            env.validate_goal(goal)
            cases.append([30201 + len(cases), goal])
    return cases


class HandoffPolicy:
    def __init__(self, actor, position_controller, guard):
        self.actor = actor
        self.position_controller = position_controller
        self.guard = guard
        self.stalled_steps = 0
        self.step = 0
        self.handoff_step = None

    def __call__(self, observation):
        self.step += 1
        if self.handoff_step is not None:
            return self.position_controller(observation)
        action = self.actor(observation)
        stalled = (np.max(np.abs(observation[4:6] - observation[:2])) < self.guard["near_goal_rad"]
                   and np.max(np.abs(observation[2:4])) < self.guard["speed_rad_s"]
                   and np.max(np.abs(action)) < self.guard["normalized_action_abs"])
        self.stalled_steps = self.stalled_steps + 1 if stalled else 0
        if self.stalled_steps >= self.guard["consecutive_steps"]:
            self.handoff_step = self.step
            return self.position_controller(observation)
        return action


def summarize_trace(episode, rows, switch):
    after = np.array([row["next_observation"] for row in rows])
    errors = np.max(np.abs(after[:, :2] - after[:, 4:6]), axis=1)
    return {**episode, "handoff_step": switch, "best_max_axis_error_rad": float(errors.min()),
            "final_max_axis_error_rad": float(errors[-1])}


def evaluate_set(reach, env, actor, guard, cases, root, historical=None):
    root.mkdir()
    by_arm = {}
    for arm in ("actor", "handoff"):
        log_path = root / f"{arm}.jsonl"
        episodes = {}
        with log_path.open("x", encoding="utf-8") as log:
            for case, goal in cases:
                controller = HandoffPolicy(actor, reach.command_controller, guard)
                policy = actor if arm == "actor" else controller
                summary = reach.run_episode(env, policy, goal, case, 0, log)
                episodes[case] = (summary, controller.handoff_step)
        traces = grouped_trace(log_path)
        by_arm[arm] = (traces, {
            case: summarize_trace(episode, traces[case], switch)
            for case, (episode, switch) in episodes.items()
        })
    actor_traces, base = by_arm["actor"]
    handoff_traces, changed = by_arm["handoff"]
    if historical is not None:
        assert actor_traces == historical
    for case in base:
        switch = changed[case]["handoff_step"]
        if switch is None:
            assert actor_traces[case] == handoff_traces[case]
        else:
            assert actor_traces[case][:switch - 1] == handoff_traces[case][:switch - 1]
    timeout_ids = [case for case, stats in base.items() if stats["outcome"] == "timeout"]
    return {
        "count": len(cases),
        "actor_successes": sum(s["outcome"] == "success" for s in base.values()),
        "handoff_successes": sum(s["outcome"] == "success" for s in changed.values()),
        "recovered_timeouts": sum(changed[c]["outcome"] == "success" for c in timeout_ids),
        "regressions": sum(base[c]["outcome"] == "success" and changed[c]["outcome"] != "success" for c in base),
        "handoff_count": sum(s["handoff_step"] is not None for s in changed.values()),
        "original_timeout_median_best_error_rad": (
            median(base[c]["best_max_axis_error_rad"] for c in timeout_ids) if timeout_ids else None),
        "handoff_on_same_cases_median_best_error_rad": (
            median(changed[c]["best_max_axis_error_rad"] for c in timeout_ids) if timeout_ids else None),
        "cases": [{"case_id": case, "actor": base[case], "handoff": changed[case]} for case in base],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "scene", "prior"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project))
    import torch
    import panda_reach as reach
    from panda_sac_smoke import reload_agent

    out = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/precision")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    prior_config = json.loads((args.prior / "heldout/config.json").read_text(encoding="utf-8"))
    history = grouped_trace(args.prior / "heldout/sac.jsonl")
    original_cases = prior_config["cases"]
    original_diagnosis = diagnose(history)
    assert len(original_diagnosis) == 10
    reach.write_json(out / "stall-diagnosis.json", original_diagnosis)

    env = reach.PandaTwoJointEnv(args.scene, max_steps=reach.MAX_STEPS)
    guard = {"near_goal_rad": 3 * reach.POSITION_ERROR_THRESHOLD_RAD,
             "speed_rad_s": reach.SPEED_THRESHOLD_RAD_S,
             "normalized_action_abs": .01, "consecutive_steps": 3}
    goal_seed = 6111202
    previous = [goal for cases in reach.EVAL_CASES.values() for _, goal in cases]
    new_cases = fresh_cases(original_cases, [*reach.TRAIN_GOALS, *previous], env, goal_seed)
    checkpoint = args.prior / "train/sac_cpu.pt"
    digest = sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == prior_config["checkpoint_sha256"]
    model = reach.model_info(args.scene)
    assert all(model[key] == prior_config[key] for key in model)
    protocol = {
        **model, "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": digest,
        "train_seed": 43, "diagnostic_cases": original_cases, "fresh_goal_seed": goal_seed,
        "fresh_cases": new_cases, "min_goal_separation_rad": .021, "guard": guard,
        "position_error_threshold_rad": reach.POSITION_ERROR_THRESHOLD_RAD,
        "speed_threshold_rad_s": reach.SPEED_THRESHOLD_RAD_S, "delay_steps": 0,
    }
    reach.write_json(out / "protocol.json", protocol)
    agent = reload_agent(checkpoint)

    def actor(observation):
        return reach.deterministic_action(agent, observation)

    original = evaluate_set(reach, env, actor, guard, original_cases, out / "regression", historical=history)
    fresh = evaluate_set(reach, env, actor, guard, new_cases, out / "fresh")
    with (out / "demonstrations.jsonl").open("x", encoding="utf-8") as log:
        demos = [reach.run_episode(env, reach.command_controller, goal, 40001 + i, 0, log)
                 for i, goal in enumerate(reach.TRAIN_GOALS)]
    report = {
        "historic_actor_reproduced": True, "trajectories_match_before_switch": True,
        "regression": original, "fresh": fresh, "early_training_demo_feasibility": {
            "successes": sum(row["outcome"] == "success" for row in demos),
            "goals": len(demos), "steps": sum(row["steps"] for row in demos),
            "cases": demos,
        },
    }
    reach.write_json(out / "comparison.json", report)
    print(json.dumps({key: {k: v for k, v in val.items() if k != "cases"}
                      for key, val in report.items() if isinstance(val, dict)}, indent=2))


if __name__ == "__main__":
    main()
