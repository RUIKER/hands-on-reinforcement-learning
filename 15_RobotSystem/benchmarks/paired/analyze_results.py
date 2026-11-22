"""Verify paired Panda runs, aggregate timeouts, and compare held-out goals."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
from statistics import mean, median


RUN = Path(os.environ.get("PANDA_PAIRED_RUN", "runs/panda-paired")).resolve()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def training_stats(root):
    summary = read(root / "summary.json")
    config = read(root / "config.json")
    episodes = rows(root / "episodes.jsonl")
    totals = Counter(row["outcome"] for row in episodes)
    assert summary["status"] == "PASS" and summary["updates"] == config["steps"] - config["warmup"]
    assert summary["reload_action_max_abs_difference"] == 0.0
    assert sum(row["episode_steps"] for row in episodes) == config["steps"]
    assert episodes[-1]["step"] == config["steps"] == summary["steps"]
    if "episode_outcomes" in summary:
        assert totals == summary["episode_outcomes"]
    completed = [row for row in episodes if row["outcome"] != "partial"]
    successes = [row for row in completed if row["outcome"] == "success"]
    timeouts = [row for row in completed if row["outcome"] == "timeout"]
    windows = []
    for lo in range(0, config["steps"], 10_000):
        hi = min(lo + 10_000, config["steps"])
        outcomes = Counter(row["outcome"] for row in completed if lo < row["step"] <= hi)
        windows.append({"step_range": [lo + 1, hi], "outcomes": dict(outcomes)})
    return {
        "outcomes": dict(totals),
        "completed": len(completed),
        "timeout_fraction": len(timeouts) / len(completed),
        "first_success_step": successes[0]["step"] if successes else None,
        "last_timeout_step": timeouts[-1]["step"] if timeouts else None,
        "training_goals_succeeded": len({tuple(row["goal_rad"]) for row in successes}),
        "last_100_completed": dict(Counter(row["outcome"] for row in completed[-100:])),
        "windows_by_episode_end": windows,
        "rollout_update_s": summary["rollout_update_s"],
        "final_alpha": summary.get("final_alpha"),
        "her_replay_counts": summary.get("her"),
    }


def evaluation_stats(root, expected, checkpoint):
    config = read(root / "config.json")
    summary = read(root / "summary.json")
    assert config["cases"] == expected["cases"]
    assert config["split"] == expected["split"] and config["delay_steps"] == expected["delay_steps"]
    assert config["checkpoint_sha256"] == sha(checkpoint)
    case_steps = defaultdict(list)
    for row in rows(root / "sac.jsonl"):
        case_steps[row["seed"]].append(row)
    pos_limit, speed_limit = config["position_error_threshold_rad"], config["speed_threshold_rad_s"]
    cases = []
    for (case_id, goal), episode in zip(expected["cases"], summary["sac"]["per_episode"], strict=True):
        assert episode["seed"] == case_id and episode["goal_rad"] == goal
        trace = case_steps[case_id]
        assert len(trace) == episode["steps"] and trace[-1]["outcome"] == episode["outcome"]
        error = [max(abs(q - g) for q, g in zip(row["next_observation"][:2], goal)) for row in trace]
        speed = [max(map(abs, row["next_observation"][2:4])) for row in trace]
        position_count = sum(value < pos_limit for value in error)
        both_count = sum(e < pos_limit and v < speed_limit for e, v in zip(error, speed))
        outcome = episode["outcome"]
        assert (outcome == "success") == (both_count > 0 and not episode["physics_warning"]
                                           and not episode["actual_limit_violation"])
        timeout_type = None
        if outcome == "timeout":
            timeout_type = "position_only" if position_count else "never_in_position"
        cases.append({
            "case_id": case_id, "goal_rad": goal, "outcome": outcome, "steps": episode["steps"],
            "min_max_abs_error_rad": min(error), "final_max_abs_error_rad": error[-1],
            "final_max_abs_speed_rad_s": speed[-1], "position_steps": position_count,
            "low_speed_steps": sum(value < speed_limit for value in speed), "timeout_type": timeout_type,
        })
    totals = Counter(case["outcome"] for case in cases)
    assert totals["success"] == summary["sac"]["successes"]
    return {
        "outcomes": dict(totals), "cases": cases,
        "baseline_successes": summary["position_target"]["successes"],
        "baseline_mean_success_steps": summary["position_target"]["mean_success_steps"],
        "baseline_trace_sha256": sha(root / "position_target.jsonl"),
        "mean_success_steps": summary["sac"]["mean_success_steps"],
        "median_best_error_rad": median(row["min_max_abs_error_rad"] for row in cases),
        "timeout_types": dict(Counter(row["timeout_type"] for row in cases if row["timeout_type"])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sparse_reference", type=Path)
    parser.add_argument("her_reference", type=Path)
    args = parser.parse_args()
    protocol_path = RUN / "protocol.json"
    protocol = read(protocol_path)
    status = rows(RUN / "status.jsonl")
    assert status[0]["protocol_sha256"] == sha(protocol_path)
    assert status[0]["time_utc"] > protocol["frozen_at_utc"]
    assert status[-1]["stage"] == "job" and status[-1]["status"] == "complete"
    train_ref = read(args.sparse_reference / "train/config.json")
    checkpoint_refs = {"sac": args.sparse_reference, "sac_her": args.her_reference}
    reference, paired = {}, {}
    source_digests = []
    baseline_digests = []
    for seed in (42, *protocol["training_seeds"]):
        pair = {}
        warmup_prefixes = []
        for arm in protocol["arms"]:
            root = RUN / ("reference-42" if seed == 42 else f"seed-{seed}") / arm
            train_root = (checkpoint_refs[arm] if seed == 42 else root) / "train"
            config = read(train_root / "config.json")
            assert config["seed"] == seed and config["action_rng_seed"] == seed + 1
            for key, value in train_ref.items():
                if key not in {"seed", "action_rng_seed"}:
                    assert config[key] == value, (seed, arm, key)
            if seed != 42:
                assert config["her"]["enabled"] == (arm == "sac_her")
                source_digests.append(config["source_sha256"])
                with (train_root / "steps.jsonl").open("rb") as stream:
                    warmup_prefixes.append(list(islice(stream, config["warmup"])))
            training = training_stats(train_root)
            evaluation = evaluation_stats(root / "heldout", protocol["evaluation"], train_root / "sac_cpu.pt")
            baseline_digests.append(evaluation["baseline_trace_sha256"])
            pair[arm] = {"training": training, "evaluation": evaluation}
        if seed == 42:
            reference = pair
        else:
            assert len(warmup_prefixes[0]) == train_ref["warmup"]
            assert warmup_prefixes[0] == warmup_prefixes[1], seed
            paired[str(seed)] = pair
    assert all(entry == source_digests[0] for entry in source_digests)
    assert len(set(baseline_digests)) == 1
    totals = {}
    for arm in protocol["arms"]:
        arm_results = [pair[arm] for pair in paired.values()]
        train_outcomes = Counter()
        late_train_outcomes = Counter()
        eval_outcomes = Counter()
        eval_timeout_types = Counter()
        for result in arm_results:
            train_outcomes.update(result["training"]["outcomes"])
            late_train_outcomes.update(result["training"]["windows_by_episode_end"][-1]["outcomes"])
            eval_outcomes.update(result["evaluation"]["outcomes"])
            eval_timeout_types.update(result["evaluation"]["timeout_types"])
        totals[arm] = {
            "training": dict(train_outcomes), "training_last_10000_steps": dict(late_train_outcomes),
            "evaluation": dict(eval_outcomes), "evaluation_timeout_types": dict(eval_timeout_types),
            "evaluation_macro_success_fraction": mean(
                result["evaluation"]["outcomes"].get("success", 0) / len(protocol["evaluation"]["cases"])
                for result in arm_results
            ),
        }
    report = {"protocol_sha256": sha(protocol_path), "matching_training_parameters": True,
              "paired_warmup_trajectories_byte_identical": True,
              "position_baselines_byte_identical": True, "reference_seed42": reference,
              "paired_runs": paired, "three_seed_totals": totals}
    with (RUN / "results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    for seed, pair in [("reference-42", reference), *paired.items()]:
        print(seed)
        for arm, result in pair.items():
            train, test = result["training"], result["evaluation"]
            print(arm, "train", train["outcomes"], "late", train["windows_by_episode_end"][-1]["outcomes"],
                  "test", test["outcomes"], "test timeout types", test["timeout_types"])
    print(json.dumps(totals))


if __name__ == "__main__":
    main()
