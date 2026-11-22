"""Separate demonstration, mixed, and actor-only episodes in the paired run."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from statistics import median


RUN = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/warmup_seed43")).resolve()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def stream(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def summarize_training(root, protocol, demonstration_steps):
    config, summary = (load(root / name) for name in ("config.json", "summary.json"))
    assert config["seed"] == protocol["train_seed"]
    assert config["steps"] == summary["steps"] == protocol["steps"]
    assert config["warmup"] == protocol["warmup"]
    assert summary["updates"] == config["steps"] - config["warmup"]
    assert summary["status"] == "PASS" and summary["reload_action_max_abs_difference"] == 0.0
    expected_sources = Counter(demonstration=demonstration_steps,
                               warmup=config["warmup"] - demonstration_steps,
                               actor=config["steps"] - config["warmup"])
    episodes = defaultdict(lambda: {"sources": Counter(), "start": None})
    actual_sources = Counter()
    for step, row in enumerate(stream(root / "steps.jsonl"), 1):
        assert step == row["step"]
        expected = ("demonstration" if step <= demonstration_steps
                    else "warmup" if step <= config["warmup"] else "actor")
        assert row["source"] == expected
        ep = episodes[row["episode"]]
        if ep["start"] is None:
            assert row["episode_step"] == 1
            ep["start"] = step
        ep["sources"].update([row["source"]])
        ep["end"] = step
        ep["outcome"] = row["outcome"]
        actual_sources.update([row["source"]])
    assert actual_sources == +expected_sources
    results = defaultdict(list)
    all_outcomes = Counter()
    early = protocol["early_window_steps"]
    running_at_cutoff = None
    for row in stream(root / "episodes.jsonl"):
        ep = episodes.pop(row["episode"])
        assert sum(ep["sources"].values()) == row["episode_steps"]
        assert ep["end"] == row["step"]
        assert ep["outcome"] == ("running" if row["outcome"] == "partial" else row["outcome"])
        source = next(iter(ep["sources"])) if len(ep["sources"]) == 1 else "mixed"
        end = {"start_step": ep["start"], "end_step": ep["end"], "outcome": row["outcome"],
               "sources": dict(ep["sources"]), "goal_rad": row["goal_rad"]}
        results[source].append(end)
        all_outcomes.update([row["outcome"]])
        if ep["start"] <= early < ep["end"]:
            running_at_cutoff = {**end, "class": source}
    assert not episodes and all_outcomes == summary["episode_outcomes"]
    actor = results["actor"]
    early_actor = [row for row in actor if row["end_step"] <= early and row["outcome"] != "partial"]
    early_outcomes = Counter(row["outcome"] for row in early_actor)
    first_success = next((row["end_step"] for row in actor if row["outcome"] == "success"), None)
    return {
        "sources_by_step": dict(actual_sources),
        "total_outcomes_by_source": {key: dict(Counter(row["outcome"] for row in items))
                                     for key, items in results.items()},
        "demonstration_episodes": results["demonstration"],
        "random_only_episodes": results["warmup"],
        "mixed_episodes": results["mixed"],
        "early_actor_episode_count": len(early_actor),
        "early_actor_outcomes": dict(early_outcomes),
        "early_actor_timeout_fraction": early_outcomes["timeout"] / len(early_actor),
        "first_actor_only_success_step": first_success,
        "first_actor_only_success_within_early_window": (
            first_success if first_success is not None and first_success <= early else None),
        "episode_in_progress_at_early_cutoff": running_at_cutoff,
        "last_100_completed_actor_outcomes": dict(Counter(
            row["outcome"] for row in [row for row in actor if row["outcome"] != "partial"][-100:]
        )),
        "runtime_s": summary["rollout_update_s"],
    }


def summarize_evaluation(root, protocol, checkpoint):
    config = load(root / "config.json")
    for key in ("cases", "delay_steps", "position_error_threshold_rad", "speed_threshold_rad_s"):
        assert config[key] == protocol[key], key
    assert config["checkpoint_sha256"] == sha256(checkpoint.read_bytes()).hexdigest()
    summary = load(root / "summary.json")
    errors = defaultdict(list)
    for row in stream(root / "sac.jsonl"):
        obs = row["next_observation"]
        errors[row["seed"]].append(max(abs(q - goal) for q, goal in zip(obs[:2], obs[4:6])))
    cases = [{"case_id": row["seed"], "goal_rad": row["goal_rad"], "outcome": row["outcome"],
              "steps": row["steps"], "min_max_axis_error_rad": min(errors[row["seed"]]),
              "final_max_axis_error_rad": max(row["final_abs_error_rad"]),
              "final_max_axis_speed_rad_s": max(row["final_abs_speed_rad_s"])}
             for row in summary["sac"]["per_episode"]]
    return {"outcomes": dict(Counter(row["outcome"] for row in cases)), "cases": cases,
            "mean_success_steps": summary["sac"]["mean_success_steps"],
            "baseline_successes": summary["position_target"]["successes"],
            "median_best_error_rad": median(row["min_max_axis_error_rad"] for row in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("precision", type=Path)
    args = parser.parse_args()
    protocol_path = RUN / "protocol.json"
    protocol = load(protocol_path)
    status = list(stream(RUN / "status.jsonl"))
    assert status[0]["protocol_sha256"] == sha256(protocol_path.read_bytes()).hexdigest()
    assert status[-1]["stage"] == "job" and status[-1]["status"] == "complete"
    old, new = args.base / "train", RUN / "train"
    old_config, new_config = (load(p / "config.json") for p in (old, new))
    assert new_config["warmup_policy"] == protocol["demo_warmup_policy"]
    for key, value in old_config.items():
        if key != "source_sha256":
            assert new_config[key] == value, key
    for file, digest in old_config["source_sha256"].items():
        if file != "panda_reach.py":
            assert new_config["source_sha256"][file] == digest, file

    training = {"random": summarize_training(old, protocol, 0),
                "demo": summarize_training(new, protocol, protocol["demo_warmup_policy"]["demonstration_steps"])}
    assert len(training["demo"]["demonstration_episodes"]) == 8
    assert all(row["outcome"] == "success" for row in training["demo"]["demonstration_episodes"])
    evaluation = {"random": summarize_evaluation(RUN / "evaluation-random", protocol, old / "sac_cpu.pt"),
                  "demo": summarize_evaluation(RUN / "evaluation-demo", protocol, new / "sac_cpu.pt")}
    assert ((RUN / "evaluation-random/sac.jsonl").read_bytes()
            == (args.precision / "fresh/actor.jsonl").read_bytes())
    assert ((RUN / "evaluation-random/position_target.jsonl").read_bytes()
            == (RUN / "evaluation-demo/position_target.jsonl").read_bytes())
    report = {"training": training, "evaluation": evaluation, "training_config_matches_control": True,
              "historical_actor_eval_reproduced": True}
    with (RUN / "comparison.json").open("x", encoding="utf-8") as out:
        json.dump(report, out, indent=2, allow_nan=False)
    for key, value in training.items():
        print(key, json.dumps({k: v for k, v in value.items() if k != "demonstration_episodes"}))
        print("evaluation", json.dumps({k: v for k, v in evaluation[key].items() if k != "cases"}))


if __name__ == "__main__":
    main()
