"""Run paired BC, BC + SAC, and BC + SAC + HER evaluations on fixed seven-axis goals."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

import mujoco
import numpy as np
import torch

from panda_cartesian_reach import (
    CLONING_UPDATES, CLONING_WEIGHT, CONFIG, CartesianEnvironment, SAC, evaluate, infer, train,
)
from panda_ee_reach import DISTANCE, HORIZON, SPEED


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_actor(path):
    agent = SAC(**CONFIG)
    with np.load(path, allow_pickle=False) as weights:
        agent.actor.load_state_dict({key: torch.from_numpy(weights[key]) for key in weights.files})
    return agent


def save_actor(path, agent):
    np.savez_compressed(path, **{k: v.numpy() for k, v in agent.actor.state_dict().items()})


def run_pair(env, goals, seed, steps, out):
    out.mkdir()
    training = {}
    for name, use_her in (("bc_sac", False), ("bc_sac_her", True)):
        print(f"Training {name}, seed {seed}", flush=True)
        bc, agent, record = train(env, goals["train"], seed, steps, use_her=use_her)
        if not use_her:
            save_actor(out / "bc.npz", bc)
        else:
            if record["bc_examples_sha256"] != training["bc_sac"]["bc_examples_sha256"]:
                raise RuntimeError("The two arms saw different demonstration examples")
            with np.load(out / "bc.npz", allow_pickle=False) as weights:
                for key, value in bc.actor.state_dict().items():
                    np.testing.assert_array_equal(value.numpy(), weights[key])
        save_actor(out / f"{name}.npz", agent)
        training[name] = record
    save_json(out / "training.json", training)
    report = evaluate_actors(env, goals, out)
    save_json(out / "evaluation.json", report)
    return report


def evaluate_actors(env, goals, out):
    cases = {split: goals[split] for split in ("frozen", "fresh")}
    policies = {}
    for name in ("bc", "bc_sac", "bc_sac_her"):
        agent = load_actor(out / f"{name}.npz")
        policies[name] = lambda state, model=agent: infer(model, state)
    report, _ = evaluate(env, cases, policies, record_traces=False)
    return report


def summarize(rows):
    return {
        "episodes": len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "median_final_error_mm": 1000 * median(row["final_error_m"] for row in rows),
        "mean_steps": mean(row["steps"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path, help="New result directory, or saved run with --verify")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--verify", action="store_true",
                        help="Re-evaluate saved weights and compare each episode on the same runtime")
    args = parser.parse_args()
    if args.steps <= 0 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive steps and distinct seeds")
    torch.set_num_threads(1)
    saved = Path(__file__).resolve().parent / "results" / "20261122_cartesian_reach"
    goal_file = saved / "goals.json"
    goal_hash = sha256(goal_file.read_bytes()).hexdigest()
    reference = json.loads((saved / "protocol.json").read_text(encoding="utf-8"))
    hashes = {p.name: sha256(p.read_bytes()).hexdigest()
              for p in (args.scene, args.scene.with_name("panda.xml"))}
    if hashes != reference["model_sha256"]:
        raise ValueError("This comparison requires the original model XML files")
    goals = json.loads(goal_file.read_text(encoding="utf-8"))
    env = CartesianEnvironment(args.scene)
    if args.verify:
        protocol = json.loads((args.output / "protocol.json").read_text(encoding="utf-8"))
        if hashes != protocol["model_sha256"] or goal_hash != protocol["goals_sha256"]:
            raise ValueError("Model or frozen goal data has changed since this run")
        for seed in protocol["seeds"]:
            out = args.output / f"seed{seed}"
            previous = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
            if evaluate_actors(env, goals, out) != previous:
                raise RuntimeError(f"Seed {seed} differs from the saved per-goal evaluation")
        print("All saved episode results reproduced.", flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "protocol.json", {
        "seeds": args.seeds, "steps_per_online_arm": args.steps,
        "goals_repo_path": goal_file.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
        "goals_sha256": goal_hash,
        "unique_goals": {k: len(v) for k, v in goals.items()}, "model_sha256": hashes,
        "initial_state": "home", "horizon_steps": HORIZON, "physics_steps_per_action": 10,
        "action_period_s": .02, "distance_lt_m": DISTANCE, "speed_lt_m_s": SPEED,
        "cloning_updates": CLONING_UPDATES, "online_actor_cloning_weight": CLONING_WEIGHT,
        "online_batch_size": 64, "her_candidate_fraction": 0.5, "actor_critic_config": CONFIG,
        "pairing": "Noisy demonstrations and actor initialization vary across seeds. Within each "
                   "seed, the script checks identical BC inputs, labels and pretrained weights. "
                   "Both online arms retain the same BC regularizer. Only HER sampling is toggled.",
        "evaluation": "Each saved actor is reloaded and evaluated deterministically. Episode "
                      "order matches goals.json within each split; seeds repeat the same goals.",
        "mujoco": mujoco.__version__, "torch": str(torch.__version__), "numpy": np.__version__,
    })
    by_seed = {}
    all_rows = {arm: {"frozen": [], "fresh": []} for arm in ("bc", "bc_sac", "bc_sac_her")}
    for seed in args.seeds:
        report = run_pair(env, goals, seed, args.steps, args.output / f"seed{seed}")
        by_seed[seed] = {}
        for arm, splits in report.items():
            by_seed[seed][arm] = {}
            for split, rows in splits.items():
                by_seed[seed][arm][split] = summarize(rows)
                all_rows[arm][split].extend(rows)
    pooled = {arm: {split: summarize(rows) for split, rows in splits.items()}
              for arm, splits in all_rows.items()}
    save_json(args.output / "summary.json", {"by_seed": by_seed, "pooled": pooled})


if __name__ == "__main__":
    main()
