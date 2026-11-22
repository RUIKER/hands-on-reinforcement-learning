"""Freeze Panda evaluation goals and supervise paired SAC/HER runs on CPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from threading import Lock


RUN = Path(os.environ.get("PANDA_PAIRED_RUN", "runs/panda-paired")).resolve()
MANIFEST = RUN / "protocol.json"
STATUS = RUN / "status.jsonl"
LOCK = Lock()
SEEDS = (43, 137, 2027)
ARMS = ("sac", "sac_her")
GOAL_RNG_SEED = 6111201


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze(project, scene):
    sys.path.insert(0, str(project))
    from panda_reach import EVAL_CASES, TRAIN_GOALS
    from panda_two_joint_env import PandaTwoJointEnv

    env = PandaTwoJointEnv(scene)
    excluded = [tuple(round(x * 1000) for x in goal) for goal in TRAIN_GOALS]
    excluded.extend(tuple(round(x * 1000) for x in goal)
                    for cases in EVAL_CASES.values() for _, goal in cases)
    rng = random.Random(GOAL_RNG_SEED)
    cases = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for _ in range(6):
            for _ in range(10000):
                candidate = (sx * rng.randint(40, 160), sy * rng.randint(40, 160))
                if all(max(abs(x - px), abs(y - py)) >= 21
                       for x, y in [candidate] for px, py in excluded):
                    break
            else:
                raise RuntimeError("Could not place separated held-out goals")
            excluded.append(candidate)
            goal = [x / 1000 for x in candidate]
            env.validate_goal(goal)
            cases.append([30101 + len(cases), goal])
    protocol = {
        "frozen_at_utc": now(), "training_seeds": SEEDS, "arms": ARMS,
        "steps_per_run": 50000, "step_log_every": 1, "update_log_every": 250,
        "workers": len(SEEDS), "goal_rng_seed": GOAL_RNG_SEED,
        "goals": {"per_sign_quadrant": 6, "abs_min_rad": 0.04, "abs_max_rad": 0.16,
                  "grid_rad": 0.001, "min_chebyshev_distance_rad": 0.021},
        "evaluation": {"split": "heldout_v1", "delay_steps": 0, "policy": "deterministic",
                       "cases": cases, "evaluate_seed42_reference_checkpoints": True},
    }
    with MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    print(json.dumps(protocol))


def record(stage, state, **details):
    row = {"stage": stage, "status": state, "time_utc": now(), **details}
    with LOCK, STATUS.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def launch(stage, seed, arm, command, log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    record(stage, "running", seed=seed, arm=arm, command=command)
    with (log_dir / f"{stage}.stdout.log").open("xb") as stdout, (log_dir / f"{stage}.stderr.log").open("xb") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode:
        record(stage, "failed", seed=seed, arm=arm, exit_code=process.returncode)
        raise RuntimeError(f"{seed} {arm} {stage} exited {process.returncode}")
    record(stage, "complete", seed=seed, arm=arm, exit_code=0)


def evaluate(project, scene, checkpoint, output):
    sys.path.insert(0, str(project))
    import panda_reach
    protocol = json.loads(MANIFEST.read_text(encoding="utf-8"))
    evaluation = protocol["evaluation"]
    panda_reach.EVAL_CASES[evaluation["split"]] = evaluation["cases"]
    result = panda_reach.evaluate(argparse.Namespace(
        scene=scene, output=output, checkpoint=checkpoint, split=evaluation["split"],
        delay_steps=evaluation["delay_steps"],
    ))
    print(json.dumps(result, indent=2, allow_nan=False))


def supervise(project, scene, sparse_reference, her_reference):
    manifest_bytes = MANIFEST.read_bytes()
    protocol = json.loads(manifest_bytes)
    record("job", "running", protocol_sha256=sha256(manifest_bytes).hexdigest())

    def evaluate_arm(seed, arm, root, checkpoint):
        launch("evaluate", seed, arm, [sys.executable, __file__, "evaluate", str(project),
                                      str(scene), str(checkpoint), str(root / "heldout")], root)

    def train_pair(seed):
        order = list(protocol["arms"])
        if seed == protocol["training_seeds"][1]:
            order.reverse()
        for arm in order:
            root = RUN / f"seed-{seed}" / arm
            command = [sys.executable, str(project / "panda_reach.py"), "train",
                       str(scene), str(root / "train"), "--steps", str(protocol["steps_per_run"]),
                       "--seed", str(seed), "--step-log-every", str(protocol["step_log_every"]),
                       "--update-log-every", str(protocol["update_log_every"])]
            if arm == "sac_her":
                command.append("--her")
            launch("train", seed, arm, command, root)
            evaluate_arm(seed, arm, root, root / "train" / "sac_cpu.pt")

    failures = []
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        futures = {pool.submit(train_pair, seed): seed for seed in protocol["training_seeds"]}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures.append({"seed": futures[future], "error": str(exc)})
    for arm, checkpoint in zip(protocol["arms"], (sparse_reference, her_reference)):
        try:
            evaluate_arm(42, arm, RUN / "reference-42" / arm, checkpoint)
        except Exception as exc:
            failures.append({"seed": 42, "arm": arm, "error": str(exc)})
    record("job", "failed" if failures else "complete", failures=failures)
    if failures:
        raise RuntimeError(f"Some runs did not finish: {failures}")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("freeze", "run", "evaluate"):
        command = sub.add_parser(mode)
        command.add_argument("project", type=Path)
        command.add_argument("scene", type=Path)
        if mode == "run":
            command.add_argument("sparse_reference", type=Path)
            command.add_argument("her_reference", type=Path)
        if mode == "evaluate":
            command.add_argument("checkpoint", type=Path)
            command.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.project, args.scene)
    elif args.mode == "run":
        supervise(args.project, args.scene, args.sparse_reference, args.her_reference)
    else:
        evaluate(args.project, args.scene, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
