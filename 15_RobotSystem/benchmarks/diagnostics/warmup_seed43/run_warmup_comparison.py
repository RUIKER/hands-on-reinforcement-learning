"""Run and evaluate a seed-43 demonstration-warmup comparison."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys


RUN = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/warmup_seed43")).resolve()
PROTOCOL = RUN / "protocol.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prefix(path, count):
    with path.open("rb") as stream:
        return list(islice(stream, count))


def record(stage, state, **details):
    data = {"stage": stage, "status": state, "time_utc": datetime.now(timezone.utc).isoformat(), **details}
    with (RUN / "status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(data) + "\n")


def prepare(base, precision):
    old = base / "train"
    demo = RUN / "demo-smoke"
    assert prefix(RUN / "random-smoke/steps.jsonl", 320) == prefix(old / "steps.jsonl", 320)
    assert prefix(RUN / "random-smoke/updates.jsonl", 1) == prefix(old / "updates.jsonl", 1)
    demo_rows = [json.loads(line) for line in prefix(demo / "steps.jsonl", 320)]
    assert [row["source"] for row in demo_rows] == ["demonstration"] * 152 + ["warmup"] * 104 + ["actor"] * 64
    assert Counter(row["outcome"] for row in demo_rows[:152]) == {"running": 144, "success": 8}
    assert demo_rows[151]["terminated"] and demo_rows[152]["episode_step"] == 1
    assert read(demo / "summary.json")["updates"] == 64
    assert json.loads(prefix(demo / "updates.jsonl", 1)[0])["step"] == 257
    config = read(precision / "protocol.json")
    policy = read(demo / "config.json")["warmup_policy"]
    protocol = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "train_seed": 43,
        "steps": 50000, "warmup": 256, "demo_warmup_policy": policy,
        "control_checkpoint": str(old / "sac_cpu.pt"),
        "control_checkpoint_sha256": sha256((old / "sac_cpu.pt").read_bytes()).hexdigest(),
        "early_window_steps": 10000,
        "autonomous_episode": "completed by the end of the window and controlled by actor from reset",
        "fresh_goal_seed": config["fresh_goal_seed"], "cases": config["fresh_cases"],
        "goal_protocol_sha256": sha256((precision / "protocol.json").read_bytes()).hexdigest(),
        "position_error_threshold_rad": config["position_error_threshold_rad"],
        "speed_threshold_rad_s": config["speed_threshold_rad_s"], "delay_steps": 0,
        "smoke": {"control_320_steps_match_old_run": True,
                  "demonstration_steps_and_update_boundary_verified": True},
    }
    with PROTOCOL.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)


def launch(stage, command):
    record(stage, "running", command=command)
    with (RUN / f"{stage}.stdout.log").open("xb") as stdout, (RUN / f"{stage}.stderr.log").open("xb") as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    record(stage, "complete" if process.returncode == 0 else "failed", exit_code=process.returncode)
    if process.returncode:
        sys.exit(process.returncode)


def supervise(project, scene):
    config = read(PROTOCOL)
    record("job", "running", protocol_sha256=sha256(PROTOCOL.read_bytes()).hexdigest())
    launch("train", [sys.executable, str(project / "panda_reach.py"), "train", str(scene),
                     str(RUN / "train"), "--steps", str(config["steps"]), "--seed", str(config["train_seed"]),
                     "--her", "--demo-warmup", "--step-log-every", "1", "--update-log-every", "250"])
    for arm, checkpoint in (("demo", RUN / "train/sac_cpu.pt"), ("random", config["control_checkpoint"])):
        launch(f"evaluate-{arm}", [sys.executable, __file__, "evaluate", str(project), str(scene),
                                  str(checkpoint), str(RUN / f"evaluation-{arm}")])
    record("job", "complete")


def evaluate(project, scene, checkpoint, output):
    sys.path.insert(0, str(project))
    import panda_reach

    protocol = read(PROTOCOL)
    panda_reach.EVAL_CASES["precision_fresh"] = protocol["cases"]
    result = panda_reach.evaluate(argparse.Namespace(
        scene=scene, output=output, checkpoint=checkpoint, split="precision_fresh",
        delay_steps=protocol["delay_steps"],
    ))
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    setup = commands.add_parser("prepare")
    setup.add_argument("base", type=Path)
    setup.add_argument("precision", type=Path)
    for mode in ("run", "evaluate"):
        cmd = commands.add_parser(mode)
        cmd.add_argument("project", type=Path)
        cmd.add_argument("scene", type=Path)
        if mode == "evaluate":
            cmd.add_argument("checkpoint", type=Path)
            cmd.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.base, args.precision)
    elif args.mode == "run":
        supervise(args.project, args.scene)
    else:
        evaluate(args.project, args.scene, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
