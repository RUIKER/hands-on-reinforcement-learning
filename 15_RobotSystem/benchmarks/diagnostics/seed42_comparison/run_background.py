import json
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

entry, scene = sys.argv[1:]
run = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/seed42_comparison")).resolve()
run.mkdir(parents=True, exist_ok=True)
commands = {
    "train": [sys.executable, entry, "train", scene, str(run / "train"),
              "--steps", "50000", "--seed", "42", "--her",
              "--step-log-every", "1", "--update-log-every", "250"],
    "validation": [sys.executable, entry, "evaluate", scene, str(run / "validation"),
                   "--split", "validation", "--delay-steps", "0",
                   "--checkpoint", str(run / "train" / "sac_cpu.pt")],
}

def record(stage, status, **details):
    row = dict(stage=stage, status=status, time_utc=datetime.now(timezone.utc).isoformat(), **details)
    with (run / "status.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")

for stage, command in commands.items():
    record(stage, "running", command=command)
    try:
        with (run / f"{stage}.stdout.log").open("xb") as stdout, (run / f"{stage}.stderr.log").open("xb") as stderr:
            process = subprocess.run(command, stdout=stdout, stderr=stderr)
    except Exception as exc:
        record(stage, "failed", error=repr(exc))
        raise
    if process.returncode:
        record(stage, "failed", exit_code=process.returncode)
        sys.exit(process.returncode)
    record(stage, "complete", exit_code=0)
record("job", "complete")
