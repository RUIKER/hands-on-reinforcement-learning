"""Paired warmup experiment; default is a read-only command preview."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SCENE = Path(os.environ.get("PANDA_SCENE", "mujoco_menagerie/franka_emika_panda/scene.xml"))
ROOT = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/panda-warmup-multiseed")).resolve()
SEEDS = (251, 379, 487, 593, 719)
GOAL_SEED = 2026111301


def cases():
    rng = random.Random(GOAL_SEED)
    rows = []
    for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        for _ in range(12):
            rows.append([40101 + len(rows), [round(sx * rng.uniform(.04, .16), 6),
                                             round(sy * rng.uniform(.04, .16), 6)]])
    return rows


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def plan():
    for seed in SEEDS:
        for arm in ('random', 'demo'):
            base = ROOT / f'seed-{seed}' / arm
            cmd = [sys.executable, str(PROJECT / 'panda_reach.py'), 'train', str(SCENE),
                   str(base / 'train'), '--steps', '50000', '--seed', str(seed), '--her',
                   '--step-log-every', '1', '--update-log-every', '250']
            if arm == 'demo':
                cmd.append('--demo-warmup')
            yield seed, arm, base, cmd


def evaluate(checkpoint, output, protocol_path):
    sys.path.insert(0, str(PROJECT))
    import panda_reach
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    panda_reach.EVAL_CASES['confirmatory'] = protocol['cases']
    panda_reach.evaluate(argparse.Namespace(scene=SCENE, output=output, checkpoint=checkpoint,
                                           split='confirmatory', delay_steps=0))


def record(stage, **values):
    with (ROOT / 'status.jsonl').open('a', encoding='utf-8') as out:
        out.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), 'stage': stage,
                              **values}, ensure_ascii=False) + '\n')


def launch(cmd, log):
    record('start', command=cmd)
    with (log.parent / (log.name + '.stdout.log')).open('xb') as out, \
         (log.parent / (log.name + '.stderr.log')).open('xb') as err:
        result = subprocess.run(cmd, stdout=out, stderr=err)
    record('finish', command=cmd, exit_code=result.returncode)
    if result.returncode:
        raise RuntimeError(f'Command failed with exit code {result.returncode}; see {log}.*.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', help='actually start experiment')
    parser.add_argument('--reference-config', type=Path, help='Original run config for historical hash and version checks')
    parser.add_argument('--eval', nargs=3, metavar=('CHECKPOINT', 'OUTPUT', 'PROTOCOL'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.eval:
        evaluate(*(Path(p) for p in args.eval))
        return
    if ROOT.exists():
        raise FileExistsError(f'Result directory already exists: {ROOT}; preserve it and choose a new directory')
    print('Result directory:', ROOT)
    for _, _, base, cmd in plan():
        print(subprocess.list2cmdline(cmd))
        print(subprocess.list2cmdline([sys.executable, __file__, '--eval', str(base / 'train/sac_cpu.pt'),
                                      str(base / 'evaluation'), str(ROOT / 'protocol.json')]))
    if not args.run:
        print('DRY RUN: no experiment started and no result directory created.')
        return
    if args.reference_config is None:
        parser.error('--run requires --reference-config from the historical exploratory run')
    old = args.reference_config
    historical = json.loads(old.read_text(encoding='utf-8'))
    for name in ('panda_reach.py', 'panda_two_joint_env.py', 'panda_sac_smoke.py', 'panda_her.py'):
        if digest(PROJECT / name) != historical['source_sha256'][name]:
            raise RuntimeError(f'Source changed since exploratory run: {name}; review protocol before running')
    for name, file in [('scene', SCENE), ('panda', SCENE.with_name('panda.xml'))]:
        if digest(file) != historical['xml_sha256'][name]:
            raise RuntimeError(f'Model changed since exploratory run: {file}')
    sys.path.insert(0, str(PROJECT))
    import mujoco
    import numpy
    import torch
    import panda_sac_smoke
    for name in ('agents.py', 'networks.py', 'rl_utils.py'):
        if digest(panda_sac_smoke.SAC_DIR / name) != historical['source_sha256'][name]:
            raise RuntimeError(f'SAC source changed since exploratory run: {name}')
    for name, actual in [('mujoco', mujoco.__version__), ('numpy', numpy.__version__), ('torch', str(torch.__version__))]:
        if actual != historical[name]:
            raise RuntimeError(f'{name} version changed: {actual} != {historical[name]}')
    ROOT.mkdir(parents=True)
    protocol = {'seeds': SEEDS, 'steps': 50000, 'goal_seed': GOAL_SEED, 'cases': cases(),
                'goal_distribution': '12 per quadrant; independent uniform magnitudes [0.04, 0.16]',
                'delay_steps': 0, 'historical_config_sha256': digest(old),
                'runner_sha256': digest(Path(__file__)),
                'success_rule': 'existing environment; max position error < 0.01 rad and max speed < 0.05 rad/s',
                'primary': 'mean across paired training seeds of (demo success rate - random success rate)'}
    protocol_path = ROOT / 'protocol.json'
    protocol_path.write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    record('protocol_frozen', sha256=digest(protocol_path))
    try:
        for seed, arm, base, cmd in plan():
            base.mkdir(parents=True)
            launch(cmd, base / 'train')
            summary = json.loads((base / 'train/summary.json').read_text(encoding='utf-8'))
            if summary['status'] != 'PASS' or summary['steps'] != 50000:
                raise RuntimeError(f'Invalid training summary: {base}')
            launch([sys.executable, __file__, '--eval', str(base / 'train/sac_cpu.pt'),
                    str(base / 'evaluation'), str(protocol_path)], base / 'evaluate')
            evaluation = json.loads((base / 'evaluation/summary.json').read_text(encoding='utf-8'))
            if evaluation['sac']['episodes'] != len(protocol['cases']):
                raise RuntimeError(f'Incomplete evaluation: {base}')
    except Exception as exc:
        record('job_failed', error=str(exc))
        raise
    record('job_complete')
    print('Completed. Inspect status and summaries; analyze paired differences as specified in the plan.')


if __name__ == '__main__':
    main()
