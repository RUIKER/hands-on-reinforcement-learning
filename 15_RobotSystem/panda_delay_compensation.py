"""Check whether accounting for commands in a fixed FIFO restores point-to-point stopping."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median

import torch

from check_cartesian_execution import digest, load_actor, load_json, rollout
from panda_cartesian_reach import CartesianEnvironment, fresh_goals, infer, teacher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('saved', type=Path, help='20261122_cartesian_reach directory')
    parser.add_argument('uncompensated', type=Path, help='Previously measured delay_cases.json')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    reference = load_json(args.uncompensated)
    goals = load_json(args.saved / 'goals.json')
    model_hashes = {p.name: digest(p) for p in (args.scene, args.scene.with_name('panda.xml'))}
    if model_hashes != load_json(args.saved / 'protocol.json')['model_sha256']:
        raise ValueError('model differs from training')
    env = CartesianEnvironment(args.scene)
    confirm_seed = 61124
    goals['confirm'] = fresh_goals(env, goals['train'] + goals['frozen'] + goals['fresh'], seed=confirm_seed)
    bc = load_actor(args.saved / 'bc.npz')
    rl = load_actor(args.saved / 'sac_her_bc.npz')
    policies = {'cartesian_feedback': teacher, 'bc': lambda s: infer(bc, s), 'sac_her_bc': lambda s: infer(rl, s)}
    cases, summary = {}, {}
    for name, policy in policies.items():
        cases[name], summary[name] = {}, {}
        for delay in (0, 1, 2):
            key = str(delay)
            cases[name][key], summary[name][key] = {}, {}
            for split in ('frozen', 'fresh', 'confirm'):
                rows, control = [], []
                for idx, goal in enumerate(goals[split]):
                    before, _ = rollout(env, policy, goal, delay)
                    if split != 'confirm' and before != reference[name][key][split][idx]:
                        raise AssertionError(f'Uncompensated results differ at {name}/{key}/{split}/{idx}')
                    control.append(before)
                    after, _ = rollout(env, policy, goal, delay, project_pending=True)
                    if delay == 0 and before != after:
                        raise AssertionError('Compensation changed zero-delay behavior')
                    rows.append(after)
                cases[name][key][split] = {'before': control, 'after': rows}
                summary[name][key][split] = {
                    'uncompensated': dict(Counter(r['outcome'] for r in control)),
                    'compensated': dict(Counter(r['outcome'] for r in rows)),
                    'compensated_median_final_error_mm': median(r['final_error_m'] * 1000 for r in rows),
                }
    report = {'training_seed': 43, 'delay_steps': [0, 1, 2], 'control_period_ms': 20,
              'inputs': 'Measured state, current commanded joints, own pending commands and known fixed delay. '
                        'Roll forward only commanded joints and the corresponding commanded hand pose. '
                        'Keep measured joints, hand and speed unchanged. No policy retraining.',
              'input_sha256': {p.name: digest(p) for p in (args.uncompensated, args.saved / 'goals.json',
                                                           args.saved / 'bc.npz', args.saved / 'sac_her_bc.npz')},
              'confirm_goal_seed': confirm_seed, 'confirm_goals_m': goals['confirm'],
              'model_sha256': model_hashes, 'summary': summary, 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(summary['sac_her_bc'], indent=2))


if __name__ == '__main__':
    main()
