"""Evaluate an IK handoff for the archived seven-joint SAC+HER actor."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np
import torch

from panda_ee_reach import CONFIG_7, Environment, LIMIT, SAC


PATIENCE = 25  # control steps, 0.5 s in simulation
PROGRESS_M = 0.001


def rollout(env, goal, actor, allow_handoff):
    observation = env.reset(goal)
    reference = float(np.linalg.norm(observation[21:24] - goal))
    last_progress = 0
    target = None
    handoff = None
    prefix = []
    while not env.done:
        if allow_handoff and handoff is None and env.steps - last_progress >= PATIENCE:
            solution = env.ik.solve(goal)
            if solution['status'] != 'converged':
                raise ValueError('IK handoff requires an IK-convergent goal')
            target = np.asarray(solution['joint_angles_rad'])
            handoff = env.steps
        if target is None:
            with torch.inference_mode():
                x = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
                action = actor.deterministic_action(x).squeeze(0).numpy()
            prefix.append((observation.tobytes(), action.tobytes()))
        else:
            action = np.clip((target - observation[14:21]) / LIMIT, -1., 1.)
        observation, _, _, _, outcome = env.step(action)
        error = float(np.linalg.norm(observation[21:24] - goal))
        if error <= reference - PROGRESS_M:
            reference = error
            last_progress = env.steps
    return {'outcome': outcome, 'steps': env.steps, 'handoff_after_steps': handoff,
            'final_error_m': error, 'final_speed_m_s': float(np.linalg.norm(observation[24:27]))}, prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('legacy', type=Path, help='20261122_seven_axis_reach result directory')
    parser.add_argument('cartesian', type=Path, help='20261122_cartesian_reach result directory')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(1)
    env = Environment(args.scene)
    actor = SAC(**CONFIG_7).actor
    weights = args.legacy / 'sac_her' / 'actor.npz'
    with np.load(weights, allow_pickle=False) as archive:
        actor.load_state_dict({k: torch.from_numpy(archive[k].copy()) for k in archive.files})
    recorded = json.loads((args.legacy / 'evaluation.json').read_text())['sac_her']
    goals = json.loads((args.cartesian / 'goals.json').read_text())
    if goals['frozen'] != json.loads((args.legacy / 'goals.json').read_text())['test']:
        raise ValueError('frozen target order differs from the legacy evaluation')
    results, summary = {}, {}
    for split in ('frozen', 'fresh'):
        rows = []
        for i, goal in enumerate(goals[split]):
            raw, full_trace = rollout(env, goal, actor, False)
            hybrid, prefix = rollout(env, goal, actor, True)
            if prefix != full_trace[:len(prefix)]:
                raise RuntimeError(f'actor paths differ before handoff for {split}/{i}')
            if split == 'frozen':
                original = recorded[i]
                if (raw['outcome'] != original['outcome'] or raw['steps'] != original['steps']
                        or abs(raw['final_error_m'] - original['final_error_m']) > 1e-8):
                    raise RuntimeError(f'failed to reproduce archived actor for goal {i}')
            rows.append({'goal_index': i, 'goal_link0_m': goal, 'actor': raw, 'actor_then_ik': hybrid})
        results[split] = rows
        switches = [r['actor_then_ik']['handoff_after_steps'] for r in rows
                    if r['actor_then_ik']['handoff_after_steps'] is not None]
        summary[split] = {k: dict(Counter(r[k]['outcome'] for r in rows)) for k in ('actor', 'actor_then_ik')}
        summary[split].update(handoffs=len(switches), median_handoff_step=median(switches) if switches else None)
    report = {'actor_sha256': sha256(weights.read_bytes()).hexdigest(),
              'goals_sha256': sha256((args.cartesian / 'goals.json').read_bytes()).hexdigest(),
              'scene_sha256': sha256(args.scene.read_bytes()).hexdigest(),
              'patience_steps': PATIENCE, 'minimum_progress_m': PROGRESS_M,
              'archived_frozen_actor_reproduced': True, 'identical_actor_prefixes': True,
              'summary': summary, 'cases': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
