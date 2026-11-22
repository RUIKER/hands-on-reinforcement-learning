"""Short paired study of rewards, behavior smoothing, and replay warm-up on Panda."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random

import mujoco
import numpy as np
import torch

from panda_cartesian_reach import CONFIG, evaluate, infer, teacher
from panda_ee_env import CartesianEnvironment, FAILURE_REWARD, distance_reward, features
from panda_ee_reach import Replay, SAC


ROOT = Path(__file__).resolve().parent
WARMUP = 256
BUDGET = 1536
SEEDS = (42, 43, 44)
# reward type, EMA weight of the current behavior action, warm-up action source
ARMS = {
    'sparse_raw': ('sparse', 1., 'random'),
    'sparse_ema': ('sparse', .5, 'random'),
    'dense_raw': ('dense', 1., 'random'),
    'dense_ema': ('dense', .5, 'random'),
    'dense_teacher_warmup': ('dense', 1., 'teacher'),
}


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sparse_reward(state, goal, reached):
    return 0. if reached else -1.


def run_arm(env, goals, seed, arm, out):
    kind, weight, warmup = ARMS[arm]
    reward_fn = distance_reward if kind == 'dense' else sparse_reward
    random.seed(seed)
    torch.manual_seed(seed)
    goals_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 1)
    learner = SAC(**CONFIG)
    replay = Replay(True, seed + 2, relabel_reward=reward_fn)
    warmup_digest = sha256()
    outcomes, prefix_outcomes, snapshots = Counter(), Counter(), []
    first_autonomous_success = None
    episode_start = 1
    previous = np.zeros(3)
    state = env.reset(goals['train'][goals_rng.integers(len(goals['train']))])
    for step in range(1, BUDGET + 1):
        if step <= WARMUP:
            action = teacher(state) if warmup == 'teacher' else action_rng.uniform(-1, 1, size=3)
        else:
            requested = learner.take_action(features(state))
            action = weight * requested + (1. - weight) * previous
        nxt, _, terminal, truncated, outcome = env.step(action)
        reward = FAILURE_REWARD if outcome == 'failure' else reward_fn(nxt, env.goal, outcome == 'success')
        replay.add(state, action, reward, nxt, terminal, truncated, outcome == 'failure')
        if step <= WARMUP:
            # Physical transitions must match for the four random-warmup groups.
            for data in (state, action, nxt):
                warmup_digest.update(np.asarray(data, dtype=np.float64).tobytes())
        else:
            old, actions, rewards, next_states, terminals = replay.sample(64)
            batch = features(old), actions, rewards, features(next_states), terminals
            capture = step in (WARMUP + 1, 512, 1024, BUDGET)
            diagnostic = learner.update(batch, diagnostics=capture)
            if capture:
                snapshots.append({'environment_step': step, **diagnostic})
        state, previous = nxt, action
        if env.done:
            if episode_start > WARMUP:
                outcomes[outcome] += 1
                if outcome == 'success' and first_autonomous_success is None:
                    first_autonomous_success = step
            else:
                prefix_outcomes[outcome] += 1
            episode_start = step + 1
            previous = np.zeros(3)
            state = env.reset(goals['train'][goals_rng.integers(len(goals['train']))])
    weights = {key: value.detach().cpu().numpy() for key, value in learner.actor.state_dict().items()}
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'actor.npz', **weights)
    # This is the original 24-goal diagnostic set. Keep the held-out 48 out of model selection.
    evaluation, _ = evaluate(env, {'frozen': goals['frozen']}, {'actor': lambda s: infer(learner, s)},
                             record_traces=False)
    cases = evaluation['actor']['frozen']
    write_json(out / 'evaluation.json', cases)
    result = {'success': sum(r['outcome'] == 'success' for r in cases),
              'eval_episodes': len(cases),
              'median_final_error_mm': float(np.median([r['final_error_m'] * 1000. for r in cases])),
              'warmup_dynamics_sha256': warmup_digest.hexdigest(), 'her_draws': dict(replay.counts),
              'warmup_or_crossing_completed_episodes': dict(prefix_outcomes),
              'fully_autonomous_completed_episodes': dict(outcomes),
              'first_fully_autonomous_success_step': first_autonomous_success,
              'training_snapshots': snapshots}
    write_json(out / 'training.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = ROOT / 'results' / '20261122_cartesian_reach'
    protocol = json.loads((source / 'protocol.json').read_text(encoding='utf-8'))
    models = {p.name: sha256(p.read_bytes()).hexdigest() for p in (args.scene, args.scene.with_name('panda.xml'))}
    if models != protocol['model_sha256']:
        raise ValueError('Use the model XML files from the seven-axis training protocol')
    goal_file = source / 'goals.json'
    goals = json.loads(goal_file.read_text(encoding='utf-8'))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'protocol.json', {
        'seeds': SEEDS, 'arms_reward_current_weight_warmup': ARMS,
        'environment_steps': BUDGET, 'warmup_steps': WARMUP, 'updates': BUDGET - WARMUP,
        'sac': CONFIG, 'batch_size': 64, 'her_attempted_fraction': .5,
        'true_failure_reward': FAILURE_REWARD, 'cloning_updates': 0, 'cloning_regularizer': 0,
        'initial_state': 'home', 'evaluation': 'First-hit position and speed, original 24 diagnostic goals, '
        'deterministic actor without smoothing. No selection or early stopping. 48 fresh goals unused.',
        'behavior_ema': 'Training only, after warm-up. Previous applied normalized action, zero at episode reset. '
        'Replay stores the actual smoothed action. A warm-up episode may continue into actor control.',
        'demo_warmup': 'First 256 steps driven by the Cartesian feedback teacher, stored in replay. '
        'No supervised loss or online demonstration regularizer in any arm.',
        'goals_sha256': sha256(goal_file.read_bytes()).hexdigest(), 'model_sha256': models,
        'mujoco': mujoco.__version__, 'torch': str(torch.__version__)})
    env = CartesianEnvironment(args.scene)
    results = {}
    for seed in SEEDS:
        arms = {}
        for name in ARMS:
            print(f'Running seed {seed} / {name}', flush=True)
            arms[name] = run_arm(env, goals, seed, name, args.output / f'seed{seed}' / name)
        if len({arms[a]['warmup_dynamics_sha256'] for a in ARMS if ARMS[a][2] == 'random'}) != 1:
            raise AssertionError('Unpaired initial random trajectories')
        results[seed] = {name: {key: data[key] for key in ('success', 'eval_episodes', 'median_final_error_mm')}
                         for name, data in arms.items()}
    write_json(args.output / 'summary.json', results)


if __name__ == '__main__':
    main()
