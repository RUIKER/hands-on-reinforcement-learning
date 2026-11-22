"""Train and evaluate a demonstration-guided Cartesian SAC policy for Panda."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from panda_ee_env import (ACTION_SCALE, FAILURE_REWARD, JOINT_MARGIN, CartesianEnvironment,
                          distance_reward, features)
from panda_ee_reach import CONFIG_7, Replay, SAC


FRESH_SEED = 61123
CLONING_UPDATES = 2400
CLONING_WEIGHT = 20.
CONFIG = dict(CONFIG_7, state_dim=16, hidden_dim=64, action_low=[-1.] * 3,
              action_high=[1.] * 3, target_entropy=-3.)


def teacher(state):
    # Smooth Cartesian setpoint feedback used to collect demonstrations.
    return np.tanh((state[..., 27:30] - state[..., 30:33]) / ACTION_SCALE)


def infer(agent, state):
    with torch.no_grad():
        return agent.actor.deterministic_action(torch.as_tensor(features(state), dtype=torch.float32)).numpy()


def train(env, goals, seed, steps, *, use_her=True):
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    agent = SAC(**CONFIG)
    replay = Replay(use_her, seed + 1, relabel_reward=distance_reward)
    demonstrations = []
    demo_states = []
    demo_outcomes = Counter()
    for noise in (0., .15):
        for goal in goals:
            state = env.reset(goal)
            while not env.done:
                expert_action = teacher(state)
                action = np.clip(expert_action + rng.normal(0., noise, 3), -1., 1.)
                next_state, reward, terminal, timeout, outcome = env.step(action)
                replay.add(state, action, reward, next_state, terminal, timeout, outcome == 'failure')
                demonstrations.append(expert_action)
                demo_states.append(features(state))
                state = next_state
            demo_outcomes[outcome] += 1

    inputs = torch.as_tensor(np.asarray(demo_states), dtype=torch.float32)
    labels = torch.as_tensor(np.asarray(demonstrations), dtype=torch.float32)
    demo_digest = sha256(inputs.numpy().tobytes())
    demo_digest.update(labels.numpy().tobytes())

    def cloning_loss(actor):
        indices = rng.integers(len(inputs), size=128)
        return F.mse_loss(actor.deterministic_action(inputs[indices]), labels[indices])

    for _ in range(CLONING_UPDATES):
        agent.actor_optimizer.zero_grad()
        loss = cloning_loss(agent.actor)
        loss.backward()
        agent.actor_optimizer.step()
    pretrained = SAC(**CONFIG)
    pretrained.actor.load_state_dict(agent.actor.state_dict())

    def actor_regularizer(actor):
        return CLONING_WEIGHT * cloning_loss(actor)

    state = env.reset(goals[rng.integers(len(goals))])
    outcomes = Counter()
    for _ in range(steps):
        action = agent.take_action(features(state))
        next_state, reward, terminal, timeout, outcome = env.step(action)
        replay.add(state, action, reward, next_state, terminal, timeout, outcome == 'failure')
        states, actions, rewards, next_states, dones = replay.sample(64)
        batch = (features(states), actions, rewards, features(next_states), dones)
        agent.update(batch, actor_regularizer=actor_regularizer)
        state = next_state
        if env.done:
            outcomes[outcome] += 1
            state = env.reset(goals[rng.integers(len(goals))])
    return pretrained, agent, {'demo_transitions': len(inputs), 'demo_outcomes': dict(demo_outcomes),
                               'bc_examples_sha256': demo_digest.hexdigest(),
                               'online_steps': steps, 'online_outcomes': dict(outcomes),
                               'her': dict(replay.counts)}


def fresh_goals(env, excluded, *, seed=FRESH_SEED):
    rng = np.random.default_rng(seed)
    home = env.reset(np.zeros(3))[21:24]
    goals = []
    for _ in range(10000):
        offset = rng.uniform(-.10, .10, size=3)
        goal = home + offset
        if not .05 <= np.linalg.norm(offset) <= .13:
            continue
        if min(np.linalg.norm(goal - point) for point in excluded + goals) < .01:
            continue
        if env.ik.solve(goal)['status'] == 'converged':
            goals.append(goal.tolist())
        if len(goals) == 48:
            return goals
    raise RuntimeError('Could not sample 48 independent goals')


def evaluate(env, cases, policies, *, record_traces=True):
    report = {}
    traces = {}
    for name, policy in policies.items():
        report[name], traces[name] = {}, {}
        for split, goals in cases.items():
            outcomes, paths = [], []
            for goal in goals:
                state = env.reset(goal)
                path = []
                while True:
                    if record_traces:
                        path.append({'q': state[:7].tolist(), 'xyz': state[21:24].tolist(),
                                     'time': round(env.steps * .02, 8)})
                    if env.done:
                        break
                    state, _, _, _, outcome = env.step(policy(state))
                outcomes.append({'outcome': outcome, 'steps': env.steps,
                                 'final_error_m': float(np.linalg.norm(state[21:24] - goal)),
                                 'final_speed_m_s': float(np.linalg.norm(state[24:27]))})
                paths.append(path)
            counts = Counter(row['outcome'] for row in outcomes)
            print(name, split, dict(counts), flush=True)
            report[name][split] = outcomes
            traces[name][split] = paths
    return report, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('goals', type=Path, help='Original seven-axis goals.json')
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=12000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--weights', type=Path, help='Load a prior run directory instead of training')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.set_num_threads(1)
    original = json.loads(args.goals.read_text(encoding='utf-8'))
    env = CartesianEnvironment(args.scene)
    cases = {'frozen': original['test'], 'fresh': fresh_goals(env, original['train'] + original['test'])}
    source_hash = sha256(args.goals.read_bytes()).hexdigest()
    model_hash = {path.name: sha256(path.read_bytes()).hexdigest()
                  for path in (args.scene, args.scene.with_name('panda.xml'))}
    if args.weights:
        saved = json.loads((args.weights / 'protocol.json').read_text(encoding='utf-8'))
        if saved['source_goals_sha256'] != source_hash or saved['model_sha256'] != model_hash:
            raise ValueError('Saved policy requires the same frozen goals and model XML')
        args.seed = saved['training_seed']
    args.output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    if args.weights:
        bc, agent = SAC(**CONFIG), SAC(**CONFIG)
        for name, item in (('bc', bc), ('sac_her_bc', agent)):
            with np.load(args.weights / f'{name}.npz', allow_pickle=False) as archive:
                item.actor.load_state_dict({key: torch.from_numpy(archive[key]) for key in archive.files})
        training = None
    else:
        bc, agent, training = train(env, original['train'], args.seed, args.steps)
        for name, item in (('bc', bc), ('sac_her_bc', agent)):
            np.savez_compressed(args.output / f'{name}.npz',
                                **{key: value.numpy() for key, value in item.actor.state_dict().items()})
    if training is not None:
        training['seconds'] = perf_counter() - started
    report, traces = evaluate(env, cases, {
        'cartesian_feedback': teacher, 'bc': lambda state: infer(bc, state),
        'sac_her_bc': lambda state: infer(agent, state),
    })
    records = {'goals': {**cases, 'train': original['train']}, 'evaluation': report,
               'trajectories': traces, 'protocol': {
                   'training_seed': args.seed, 'fresh_goal_seed': FRESH_SEED,
                   'action_scale_m': ACTION_SCALE, 'joint_margin_rad': JOINT_MARGIN,
                   'failure_reward': FAILURE_REWARD, 'source_goals_sha256': source_hash,
                   'model_sha256': model_hash, 'cloning_updates': CLONING_UPDATES,
                   'cloning_weight': CLONING_WEIGHT, 'actor_critic_config': CONFIG,
                   'mujoco': mujoco.__version__, 'torch': str(torch.__version__), 'training': training,
               }}
    for name, record in records.items():
        (args.output / f'{name}.json').write_text(json.dumps(record, allow_nan=False,
                                                              indent=None if name == 'trajectories' else 2),
                                                  encoding='utf-8')


if __name__ == '__main__':
    main()
