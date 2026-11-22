"""Audit saved seven-axis policies under fixed command delay and inspect replay semantics."""

import argparse
from collections import Counter, deque
from hashlib import sha256
import json
from pathlib import Path
import random

import mujoco
import numpy as np
import torch

from panda_cartesian_reach import (ACTION_SCALE, CONFIG, FAILURE_REWARD, JOINT_MARGIN,
                                   CartesianEnvironment, distance_reward, features, infer, teacher)
from panda_ee_reach import DISTANCE, HORIZON, LIMIT, SPEED, Replay, SAC, success


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_actor(path):
    actor = SAC(**CONFIG)
    with np.load(path, allow_pickle=False) as weights:
        actor.actor.load_state_dict({key: torch.from_numpy(weights[key].copy()) for key in weights})
    return actor


def rollout(env, policy, goal, delay, save_timing=False, *, project_pending=False):
    state = env.reset(goal)
    queued = deque(np.zeros(3) for _ in range(delay))
    timing = []
    while not env.done:
        observation = state
        if project_pending and queued:
            observation = state.copy()
            command = state[14:21]
            for pending in queued:
                command = env.project_action(command, pending)
            observation[14:21] = command
            observation[30:33], _ = env.ik.position(command)
        submitted = policy(observation)
        queued.append(submitted.copy())
        applied = queued.popleft()
        nxt, _, _, _, outcome = env.step(applied)
        delta_q = nxt[14:21] - state[14:21]
        if np.any(np.abs(delta_q) > LIMIT + 1e-10):
            raise AssertionError('Joint increment exceeds the controller limit')
        if (np.any(nxt[14:21] < env.ik.lower + JOINT_MARGIN - 1e-10)
                or np.any(nxt[14:21] > env.ik.upper - JOINT_MARGIN + 1e-10)):
            raise AssertionError('Position command violated its configured margin')
        if save_timing:
            timing.append({'step': env.steps, 'submitted': submitted.tolist(), 'applied': applied.tolist()})
        state = nxt
    if save_timing:
        for index, row in enumerate(timing):
            previous = timing[index - delay]['submitted'] if index >= delay else np.zeros(3)
            np.testing.assert_array_equal(row['applied'], previous)
    return {'outcome': outcome, 'steps': env.steps,
            'final_error_m': float(np.linalg.norm(state[21:24] - goal)),
            'final_speed_m_s': float(np.linalg.norm(state[24:27]))}, timing[:6]


def delay_comparison(env, saved, goals, policies):
    reference = load_json(saved / 'evaluation.json')
    rows, summaries, timeline = {}, {}, {}
    for name, policy in policies.items():
        rows[name], summaries[name], timeline[name] = {}, {}, {}
        for delay in (0, 1, 2):
            key = str(delay)
            rows[name][key], summaries[name][key] = {}, {}
            for split in ('frozen', 'fresh'):
                cases = []
                for index, goal in enumerate(goals[split]):
                    row, sample = rollout(env, policy, goal, delay, save_timing=(split == 'frozen' and index == 0))
                    if sample:
                        timeline[name][key] = sample
                    if delay == 0:
                        prior = reference[name][split][index]
                        if (row['outcome'], row['steps']) != (prior['outcome'], prior['steps']):
                            raise AssertionError('Zero delay diverged from saved evaluation')
                        for metric in ('final_error_m', 'final_speed_m_s'):
                            np.testing.assert_allclose(row[metric], prior[metric], rtol=1e-5, atol=1e-7)
                    cases.append(row)
                rows[name][key][split] = cases
                summaries[name][key][split] = {
                    'n': len(cases), 'outcomes': dict(Counter(r['outcome'] for r in cases)),
                    'mean_steps_including_timeouts': float(np.mean([r['steps'] for r in cases])),
                    'median_final_error_m': float(np.median([r['final_error_m'] for r in cases]))}
    return {'by_goal': rows, 'summary': summaries, 'first_six_commands_frozen_goal_0': timeline}


def check_held_commands(env, goal):
    env.reset(goal)
    independent = mujoco.MjData(env.model)
    mujoco.mj_resetDataKeyframe(env.model, independent, env.model.key('home').id)
    mujoco.mj_forward(env.model, independent)
    max_q_error, max_velocity_error = 0., 0.
    for action in ([0., 0., 0.], [1., -.5, .25], [-1., .5, -.25], [.4, 1., -.3]):
        nxt, _, _, _, _ = env.step(action)
        independent.ctrl[env.act] = nxt[14:21]
        for _ in range(10):
            mujoco.mj_step(env.model, independent)
        max_q_error = max(max_q_error, float(np.max(np.abs(nxt[:7] - independent.qpos[env.qadr]))))
        max_velocity_error = max(max_velocity_error,
                                 float(np.max(np.abs(nxt[7:14] - independent.qvel[env.dadr]))))
    np.testing.assert_allclose([max_q_error, max_velocity_error], 0., rtol=0, atol=1e-10)
    return {'physics_steps_per_command': 10, 'commands': 4, 'max_abs_q_difference_rad': max_q_error,
            'max_abs_velocity_difference_rad_s': max_velocity_error, 'elapsed_sim_s': float(env.data.time)}


def replay_audit(env, goals):
    random.seed(781)
    replay = Replay(True, 1268, relabel_reward=distance_reward)
    source = {}
    for goal in goals[:4]:
        path = []
        state = env.reset(goal)
        while not env.done:
            action = teacher(state)
            nxt, reward, term, trunc, outcome = env.step(action)
            replay.add(state, action, reward, nxt, term, trunc, outcome == 'failure')
            source[(nxt[:27].tobytes(), action.tobytes())] = (state, nxt, path, len(path))
            path.append(nxt[21:24])
            state = nxt
    counts = Counter()
    for _ in range(50):
        states, actions, rewards, next_states, terms = replay.sample(64)
        for state, action, reward, nxt, term in zip(states, actions, rewards, next_states, terms):
            old, old_next, path, step = source[(nxt[:27].tobytes(), action.tobytes())]
            np.testing.assert_array_equal(old[:27], state[:27])
            np.testing.assert_array_equal(old[30:], state[30:])
            np.testing.assert_array_equal(old_next[30:], nxt[30:])
            target = nxt[27:30]
            np.testing.assert_array_equal(state[27:30], target)
            if not np.array_equal(old[27:30], target):
                assert any(np.array_equal(target, point) for point in path[step:])
                assert not success(old, target)
                counts['relabeled'] += 1
                counts['relabeled_success'] += int(term)
            assert bool(term) == success(nxt, target)
            assert reward == distance_reward(nxt, target, term)
            counts['draws'] += 1
    assert counts['relabeled'] == replay.counts['relabeled'] and counts['relabeled_success'] > 0
    return dict(counts)


def boundary_samples(env, goal):
    state = env.reset(goal)
    action = np.array([1., -.5, .25])
    nxt, reward, term, trunc, outcome = env.step(action)
    replay = Replay(False, 12)
    replay.add(state, action, reward, nxt, term, trunc, False)
    before, actions, rewards, after, dones = replay.sample(1)
    np.testing.assert_array_equal(state, before[0])
    np.testing.assert_array_equal(action, actions[0])
    np.testing.assert_array_equal(nxt, after[0])
    assert rewards[0] == reward and not dones[0] and outcome == 'running'
    sample = {'goal_m': state[27:30].tolist(), 'actor_features': features(state).tolist(),
              'normalized_replay_action': actions[0].tolist(),
              'requested_cartesian_delta_m': (action * ACTION_SCALE).tolist(),
              'applied_joint_setpoint_delta_rad': (nxt[14:21] - state[14:21]).tolist(),
              'actual_joint_delta_rad': (nxt[:7] - state[:7]).tolist(),
              'actual_hand_delta_m': (nxt[21:24] - state[21:24]).tolist(),
              'distance_reward': reward, 'terminal': term, 'truncated': trunc}

    state = env.reset(goal)
    for _ in range(HORIZON):
        nxt, reward, term, trunc, outcome = env.step(np.zeros(3))
        if env.done:
            break
        state = nxt
    assert outcome == 'timeout' and env.steps == HORIZON and not term and trunc
    replay = Replay(False, 51)
    replay.add(state, np.zeros(3), reward, nxt, term, trunc, False)
    assert not replay.sample(1)[-1][0] and replay.episode == []

    env.reset(goal)
    env.data.qpos[env.qadr[1]] = env.ik.upper[1] + .6
    mujoco.mj_forward(env.model, env.data)
    state = env.observation()
    nxt, reward, term, trunc, outcome = env.step(np.zeros(3))
    assert outcome == 'failure' and term and not trunc and reward == FAILURE_REWARD
    replay = Replay(True, 24, relabel_reward=distance_reward)
    replay.add(state, np.zeros(3), reward, nxt, term, trunc, True)
    stationary = env.reset(goal)
    replay.add(stationary, np.zeros(3), -1., stationary, False, True, False)
    failure_draws = 0
    for _ in range(32):
        batch = replay.sample(2)
        if np.array_equal(batch[0][0], state):
            assert batch[-1][0] and batch[2][0] == FAILURE_REWARD
            np.testing.assert_array_equal(batch[3][0][27:30], goal)
            failure_draws += 1
    assert failure_draws > 0
    edge = np.zeros(33)
    at_origin = np.zeros(3)
    edge[21] = DISTANCE
    assert not success(edge, at_origin)
    edge[21] = np.nextafter(DISTANCE, 0)
    edge[24] = SPEED
    assert not success(edge, at_origin)
    edge[24] = np.nextafter(SPEED, 0)
    assert success(edge, at_origin)
    return {'transition_from_home': sample, 'timeout_step': HORIZON, 'timeout_bootstrap_mask': 1,
            'injected_joint2_failure_bootstrap_mask': 0, 'failure_reward': reward,
            'failure_draws_in_her_slot': failure_draws, 'strict_distance_and_speed_edges_passed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('saved', type=Path, help='Directory containing seed-43 policies and evaluation')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output directory')
    torch.set_num_threads(1)
    protocol = load_json(args.saved / 'protocol.json')
    if {p.name: digest(p) for p in (args.scene, args.scene.with_name('panda.xml'))} != protocol['model_sha256']:
        raise ValueError('Model XML differs from saved training protocol')
    goals = load_json(args.saved / 'goals.json')
    env = CartesianEnvironment(args.scene)
    bc, sac = (load_actor(args.saved / name) for name in ('bc.npz', 'sac_her_bc.npz'))
    policies = {'cartesian_feedback': teacher, 'bc': lambda state: infer(bc, state),
                'sac_her_bc': lambda state: infer(sac, state)}
    delay = delay_comparison(env, args.saved, goals, policies)
    checks = {'seed': protocol['training_seed'], 'mujoco': mujoco.__version__, 'torch': str(torch.__version__),
              'input_sha256': {name: digest(args.saved / name) for name in
                               ('goals.json', 'bc.npz', 'sac_her_bc.npz', 'evaluation.json')},
              'model_sha256': protocol['model_sha256'], 'delay_steps': [0, 1, 2],
              'fixed_physics_dt_s': env.model.opt.timestep, 'fixed_control_dt_s': .02,
              'command_hold': check_held_commands(env, goals['frozen'][0]),
              'boundary_samples': boundary_samples(env, goals['frozen'][0]),
              'her_sample_audit': replay_audit(env, goals['frozen'])}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, record in [('checks', checks), ('delay_summary', delay['summary']),
                         ('delay_timeline', delay['first_six_commands_frozen_goal_0']), ('delay_cases', delay['by_goal'])]:
        (args.output / f'{name}.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n',
                                                  encoding='utf-8')
    print(json.dumps(checks, indent=2))
    print(json.dumps(delay['summary'], indent=2))


if __name__ == '__main__':
    main()
