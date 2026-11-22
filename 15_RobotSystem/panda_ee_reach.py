"""Seven-joint Panda Cartesian reaching with paired SAC and HER evaluation."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import random
from time import perf_counter

import mujoco
import numpy as np
import torch

from panda_ik import HandPositionIK
from panda_sac_smoke import SAC, CONFIG


LIMIT = 0.02
DISTANCE = 0.02
SPEED = 0.03
HORIZON = 250
WARMUP = 256
CONFIG_7 = dict(CONFIG, state_dim=30, action_low=[-1.] * 7,
                action_high=[1.] * 7, target_entropy=-7.)


def success(obs, goal):
    return bool(np.linalg.norm(obs[21:24] - goal) < DISTANCE
                and np.linalg.norm(obs[24:27]) < SPEED)


class Environment:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.ik = HandPositionIK(self.model)
        self.act = np.array([self.model.actuator(f'actuator{i}').id for i in range(1, 8)])
        self.qadr = self.ik.qadr
        self.dadr = self.ik.dadr
        self.jac = np.empty((3, self.model.nv))
        self.done = True

    def observation(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        rot = self.data.xmat[self.ik.base].reshape(3, 3).T
        pos = rot @ (self.data.xpos[self.ik.hand] - self.data.xpos[self.ik.base])
        mujoco.mj_jacBody(self.model, self.data, self.jac, None, self.ik.hand)
        vel = rot @ self.jac @ self.data.qvel
        return np.r_[self.data.qpos[self.qadr], self.data.qvel[self.dadr],
                     self.data.ctrl[self.act], pos, vel, self.goal].copy()

    def reset(self, goal):
        self.goal = np.asarray(goal, dtype=float).copy()
        if self.goal.shape != (3,) or not np.isfinite(self.goal).all():
            raise ValueError('Expected finite 3D goal')
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.done = False
        return self.observation()

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended')
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError('Expected seven finite normalized actions')
        self.data.ctrl[self.act] = np.clip(self.data.ctrl[self.act] + LIMIT * action,
                                         self.ik.lower, self.ik.upper)
        mujoco.mj_step(self.model, self.data, nstep=10)
        self.steps += 1
        obs = self.observation()
        q = self.data.qpos[self.qadr]
        failed = (not np.isfinite(obs).all() or any(w.number for w in self.data.warning)
                  or np.any(q < self.ik.lower - .001) or np.any(q > self.ik.upper + .001))
        reached = not failed and success(obs, self.goal)
        terminated = bool(failed or reached)
        truncated = bool(not terminated and self.steps >= HORIZON)
        self.done = terminated or truncated
        return obs, (0. if reached else -1.), terminated, truncated, ('failure' if failed else 'success' if reached else 'timeout' if truncated else 'running')


class Replay:
    def __init__(self, her, seed, relabel_reward=None):
        self.buffer = deque(maxlen=100000)
        self.her = her
        self.rng = random.Random(seed)
        self.episode = []
        self.counts = Counter()
        self.relabel_reward = relabel_reward

    def add(self, state, action, reward, nxt, terminal, truncated, failed):
        step = len(self.episode)
        self.episode.append(nxt[21:24].copy() if not failed else None)
        self.buffer.append((state.copy(), action.copy(), reward, nxt.copy(), terminal,
                            self.episode, step, failed))
        if terminal or truncated:
            self.episode = []

    def sample(self, n):
        rows = random.sample(self.buffer, n)
        batch = []
        for index, (state, action, reward, nxt, terminal, episode, step, failed) in enumerate(rows):
            if self.her and index < n // 2 and not failed:
                choices = [p for p in episode[step:] if p is not None]
                if choices:
                    goal = self.rng.choice(choices)
                    if not success(state, goal):
                        state, nxt = state.copy(), nxt.copy()
                        state[27:30] = nxt[27:30] = goal
                        terminal = success(nxt, goal)
                        if self.relabel_reward is None:
                            reward = 0. if terminal else -1.
                        else:
                            reward = self.relabel_reward(nxt, goal, terminal)
                        self.counts['relabeled'] += 1
                        self.counts['relabeled_success'] += int(terminal)
                    else:
                        self.counts['already_at_goal'] += 1
            batch.append((state, action, reward, nxt, terminal))
        return tuple(np.asarray(v) for v in zip(*batch))


def make_goals(env):
    env.reset([0., 0., 0.])
    origin = env.observation()[21:24]
    rng = np.random.default_rng(61122)
    goals = []
    while len(goals) < 104:
        offset = rng.uniform(-.10, .10, 3)
        if not .05 <= np.linalg.norm(offset) <= .13:
            continue
        goal = origin + offset
        result = env.ik.solve(goal)
        if result['status'] == 'converged':
            goals.append(goal.tolist())
    return {'train': goals[:80], 'test': goals[80:], 'seed': 61122,
            'description': 'uniform Cartesian offsets, IK-converged, fixed home, 80 train / 24 held out'}


def train(env, goals, output, seed, use_her, steps):
    torch.manual_seed(seed)
    random.seed(seed)
    goal_rng = np.random.default_rng(seed)
    action_rng = np.random.default_rng(seed + 1)
    replay = Replay(use_her, seed + 2)
    agent = SAC(**CONFIG_7)
    obs = env.reset(goals[goal_rng.integers(len(goals))])
    counts = Counter()
    started = perf_counter()
    episodes = []
    for step in range(1, steps + 1):
        action = action_rng.uniform(-1, 1, 7) if step <= WARMUP else agent.take_action(obs)
        nxt, reward, terminal, truncated, outcome = env.step(action)
        replay.add(obs, action, reward, nxt, terminal, truncated, outcome == 'failure')
        if not np.isfinite(nxt).all():
            raise RuntimeError(f'Non-finite simulation state at step {step}')
        if step > WARMUP:
            agent.update(replay.sample(64))
        obs = nxt
        if terminal or truncated:
            counts[outcome] += 1
            episodes.append({'step': step, 'outcome': outcome, 'episode_steps': env.steps})
            obs = env.reset(goals[goal_rng.integers(len(goals))])
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'config': CONFIG_7, 'actor': agent.actor.state_dict(), 'seed': seed,
                'steps': steps, 'her': use_her}, output / 'actor.pt')
    np.savez_compressed(output / 'actor.npz',
                        **{name: tensor.detach().cpu().numpy()
                           for name, tensor in agent.actor.state_dict().items()})
    report = {'seed': seed, 'steps': steps, 'updates': steps - WARMUP,
              'seconds': perf_counter() - started, 'episodes': dict(counts),
              'her_counts': dict(replay.counts), 'history': episodes}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return agent


def evaluate(env, goals, agents, output):
    results, traces = {}, {}
    for name in ('ik', 'sac', 'sac_her'):
        results[name] = []
        traces[name] = []
        for goal in goals:
            obs = env.reset(goal)
            target = None
            if name == 'ik':
                solution = env.ik.solve(goal)
                if solution['status'] != 'converged':
                    raise RuntimeError('Frozen target is no longer IK-convergent')
                target = np.array(solution['joint_angles_rad'])
            trace = [{'q': obs[:7].tolist(), 'xyz': obs[21:24].tolist(), 'time': 0.}]
            while not env.done:
                if name == 'ik':
                    action = np.clip((target - obs[14:21]) / LIMIT, -1, 1)
                else:
                    with torch.no_grad():
                        action = agents[name].actor.deterministic_action(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                obs, _, _, _, outcome = env.step(action)
                trace.append({'q': obs[:7].tolist(), 'xyz': obs[21:24].tolist(), 'time': env.steps * .02})
            traces[name].append(trace)
            results[name].append({'outcome': outcome, 'steps': env.steps,
                                  'final_error_m': float(np.linalg.norm(obs[21:24] - goal)),
                                  'final_speed_m_s': float(np.linalg.norm(obs[24:27]))})
        print(name, Counter(x['outcome'] for x in results[name]), flush=True)
    (output / 'evaluation.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (output / 'trajectories.json').write_text(json.dumps(traces), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--evaluate-only', action='store_true',
                        help='Re-evaluate saved actors and frozen goals in an existing run')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.steps <= WARMUP:
        raise ValueError('steps must exceed warmup')
    env = Environment(args.scene)
    if args.evaluate_only:
        goals = json.loads((args.output / 'goals.json').read_text(encoding='utf-8'))
        agents = {}
        for name in ('sac', 'sac_her'):
            agent = SAC(**CONFIG_7)
            with np.load(args.output / name / 'actor.npz', allow_pickle=False) as archive:
                agent.actor.load_state_dict({key: torch.from_numpy(archive[key].copy())
                                             for key in archive.files})
            agents[name] = agent
        evaluate(env, goals['test'], agents, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    goals = make_goals(env)
    (args.output / 'goals.json').write_text(json.dumps(goals, indent=2), encoding='utf-8')
    agents = {}
    for name, her in [('sac', False), ('sac_her', True)]:
        print('training', name, flush=True)
        agents[name] = train(env, goals['train'], args.output / name, args.seed, her, args.steps)
    evaluate(env, goals['test'], agents, args.output)


if __name__ == '__main__':
    main()
