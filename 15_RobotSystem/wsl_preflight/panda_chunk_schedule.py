"""Compare fixed and goal-change-triggered planning in Panda joint reaching."""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np


BLOCK = 8
SENSOR_DELAY = 2
CHANGE_TICKS = (None, 6, 9, 13)
INITIAL_GOAL = (0.1, 0.1)
MOVED_GOAL = (-0.08, -0.05)
ACTION_LIMIT = 0.02
MAX_STEPS = 60


def make_plan(command, goal):
    plan = deque()
    for _ in range(BLOCK):
        increment = np.clip(goal - command, -ACTION_LIMIT, ACTION_LIMIT)
        plan.append(increment)
        command = command + increment
    return plan


def run_episode(env_type, scene, mode, change_tick):
    env = env_type(scene, goal=INITIAL_GOAL, max_steps=MAX_STEPS)
    observation, _ = env.reset()
    visible_tick = None if change_tick is None else change_tick + SENSOR_DELAY
    plan = deque()
    plan_tick = 0
    planned_goal = INITIAL_GOAL
    previous_action = np.zeros(2)
    response_tick = None
    response_direction = None
    trace = []

    for tick in range(MAX_STEPS):
        command = observation[6:8]
        if tick == change_tick:
            env.goal[:] = MOVED_GOAL
            response_direction = env.goal - command
        observed_goal = MOVED_GOAL if visible_tick is not None and tick >= visible_tick else INITIAL_GOAL
        replan = not plan or (mode == 'event' and observed_goal != planned_goal)
        if replan:
            planned_goal = observed_goal
            plan_tick = tick
            plan = make_plan(command, np.array(planned_goal))

        observation, _, terminated, truncated, info = env.step(plan.popleft())
        applied = info['applied_increment_rad']
        jump = float(np.max(np.abs(applied - previous_action)))
        if response_direction is not None and response_tick is None and np.dot(applied, response_direction) > 1e-12:
            response_tick = tick
        trace.append({
            'tick': tick, 'true_goal_rad': env.goal.tolist(), 'observed_goal_rad': observed_goal,
            'planned_goal_rad': planned_goal, 'replan': replan, 'plan_age_ticks': tick - plan_tick,
            'applied_increment_rad': applied.tolist(), 'increment_jump_rad': jump,
            'position_rad': observation[:2].tolist(), 'velocity_rad_s': observation[2:4].tolist(),
            'command_rad': observation[6:8].tolist(), 'outcome': info['outcome'],
        })
        previous_action = applied
        if terminated or truncated:
            break

    summary = {
        'mode': mode, 'change_tick': change_tick, 'observed_change_tick': visible_tick,
        'response_tick': response_tick,
        'response_delay_ticks': None if response_tick is None else response_tick - change_tick,
        'steps': len(trace), 'plans': sum(row['replan'] for row in trace),
        'peak_increment_jump_rad': max(row['increment_jump_rad'] for row in trace),
        'outcome': trace[-1]['outcome'],
    }
    return summary, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True, help='15_RobotSystem source directory')
    parser.add_argument('--scene', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project.resolve()))
    from panda_two_joint_env import PandaTwoJointEnv

    summaries = []
    traces = []
    for change_tick in CHANGE_TICKS:
        for mode in ('fixed', 'event'):
            summary, trace = run_episode(PandaTwoJointEnv, args.scene, mode, change_tick)
            summaries.append(summary)
            traces.append(trace)

    report = {
        'mujoco': mujoco.__version__, 'numpy': np.__version__,
        'environment_sha256': hashlib.sha256((args.project / 'panda_two_joint_env.py').read_bytes()).hexdigest(),
        'scene_sha256': hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        'control_period_s': 0.02, 'block_ticks': BLOCK, 'goal_sensor_delay_ticks': SENSOR_DELAY,
        'max_steps': MAX_STEPS, 'initial_goal_rad': INITIAL_GOAL, 'moved_goal_rad': MOVED_GOAL,
        'episodes': summaries,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, data in [('summary.json', report), ('traces.json', traces)]:
        (args.output / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
