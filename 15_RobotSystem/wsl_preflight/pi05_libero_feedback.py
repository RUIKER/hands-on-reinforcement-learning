"""Pair two next-chunk predictions from the official LIBERO π0.5 checkpoint."""

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np


def simulate(output, actions=None):
    check = runpy.run_path('scripts/install_checks/check_libero.py')
    utils = runpy.run_path('src/verl_vla/envs/libero/utils.py')
    benchmark = check['get_benchmark']('libero_spatial')()
    task = benchmark.get_task(0)
    bddl = Path(check['get_libero_path']('bddl_files')) / task.problem_folder / task.bddl_file
    env = check['OffScreenRenderEnv'](bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(benchmark.get_task_init_states(0)[0])
        for _ in range(10):
            observation, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            assert not done
        if actions is not None:
            for action in np.load(actions, allow_pickle=False):
                observation, _, done, _ = env.step(action)
                assert not done, 'Task ended before the next-chunk observation'
        state = np.concatenate([
            observation['robot0_eef_pos'], utils['quat2axisangle'](observation['robot0_eef_quat'].copy()),
            observation['robot0_gripper_qpos'],
        ])
        np.savez(output, image=utils['get_libero_image'](observation),
                 wrist_image=utils['get_libero_wrist_image'](observation), state=state, prompt=task.language)
    finally:
        env.close()


def compare(args):
    import jax
    from openpi.policies import policy_config
    from openpi.training import config
    from openpi_client import image_tools

    def observe(name, actions=None):
        target = args.output / name
        command = [str(args.sim_python), __file__, 'sim', str(target)]
        if actions:
            command += ['--actions', str(actions)]
        subprocess.run(command, cwd=args.verl, timeout=45, check=True)
        with np.load(target, allow_pickle=False) as data:
            return {
                'observation/image': image_tools.resize_with_pad(data['image'], 224, 224),
                'observation/wrist_image': image_tools.resize_with_pad(data['wrist_image'], 224, 224),
                'observation/state': data['state'], 'prompt': data['prompt'].item(),
            }

    args.output.mkdir(parents=True, exist_ok=False)
    stale = observe('initial.npz')
    cfg = config.get_config('pi05_libero')
    print('JAX devices:', jax.devices(), flush=True)
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    print('π0.5 loaded', flush=True)
    generator = np.random.default_rng(0)
    noise_shape = (cfg.model.action_horizon, cfg.model.action_dim)
    initial_noise = generator.standard_normal(noise_shape, dtype=np.float32)
    shared_noise = generator.standard_normal(noise_shape, dtype=np.float32)

    def infer(observation, noise):
        start = time.perf_counter()
        actions = policy.infer(observation, noise=noise)['actions']
        assert actions.shape == (cfg.model.action_horizon, 7) and np.isfinite(actions).all()
        return actions, time.perf_counter() - start

    first, first_s = infer(stale, initial_noise)
    first_path = args.output / 'first_block.npy'
    np.save(first_path, first[:5])
    fresh = observe('fresh.npz', first_path)
    stale_next, stale_s = infer(stale, shared_noise)
    fresh_next, fresh_s = infer(fresh, shared_noise)
    delta = fresh_next - stale_next
    np.savez(args.output / 'next_chunks.npz', stale=stale_next, fresh=fresh_next)
    report = {
        'policy': 'openpi pi05_libero', 'task_id': 0, 'initial_state_index': 0, 'noise_seed': 0,
        'executed_actions': 5,
        'eef_displacement_m': float(np.linalg.norm(fresh['observation/state'][:3] - stale['observation/state'][:3])),
        'first_action_max_delta': float(np.max(np.abs(delta[0]))),
        'next_five_max_delta': float(np.max(np.abs(delta[:5]))),
        'whole_chunk_max_delta': float(np.max(np.abs(delta))),
        'first_prediction_s': first_s, 'stale_prediction_s': stale_s, 'fresh_prediction_s': fresh_s,
    }
    result = json.dumps(report, indent=2)
    (args.output / 'summary.json').write_text(result + '\n', encoding='utf-8')
    print(result, flush=True)


parser = argparse.ArgumentParser(description=__doc__)
commands = parser.add_subparsers(dest='command', required=True)
sim = commands.add_parser('sim')
sim.add_argument('output', type=Path)
sim.add_argument('--actions', type=Path)
runner = commands.add_parser('compare')
for option in ('sim-python', 'verl', 'checkpoint', 'output'):
    runner.add_argument('--' + option, type=Path, required=True)
if __name__ == '__main__':
    arguments = parser.parse_args()
    if arguments.command == 'sim':
        simulate(arguments.output, arguments.actions)
    else:
        compare(arguments)
