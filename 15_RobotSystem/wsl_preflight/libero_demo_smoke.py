"""Replay one original HDF5 demonstration and export two ACT examples."""

import argparse
import json
from pathlib import Path
import runpy

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    check = runpy.run_path('scripts/install_checks/check_libero.py')
    utils = runpy.run_path('src/verl_vla/envs/libero/utils.py')
    benchmark = check['get_benchmark']('libero_spatial')()
    with h5py.File(args.dataset, 'r') as source:
        group = source['data']
        bddl_name = Path(group.attrs['bddl_file_name']).name
        task_id = next(i for i in range(benchmark.n_tasks) if benchmark.get_task(i).bddl_file == bddl_name)
        demo = group['demo_0']
        actions = np.asarray(demo['actions'], dtype=np.float32)
        states = np.asarray(demo['states'])
        assert actions.shape == (len(states), 7) and 5 <= len(actions) <= 400
        assert np.isfinite(actions).all() and np.isfinite(states).all()

    task = benchmark.get_task(task_id)
    bddl = Path(check['get_libero_path']('bddl_files')) / task.problem_folder / task.bddl_file
    env = check['OffScreenRenderEnv'](
        bddl_file_name=str(bddl), camera_names=list(check['CAMERA_NAMES']),
        camera_heights=64, camera_widths=64, camera_depths=False,
    )
    inputs = {'observation.images.image': [], 'observation.images.wrist_image': [], 'observation.state': [], 'action': []}
    try:
        env.seed(0)
        env.reset()
        observation = env.set_init_state(states[0])
        success = False
        for tick, action in enumerate(actions):
            if tick < 2:
                for name, fn in [('image', 'get_libero_image'), ('wrist_image', 'get_libero_wrist_image')]:
                    image = utils[fn](observation)
                    assert image.shape == (64, 64, 3) and image.max() > image.min()
                    inputs[f'observation.images.{name}'].append(image.transpose(2, 0, 1) / np.float32(255))
                inputs['observation.state'].append(np.concatenate([
                    observation['robot0_eef_pos'], utils['quat2axisangle'](observation['robot0_eef_quat'].copy()),
                    observation['robot0_gripper_qpos'],
                ]))
                inputs['action'].append(actions[tick:tick + 4])
            observation, _, _, _ = env.step(action)
            success |= bool(env.check_success())
    finally:
        env.close()

    batch = {name: np.asarray(values, dtype=np.float32) for name, values in inputs.items()}
    assert batch['observation.state'].shape == (2, 8) and batch['action'].shape == (2, 4, 7)
    assert all(np.isfinite(tensor).all() for tensor in batch.values())
    np.savez(args.output, **batch)
    print(json.dumps({'task_id': task_id, 'bddl': bddl_name, 'episode': 'demo_0',
                      'replayed_steps': len(actions), 'success_seen': success,
                      'batch_shapes': {k: list(v.shape) for k, v in batch.items()}}))


if __name__ == '__main__':
    main()
