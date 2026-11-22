"""Test upstream synchronous REACT feedback with a test-double action expert."""

import argparse
import json
from pathlib import Path
import runpy
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import torch


class ProbeExpert:
    react_layout = {'num_blocks': 3, 'block_size': 2, 'action_dim': 7, 'device': 'cpu'}

    def react_embed(self, obs):
        return torch.as_tensor(obs['position'].copy(), dtype=torch.float32)

    def react_denoise(self, position, buffer):
        proposal = torch.zeros(7)
        proposal[:3] = torch.tensor([0.25, 0.0, 0.0]) - position
        return proposal + 0.01 * buffer

    def react_decode(self, block, _prefix):
        return {'actions': block[0].clamp(-1, 1).numpy()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('react', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.react / 'packages/openpi-client/src'))
    sys.path.insert(0, str(args.react / 'src'))
    # This import is used by rolling.py for a type annotation. Keep the actual
    # upstream ReactPolicy and RollingBuffer implementations intact.
    policy_module = ModuleType('openpi.policies.policy')
    policy_module.Policy = ProbeExpert
    with patch.dict(sys.modules, {'openpi.policies.policy': policy_module}):
        rolling = runpy.run_path(str(args.react / 'src/openpi/react/rolling.py'))

    check = runpy.run_path('scripts/install_checks/check_libero.py')
    benchmark = check['get_benchmark']('libero_spatial')()
    task = benchmark.get_task(0)
    bddl = Path(check['get_libero_path']('bddl_files')) / task.problem_folder / task.bddl_file
    env = check['OffScreenRenderEnv'](bddl_file_name=str(bddl), camera_heights=64, camera_widths=64)
    try:
        def trial(use_fresh_observation):
            torch.manual_seed(0)
            env.seed(0)
            env.reset()
            observation = env.set_init_state(benchmark.get_task_init_states(0)[0])
            initial = observation['robot0_eef_pos'].copy()
            # LIBERO's 8D observation is distinct from its 7D action space.
            policy = rolling['ReactPolicy'](ProbeExpert(), warmup='noise')
            first = policy.infer({'position': initial})
            for action in first['actions']:
                observation, _, _, _ = env.step(action)
            measured = observation['robot0_eef_pos'].copy()
            second = policy.infer({'position': measured if use_fresh_observation else initial})
            assert first['react']['update'] == 1 and second['react']['update'] == 2
            return first['actions'], second['actions'], float(np.linalg.norm(measured - initial))

        first_fresh, next_fresh, displacement = trial(True)
        first_stale, next_stale, _ = trial(False)
        np.testing.assert_array_equal(first_fresh, first_stale)
        difference = float(np.max(np.abs(next_fresh - next_stale)))
        assert displacement > 0 and difference > 0
        print(json.dumps({'task_id': 0, 'seed': 0, 'mode': 'upstream_sync_wrapper_probe_expert',
                          'first_block_identical': True, 'eef_displacement_m': displacement,
                          'next_block_max_action_delta': difference}))
    finally:
        env.close()


if __name__ == '__main__':
    main()
