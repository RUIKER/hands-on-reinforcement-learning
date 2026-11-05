"""跨章节使用独立进程，避免同名 config/agents 模块相互污染。"""

import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def run_python(self, chapter, source):
        result = subprocess.run(
            [sys.executable, '-c', source], cwd=ROOT / chapter,
            text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cliff_entrypoints(self):
        for algorithm in ('q-learning', 'sarsa'):
            with self.subTest(algorithm=algorithm):
                result = subprocess.run(
                    [sys.executable, str(ROOT / '01_Cliff Walking/train.py'),
                     '--algorithm', algorithm, '--episodes', '3', '--seed', '0'],
                    capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Episode: 0', result.stdout)

    def test_sarsa_executes_sampled_next_action(self):
        self.run_python('01_Cliff Walking', '''
from unittest.mock import patch
import train
class Agent:
    def __init__(self, **kwargs): self.expected = None; self.count = 0
    def take_action(self, *state):
        self.count += 1
        return self.count % 4
    def update(self, action, state, next_state, reward, done, next_action):
        if self.expected is not None: assert action == self.expected
        self.expected = next_action
class Env:
    def __init__(self, **kwargs): self.steps = 0
    def reset(self): self.steps = 0
    def step(self, action):
        self.steps += 1
        return -1, [0, 0], self.steps == 4
with patch.object(train, 'SarsaTable', Agent), patch.object(train, 'Env', Env):
    assert train.cliff_walk('sarsa', 2) == [-4, -4]
''')

    def test_minimal_dqn_update_and_target_sync(self):
        self.run_python('02_DQN', '''
import numpy as np
import torch
from practice import DQN
torch.manual_seed(0)
a = DQN(4, 8, 2, .01, .99, .1, 2, torch.device('cpu'))
assert all(torch.equal(x, y) for x,y in zip(a.q_net.parameters(), a.target_q_net.parameters()))
before = [p.clone() for p in a.q_net.parameters()]
batch = dict(states=np.ones((8,4)), actions=[0]*8, rewards=[1]*8,
             next_states=np.zeros((8,4)), dones=[False]*8)
loss = a.update(batch)
assert np.isfinite(loss)
assert any(not torch.equal(x,y) for x,y in zip(before,a.q_net.parameters()))
assert all(p.grad is None for p in a.target_q_net.parameters())
assert all(torch.equal(x,y) for x,y in zip(before,a.target_q_net.parameters()))
a.update(batch)
assert all(torch.equal(x,y) for x,y in zip(a.q_net.parameters(),a.target_q_net.parameters()))
''')

    def test_dqn_counts_episodes_and_bootstraps_at_time_limit(self):
        self.run_python('03_Double DQN、Dueling DQN', '''
import numpy as np
from trainer import train_dqn
class Space:
    low = [-2.0]; high = [2.0]
class Env:
    action_space = Space()
    def reset(self): return np.zeros(3), {}
    def step(self, action): return np.ones(3), -1, False, True, {}
class Agent:
    action_dim = 3
    def take_action(self, state): return 1
    def max_q_value(self, state): return 0
    def update(self, batch): raise AssertionError('warmup should not update')
class Buffer:
    def __init__(self): self.terminals = []
    def add(self, s,a,r,ns,d): self.terminals.append(d)
    def size(self): return len(self.terminals)
b = Buffer()
returns, values = train_dqn(Agent(), Env(), 3, b, 100, 4)
assert returns == [-1, -1, -1] and len(values) == 3
assert b.terminals == [False]*3
''')

    @unittest.skipIf(os.name == 'nt', 'Bash checks run on POSIX')
    def test_shell_workers_propagate_failures(self):
        for script in ('seed_experiments/her_main/run_all.sh',
                       'seed_experiments/halfcheetah_sac/run_all.sh',
                       'ablations/her_ratio/run_ratio_ablation.sh',
                       'ablations/her_strategy/run_strategy_ablation.sh',
                       'ablations/sac_hparams/run_sweep.sh',
                       'ablations/sac_hparams/run_sensitivity.sh'):
            with self.subTest(script=script), tempfile.TemporaryDirectory() as tmp:
                # 复制入口和共享控制器，避免测试写入历史 results/。
                base = Path(tmp) / 'experiments'
                target = base / script
                target.parent.mkdir(parents=True)
                shutil.copy(ROOT / 'experiments' / script, target)
                shutil.copy(ROOT / 'experiments/_runner.sh', base / '_runner.sh')
                fake_python = Path(tmp) / 'python-worker'
                fake_python.write_text(
                    '#!/usr/bin/env bash\n'
                    'if [ "$1" = "grid.py" ]; then echo "1|test|--gamma 0.9"; exit 0; fi\n'
                    'exit 7\n'
                )
                fake_python.chmod(0o755)
                env = dict(os.environ, PY=str(fake_python), SEEDS='0', JOBS='1')
                env.pop('DRY', None)
                result = subprocess.run(['bash', str(target)], env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn('DONE', result.stdout)
                self.assertIn('[FAIL]', result.stderr)

    def test_worker_pool_success_and_invalid_parallelism(self):
        helper = ROOT / 'experiments/_runner.sh'
        command = f'source "{helper}"; submit_job true; submit_job true; wait_all'
        for jobs, expected in (('1', 0), ('0', 2)):
            result = subprocess.run(['bash', '-c', command],
                                    env=dict(os.environ, PY=sys.executable, JOBS=jobs),
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, expected, result.stderr)


if __name__ == '__main__':
    unittest.main()
