"""HER 的核心：按整条轨迹存储，并在采样时做 hindsight relabeling。

和 09_DDPG/Pendulum/rl_utils.py 的 ReplayBuffer 的区别：
- DDPG 存单步 (s, a, r, s', done)，采样直接随机取 transition；
- HER 必须存整条轨迹，因为 future 策略要取"同一条轨迹里 t 之后"的状态；
- HER 在 sample() 里换目标、重算 reward 和 terminated，而不是在存储时做。

这个文件是本次实验里唯一真正新增的逻辑，其余文件基本是 DDPG 的搬运。
"""

import collections
import random

import numpy as np


class Trajectory:
    """一条完整轨迹。

    states 比 actions 多一个：
        states = [s0, s1, ..., sT]
        actions / rewards / terminated = [a0, ..., a_{T-1}], ...
    """

    def __init__(self, init_state):
        self.states = [np.asarray(init_state, dtype=np.float32).copy()]
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.length = 0

    def store_step(self, action, next_state, reward, terminated):
        """存一步。next_state 是执行动作之后的状态 s_{t+1}。"""
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.states.append(np.asarray(next_state, dtype=np.float32).copy())
        self.rewards.append(float(reward))
        # 只存 terminated，不存 truncated：超时不是"到达目标"，
        # 不能作为 Bellman target 的终止信号。
        self.terminated.append(bool(terminated))
        self.length += 1

    def __len__(self):
        return self.length


class ReplayBufferTrajectory:
    """存整条轨迹的 replay buffer，并在 sample() 里做 HER 重标注。"""

    def __init__(self, capacity, env, use_her=True, her_ratio=0.8,
                 goal_strategy='future'):
        self.buffer = collections.deque(maxlen=capacity)
        # 重标注时直接用环境的评分函数，避免两份 -1/0 逻辑不一致。
        self.env = env
        self.use_her = bool(use_her)
        self.her_ratio = float(her_ratio)
        self.goal_strategy = goal_strategy
        if goal_strategy not in ('future', 'final', 'episode'):
            raise ValueError(
                "goal_strategy 必须是 'future' / 'final' / 'episode'"
            )

    def add_trajectory(self, trajectory):
        self.buffer.append(trajectory)

    def size(self):
        return len(self.buffer)

    def _choose_goal(self, traj, t, strategy):
        """按策略从轨迹里选一个假目标，只取位置两维。"""
        if strategy == 'future':
            # t 之后（含 t+1 和最后一个状态）等概率选一个时刻。
            k = random.randint(t + 1, traj.length)
        elif strategy == 'final':
            # 只取轨迹最终状态，多样性最差。
            k = traj.length
        else:
            # 整条轨迹任意时刻，可能选到已经达成的平凡目标。
            k = random.randint(0, traj.length)
        return traj.states[k][:2].copy()

    def sample(self, batch_size, use_her=None, her_ratio=None,
               goal_strategy=None):
        """采样一个 batch；use_her=True 时按比例做 hindsight relabeling。

        参数留了 None 默认值，方便消融时临时覆盖 config 里的开关。
        """
        if len(self.buffer) == 0:
            raise ValueError("replay buffer 为空，不能采样")
        if use_her is None:
            use_her = self.use_her
        if her_ratio is None:
            her_ratio = self.her_ratio
        if goal_strategy is None:
            goal_strategy = self.goal_strategy

        batch = {
            'states': [], 'actions': [], 'next_states': [],
            'rewards': [], 'dones': [],
        }
        collected = 0
        while collected < batch_size:
            traj = self.buffer[random.randrange(len(self.buffer))]
            if traj.length == 0:
                continue

            t = random.randrange(traj.length)
            state = traj.states[t]
            next_state = traj.states[t + 1]
            action = traj.actions[t]
            reward = traj.rewards[t]
            terminated = traj.terminated[t]

            if use_her and random.random() < her_ratio:
                goal = self._choose_goal(traj, t, goal_strategy)
                # 奖励描述的是"执行 a_t 之后有没有到达"，所以看 s_{t+1}。
                achieved = next_state[:2]
                reward = float(self.env.compute_reward(achieved, goal))
                terminated = bool(self.env.is_success(achieved, goal))
                # s_t 与 s_{t+1} 的目标两维必须换成同一个 g'，
                # 否则 Q(s, a | g') 的 Bellman 等式不成立。
                state = np.concatenate([state[:2], goal])
                next_state = np.concatenate([next_state[:2], goal])

            batch['states'].append(state)
            batch['actions'].append(action)
            batch['next_states'].append(next_state)
            batch['rewards'].append(reward)
            batch['dones'].append(terminated)
            collected += 1

        return {
            'states': np.asarray(batch['states'], dtype=np.float32),
            'actions': np.asarray(batch['actions'], dtype=np.float32),
            'next_states': np.asarray(batch['next_states'], dtype=np.float32),
            'rewards': np.asarray(batch['rewards'], dtype=np.float32),
            'dones': np.asarray(batch['dones'], dtype=np.float32),
        }


def moving_average(values, window_size=9):
    """因果滑动平均，输出长度与输入一致（开头窗口逐步变大）。"""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()
    cumulative = np.cumsum(np.insert(values, 0, 0))
    indices = np.arange(1, values.size + 1)
    starts = np.maximum(0, indices - window_size)
    counts = indices - starts
    return ((cumulative[indices] - cumulative[starts]) / counts).astype(
        np.float32
    )
