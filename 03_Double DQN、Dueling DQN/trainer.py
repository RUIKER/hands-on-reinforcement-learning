"""离散动作 Pendulum 的经验采集与 DQN 更新。"""

import numpy as np
from tqdm import tqdm

from environment import dis_to_con


def train_dqn(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    """执行恰好 num_episodes 个回合，返回回报和逐步平滑最大 Q 值。"""
    returns, max_q_values = [], []
    smoothed_q = 0.0
    with tqdm(total=num_episodes, desc="Training") as progress:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_return = 0.0
            while True:
                action = agent.take_action(state)
                smoothed_q = 0.005 * agent.max_q_value(state) + 0.995 * smoothed_q
                max_q_values.append(smoothed_q)
                continuous_action = dis_to_con(action, env, agent.action_dim)
                next_state, reward, terminated, truncated, _ = env.step([continuous_action])
                # 时间限制只结束采样，Bellman 目标仍保留下一状态价值。
                replay_buffer.add(state, action, reward, next_state, terminated)
                state = next_state
                episode_return += reward
                if replay_buffer.size() >= max(minimal_size, batch_size):
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    agent.update({
                        "states": states, "actions": actions, "rewards": rewards,
                        "next_states": next_states, "dones": dones,
                    })
                if terminated or truncated:
                    break
            returns.append(episode_return)
            if (episode + 1) % 10 == 0:
                progress.set_postfix(episode=episode + 1, mean_return=np.mean(returns[-10:]))
            progress.update(1)
    return returns, max_q_values


# 兼容早期笔记和调用方；新代码统一使用 snake_case。
train_DQN = train_dqn
