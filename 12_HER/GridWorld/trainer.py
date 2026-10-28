"""HER 的训练循环与评估。

和 09_DDPG/Pendulum/trainer.py 的关键区别：
1. 训练数据按"整条轨迹"收集和入 buffer，不是每个 env step 存一条 transition；
2. 每个 episode 结束后做 N_TRAIN 次更新，而不是每个 env step 更新一次；
3. 评估指标是成功率（固定一组目标），因为稀疏奖励下回报几乎只有 -50 / 0 两档。
"""

import numpy as np
from tqdm import tqdm

from rl_utils import Trajectory


def make_eval_goals(seed, num_goals, goal_low, goal_high):
    """生成固定评估集；所有算法、seed、评估点共用同一组目标。"""
    rng = np.random.default_rng(seed)
    return [rng.uniform(goal_low, goal_high) for _ in range(num_goals)]


def evaluate_success(env, agent, goals):
    """在固定目标上跑确定性策略，返回成功率。"""
    successes = []
    for goal in goals:
        state, _ = env.test_set_goal(goal)
        done = False
        success = False
        while not done:
            action = agent.take_action(state, explore=False)
            state, _, terminated, truncated, _ = env.step(action)
            if terminated:
                success = True
            done = terminated or truncated
        successes.append(1.0 if success else 0.0)
    return float(np.mean(successes))


def train_her_agent(env, eval_env, agent, replay_buffer, num_episodes,
                    minimal_episodes, n_train, batch_size, eval_interval,
                    eval_goals, seed=None, use_her=None):
    """DDPG + HER 的主循环。

    参数单位提醒：
    - minimal_episodes 是"轨迹条数"，不是 transition 条数；
    - n_train 是每条轨迹之后做多少次梯度更新。
    """
    return_list = []
    success_history = []

    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # 只在第一个 episode 播种，之后沿用同一条随机流。
            # 若每回合都 reset(seed=seed+episode)，不同 seed 的目标序列
            # 会变成彼此错位的同一条序列（实测重合比例 100%），seed 不独立。
            if seed is not None and episode == 0:
                state, _ = env.reset(seed=seed)
            else:
                state, _ = env.reset()

            traj = Trajectory(state)
            done = False
            episode_return = 0.0

            # ① 先完整跑完一条轨迹，期间不更新。
            while not done:
                action = agent.take_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # 只把 terminated 写入轨迹；truncated 不参与 Bellman 终止，
                # 但它是上面的 done，用来结束采样循环。
                traj.store_step(action, next_state, reward, terminated)
                state = next_state
                episode_return += reward

            # ② 整条轨迹入 buffer。
            replay_buffer.add_trajectory(traj)
            return_list.append(episode_return)

            # ③ 攒够轨迹后，每条轨迹做 n_train 次更新。
            if replay_buffer.size() >= minimal_episodes:
                for _ in range(n_train):
                    batch = replay_buffer.sample(
                        batch_size, use_her=use_her
                    )
                    agent.update(batch)

            # ④ 定期在固定目标上评估成功率。
            if eval_interval > 0 and (
                episode == 0 or (episode + 1) % eval_interval == 0
            ):
                success = evaluate_success(eval_env, agent, eval_goals)
                success_history.append((episode + 1, success))
                progress.set_postfix({
                    'episode': episode + 1,
                    'success': '%.2f' % success,
                    'return': '%.2f' % np.mean(return_list[-10:]),
                })
            elif (episode + 1) % 10 == 0:
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.2f' % np.mean(return_list[-10:]),
                })
            progress.update(1)

    return return_list, success_history
