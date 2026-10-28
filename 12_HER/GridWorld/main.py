"""DDPG + HER 在二维网格世界上的入口。

运行：
    python main.py

消融只改 config.py：
    USE_HER = False        纯 DDPG 对照组
    HER_RATIO = 0.5        调整假目标比例
    GOAL_STRATEGY = 'final' / 'episode'
"""

import random

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBufferTrajectory
from trainer import evaluate_success, make_eval_goals, train_her_agent
from visualization import plot_training_results


def set_seed(seed):
    """固定 Python / NumPy / PyTorch 的随机源。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env():
    return make_environment(
        max_steps=config.MAX_STEPS,
        distance_threshold=config.DISTANCE_THRESHOLD,
        action_bound=config.ACTION_BOUND,
    )


def main():
    set_seed(config.SEED)

    env, state_dim, action_dim, _, _ = make_env()
    # 评估用独立环境，避免 test_set_goal 打断训练中的 episode。
    eval_env, _, _, _, _ = make_env()

    agent = DDPG(
        state_dim=state_dim,
        hidden_dim=config.HIDDEN_DIM,
        action_dim=action_dim,
        action_bound=config.ACTION_BOUND,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        sigma=config.SIGMA,
        tau=config.TAU,
        gamma=config.GAMMA,
        device=config.DEVICE,
    )

    replay_buffer = ReplayBufferTrajectory(
        capacity=config.BUFFER_SIZE,
        env=env,
        use_her=config.USE_HER,
        her_ratio=config.HER_RATIO,
        goal_strategy=config.GOAL_STRATEGY,
    )

    eval_goals = make_eval_goals(
        config.EVAL_SEED, config.EVAL_GOALS,
        env.goal_low, env.goal_high,
    )

    return_list, success_history = train_her_agent(
        env=env,
        eval_env=eval_env,
        agent=agent,
        replay_buffer=replay_buffer,
        num_episodes=config.NUM_EPISODES,
        minimal_episodes=config.MINIMAL_EPISODES,
        n_train=config.N_TRAIN,
        batch_size=config.BATCH_SIZE,
        eval_interval=config.EVAL_INTERVAL,
        eval_goals=eval_goals,
        seed=config.SEED,
        use_her=config.USE_HER,
    )

    final_success = evaluate_success(eval_env, agent, eval_goals)
    print('USE_HER=%s  HER_RATIO=%.2f  STRATEGY=%s' % (
        config.USE_HER, config.HER_RATIO, config.GOAL_STRATEGY
    ))
    print('最后 10 回合训练平均回报: %.3f' % np.mean(return_list[-10:]))
    print('最终固定目标成功率: %.3f (%d goals)' % (
        final_success, config.EVAL_GOALS
    ))

    plot_training_results(
        return_list, success_history,
        env_name='GridWorld', use_her=config.USE_HER,
        num_eval_goals=config.EVAL_GOALS,
    )


if __name__ == '__main__':
    main()
