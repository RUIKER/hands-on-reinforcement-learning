"""Cliff Walking 训练入口；支持 Q-learning 与 SARSA。"""

import argparse
import random

from config import (
    DISCOUNT_FACTOR, LEARNING_RATE, MAP_HEIGHT, MAP_LENGTH,
    NUM_ACTIONS, NUM_EPISODES, PRINT_INTERVAL,
)
from environment import Env
from q_learning import QTable
from sarsa import SarsaTable


def cliff_walk(algorithm="q-learning", episodes=NUM_EPISODES, seed=0):
    """返回逐回合回报；seed 控制探索动作，SARSA 复用已采样的下一动作。"""
    if episodes <= 0:
        raise ValueError("episodes 必须为正整数")
    algorithms = {"q-learning": QTable, "sarsa": SarsaTable}
    if algorithm not in algorithms:
        raise ValueError(f"未知算法：{algorithm}")
    random.seed(seed)
    env = Env(length=MAP_LENGTH, height=MAP_HEIGHT)
    table = algorithms[algorithm](
        length=MAP_LENGTH, height=MAP_HEIGHT, actions=NUM_ACTIONS,
        alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR,
    )
    returns = []
    for episode in range(episodes):
        env.reset()
        state = [0, 0]
        action = table.take_action(*state)
        episode_return = 0
        while True:
            reward, next_state, terminated = env.step(action)
            next_action = None if terminated else table.take_action(*next_state)
            table.update(action, state, next_state, reward, terminated, next_action)
            episode_return += reward
            if terminated:
                break
            # SARSA 的 TD 目标必须对应下一步实际执行的动作。
            state, action = next_state, next_action
        returns.append(episode_return)
        if episode % PRINT_INTERVAL == 0:
            print(f"Episode: {episode}, Score: {episode_return}")
    return returns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("q-learning", "sarsa"), default="q-learning")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cliff_walk(args.algorithm, args.episodes, args.seed)


if __name__ == "__main__":
    main()
