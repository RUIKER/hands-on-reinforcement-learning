"""单次 HER 实验：参数化 HER 开关 / 比例 / 策略 / seed，结果写 JSON，不弹窗。

用法：
    python run_experiment.py --seed 0 --her 1 --out results/raw/her_seed0.json
    python run_experiment.py --seed 0 --her 0 --out results/raw/noher_seed0.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBufferTrajectory
from trainer import evaluate_success, make_eval_goals, train_her_agent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--her', type=int, default=1, help='1=开启 HER, 0=纯 DDPG')
    parser.add_argument('--ratio', type=float, default=config.HER_RATIO)
    parser.add_argument('--strategy', default=config.GOAL_STRATEGY)
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES)
    parser.add_argument('--eval-interval', type=int, default=50)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env():
    return make_environment(
        max_steps=config.MAX_STEPS,
        distance_threshold=config.DISTANCE_THRESHOLD,
        action_bound=config.ACTION_BOUND,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else config.DEVICE

    env, state_dim, action_dim, _, _ = make_env()
    eval_env, _, _, _, _ = make_env()
    agent = DDPG(
        state_dim, config.HIDDEN_DIM, action_dim, config.ACTION_BOUND,
        config.ACTOR_LR, config.CRITIC_LR, config.SIGMA, config.TAU,
        config.GAMMA, device,
    )
    buffer = ReplayBufferTrajectory(
        config.BUFFER_SIZE, env, use_her=bool(args.her),
        her_ratio=args.ratio, goal_strategy=args.strategy,
    )
    eval_goals = make_eval_goals(
        config.EVAL_SEED, config.EVAL_GOALS, env.goal_low, env.goal_high
    )

    started = time.perf_counter()
    returns, history = train_her_agent(
        env, eval_env, agent, buffer, args.episodes,
        config.MINIMAL_EPISODES, config.N_TRAIN, config.BATCH_SIZE,
        args.eval_interval, eval_goals, seed=args.seed,
        use_her=bool(args.her),
    )
    wall = time.perf_counter() - started
    final = evaluate_success(eval_env, agent, eval_goals)

    payload = {
        'her': bool(args.her), 'ratio': args.ratio,
        'strategy': args.strategy, 'seed': args.seed,
        'episodes': args.episodes, 'device': str(device),
        'returns': [float(r) for r in returns],
        'success_history': [[int(e), float(s)] for e, s in history],
        'final_success': float(final),
        'wall_clock_sec': float(wall),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    print('RESULT', json.dumps({
        'her': payload['her'], 'ratio': args.ratio, 'strategy': args.strategy,
        'seed': args.seed, 'final_success': final, 'wall': round(wall, 1),
    }), flush=True)


if __name__ == '__main__':
    main()
