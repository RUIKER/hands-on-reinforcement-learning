"""训练结果可视化：回报曲线 + 固定评估集上的成功率曲线。"""

import numpy as np
import matplotlib.pyplot as plt

from rl_utils import moving_average


def plot_training_results(return_list, success_history, env_name='GridWorld',
                          use_her=True, window_size=9, num_eval_goals=20,
                          save_path=None):
    episodes = np.arange(1, len(return_list) + 1)
    label = 'DDPG + HER' if use_her else 'DDPG (no HER)'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(episodes, return_list, color='#1f77b4',
                 alpha=0.30, linewidth=1, label='raw return')
    axes[0].plot(episodes, moving_average(return_list, window_size),
                 color='#1f77b4', linewidth=2,
                 label='moving average (%d)' % window_size)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('%s on %s' % (label, env_name))
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    if success_history:
        xs, ys = zip(*success_history)
        axes[1].plot(xs, ys, color='#2ca02c', marker='o', markersize=3)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Success rate')
    axes[1].set_title('Evaluation success rate (%d fixed goals)'
                      % num_eval_goals)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
