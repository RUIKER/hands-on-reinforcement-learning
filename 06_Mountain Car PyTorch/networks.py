"""Actor and critic networks."""

from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal


class Actor(nn.Module):
    """裁剪高斯策略，动作范围 [-1, 1]。

    保留 NumPy 版本的探索方式。裁剪后的分布在边界含概率质量，
    不能将其当作普通高斯分布；trainer 对裁剪动作使用高斯密度是有偏近似。
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_size, action_size)
        # 初始标准差为 1.0；学习时在 distribution() 中限制其范围。
        self.log_std = nn.Parameter(torch.full((action_size,), 0.0))

    def distribution(self, states: torch.Tensor) -> Normal:
        hidden = self.network(states)
        mean = self.mean_layer(hidden)
        standard_deviation = self.log_std.clamp(-5.0, 1.0).exp()
        return Normal(mean, standard_deviation)

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_action = distribution.rsample()
        action = raw_action.clamp(-1.0, 1.0)
        # 返回原始采样的高斯 log_prob；不是裁剪后动作分布的精确对数概率。
        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        return action, log_probability, distribution.entropy().sum(dim=-1)


class Critic(nn.Module):
    """State-value network V(s)."""

    def __init__(self, state_size: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.network(states).squeeze(-1)
