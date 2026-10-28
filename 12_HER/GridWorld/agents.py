"""DDPG agent。HER 只改 replay buffer，不改算法本身。

与 09_DDPG/Pendulum/agents.py 的结构相同，另外修了 practice.py 里的两个问题：
1. take_action 末尾强制 clip 到动作边界。
   practice.py 只加噪声不裁剪，buffer 里会存到越界动作（实测最大 1.313），
   而环境真正执行的是被 clip 后的值，critic 学到的是不一致的 (s, a)。
2. target 网络的计算放进 torch.no_grad()。
   practice.py 的 q_targets 带梯度，target 网络会累积用不到的 .grad。
"""

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPG:
    """确定性 actor + Q critic + 两个 target 网络。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, sigma, tau, gamma, device):
        self.actor = Actor(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)
        self.target_actor = Actor(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.target_critic = Critic(
            state_dim, hidden_dim, action_dim
        ).to(device)

        # target 网络初始时复制 online 网络，之后只做软更新。
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        self.action_dim = action_dim
        self.action_bound = float(action_bound)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.device = device

    def take_action(self, state, explore=True):
        """explore=True 时加高斯噪声；评估时传 False。"""
        state_t = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        if explore:
            action = action + self.sigma * np.random.randn(self.action_dim)

        # 关键：buffer 里存的必须是环境实际执行的 action。
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action.astype(np.float32)

    def soft_update(self, online_net, target_net):
        """target <- (1 - tau) * target + tau * online。"""
        with torch.no_grad():
            for target_param, online_param in zip(
                target_net.parameters(), online_net.parameters()
            ):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(self.tau * online_param)

    def update(self, batch):
        """一个 batch 的 DDPG 更新；batch 是 rl_utils 返回的 dict。"""
        states = torch.as_tensor(
            batch['states'], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch['actions'], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch['rewards'], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.as_tensor(
            batch['next_states'], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch['dones'], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # Bellman target 不需要梯度：只用来做回归目标。
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        critic_loss = F.mse_loss(self.critic(states, actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 确定性策略梯度：最大化 Q(s, mu(s))，所以 loss 取负号。
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
