"""DQN 最小组件：离散动作 Q 网络、经验回放与目标网络。

本文件提供可导入的组件；完整训练和变体对照见 03_Double DQN、Dueling DQN。
批数据的 dones 表示 MDP 终止，时间限制截断不应置为 1。
"""

import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ReplayBuffer:
    """固定容量的 FIFO 回放池，均匀抽样且不放回。"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    """单隐藏层网络，输出形状为 [batch_size, action_dim]。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, states):
        return self.fc2(F.relu(self.fc1(states)))


class DQN:
    """使用硬同步目标网络的 DQN。"""

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate,
                 gamma, epsilon, target_update, device):
        if target_update <= 0:
            raise ValueError("target_update 必须为正整数")
        self.action_dim = action_dim
        self.device = device
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state).argmax(dim=1).item()

    def update(self, transition_dict):
        """对一个 batch 执行 TD 更新；返回标量 loss 用于诊断。"""
        def tensor(key, dtype=torch.float32):
            return torch.as_tensor(np.asarray(transition_dict[key]),
                                   dtype=dtype, device=self.device)

        states = tensor("states")
        actions = tensor("actions", torch.long).view(-1, 1)
        rewards = tensor("rewards").view(-1, 1)
        next_states = tensor("next_states")
        dones = tensor("dones").view(-1, 1)
        with torch.no_grad():
            next_values = self.target_q_net(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * next_values * (1 - dones)
        loss = F.mse_loss(self.q_net(states).gather(1, actions), targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.count += 1
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        return loss.item()
