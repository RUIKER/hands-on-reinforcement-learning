"""HER 实验用的 actor / critic。

结构与 09_DDPG/Pendulum/networks.py 一致，只是动作空间是二维、用标量
ACTION_BOUND 缩放，所以不需要 action_low/action_high 两个 buffer。

和 DDPG 的差别只有一个：输入维度从 3 变成 4。
状态是 [x, y, goal_x, goal_y]，网络并不知道哪两维是目标，它只是多看到两个数。
"""

import torch
import torch.nn.functional as F


class Actor(torch.nn.Module):
    """确定性策略 mu(s)：输入 [x, y, gx, gy]，输出二维动作。"""

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = float(action_bound)

    def forward(self, states):
        hidden = F.relu(self.fc1(states))
        hidden = F.relu(self.fc2(hidden))
        # tanh 把输出压到 [-1, 1]，再乘以动作上限。
        return torch.tanh(self.fc3(hidden)) * self.action_bound


class Critic(torch.nn.Module):
    """Q(s, a)：把状态（含目标）和动作拼在一起，输出一个标量。"""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, states, actions):
        features = torch.cat([states, actions], dim=1)
        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        return self.fc3(features)
