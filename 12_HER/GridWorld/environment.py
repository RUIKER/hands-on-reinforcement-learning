"""二维网格世界：一个目标导向（goal-conditioned）的最小环境。

和 Pendulum 这类普通环境的关键区别：**奖励依赖目标**。

    状态 s = [x, y, goal_x, goal_y]
    目标 g = φ(s) = s[:2]          （即"当前位置"就是"实际到达的目标"）
    奖励   r(s, a, g) = 0  if ‖φ(s') − g‖ ≤ 阈值  else −1

注意 φ 只取位置那两维：状态里包含目标，但"实际到达的目标"只由位置决定。
HER 的全部操作都建立在这个 φ 上。

接口按 Gymnasium 拆成 terminated / truncated：
    terminated=True  到达目标（这个才是 HER 重算时要改的）
    truncated=True   达到步数上限（和时间有关，与目标无关，HER 不碰）
教程里把两者合成一个 done，直接搬过来会在 relabel 时埋 bug。
"""

import numpy as np


class GridWorldEnv:
    """二维平面上从原点走到目标点的任务。"""

    def __init__(self,
                 size=5.0,
                 max_steps=50,
                 distance_threshold=0.15,
                 action_bound=1.0,
                 goal_low=(3.5, 3.5),
                 goal_high=(4.5, 4.5)):
        self.size = size
        self.max_steps = max_steps
        self.distance_threshold = distance_threshold
        self.action_bound = action_bound
        self.goal_low = np.asarray(goal_low, dtype=np.float64)
        self.goal_high = np.asarray(goal_high, dtype=np.float64)

        self.observation_dim = 4
        self.action_dim = 2
        self.action_low = -action_bound * np.ones(self.action_dim)
        self.action_high = action_bound * np.ones(self.action_dim)

        # 用独立的 Generator，避免和 numpy 全局随机状态互相污染。
        # reset(seed=None) 时沿用上一条随机流，和 Gymnasium 的约定一致。
        self.rng = np.random.default_rng()
        self.state = np.zeros(2, dtype=np.float64)
        self.goal = np.zeros(2, dtype=np.float64)
        self.count = 0

    # ------------------------------------------------------------------ 目标
    @property
    def achieved_goal(self):
        """φ(s)：从状态里提取"实际到达的目标"。"""
        return self.state.copy()

    @property
    def desired_goal(self):
        """当前任务的目标。HER 会用它被替换成假目标。"""
        return self.goal.copy()

    def _observation(self):
        return np.concatenate([self.state, self.goal])

    def compute_reward(self, achieved_goal, desired_goal):
        """按 (实际到达, 目标) 算奖励。

        HER 重标注时直接调这个，不要另写一份逻辑——
        两份实现迟早会不一致，而且这就是评分函数本身。
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if distance <= self.distance_threshold else -1.0

    def is_success(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return bool(distance <= self.distance_threshold)

    # ---------------------------------------------------------------- 交互
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # 目标固定在右上区域：[3.5, 4.5] × [3.5, 4.5]
        self.goal = self.rng.uniform(self.goal_low, self.goal_high)
        self.state = np.zeros(2, dtype=np.float64)
        self.count = 0
        return self._observation(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64),
                         -self.action_bound, self.action_bound)
        self.state = np.clip(self.state + action, 0.0, self.size)
        self.count += 1

        reward = self.compute_reward(self.state, self.goal)
        terminated = self.is_success(self.state, self.goal)
        truncated = self.count >= self.max_steps
        return self._observation(), reward, terminated, truncated, {}

    def test_set_goal(self, goal):
        """评测用：把目标设成指定的一个，方便固定评估集。"""
        self.goal = np.asarray(goal, dtype=np.float64).copy()
        self.state = np.zeros(2, dtype=np.float64)
        self.count = 0
        return self._observation(), {}


def make_environment(**kwargs):
    env = GridWorldEnv(**kwargs)
    return env, env.observation_dim, env.action_dim, env.action_low, env.action_high
