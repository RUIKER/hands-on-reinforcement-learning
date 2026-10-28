import torch

# =============================================================================
# HER 超参
# =============================================================================
# HER（Hindsight Experience Replay）不是新算法，是 replay buffer 的改造：
# 采样时以 HER_RATIO 的概率把目标 g 换成同一条轨迹里 t 之后真实到达过的位置，
# 再用新目标重算奖励和 done。失败轨迹因此变成"以失败点为目标的成功轨迹"。
#
# 默认值来自《动手学强化学习》第 19 章，改之前先复现一遍。

ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
HIDDEN_DIM = 128

# 状态是 [x, y, goal_x, goal_y]，动作是二维平面位移。
STATE_DIM = 4
ACTION_DIM = 2
ACTION_BOUND = 1.0

GAMMA = 0.98
TAU = 0.005
# 行为策略的探索噪声（作用在归一化动作上，范围 [-1, 1]）
SIGMA = 0.1

# ------------------------------------------------------------------ 训练量
NUM_EPISODES = 2000
# buffer 里攒够这么多条**轨迹**才开始更新。
# off-policy 的过渡在 Pendulum 里是 1000 条 transition，
# 这里单位是整条轨迹，差别很大。
MINIMAL_EPISODES = 200
# 每采一条轨迹，做多少次梯度更新
N_TRAIN = 20
BATCH_SIZE = 256
# buffer 存的是轨迹对象，一条轨迹最多 MAX_STEPS 个 transition
BUFFER_SIZE = 10000

# ------------------------------------------------------------------ 环境
MAX_STEPS = 50
DISTANCE_THRESHOLD = 0.15

# ------------------------------------------------------------------ HER 开关
# 消融实验就靠这两个开关，别改代码结构。
USE_HER = True
# 0.0 = 纯 DDPG（对照组）；1.0 = 全部用假目标
HER_RATIO = 0.8

# 目标采样策略：'future' / 'final' / 'episode'
# future：从 t 之后的时刻取（教程默认，效果最好）
# final ：只取轨迹最后一个状态，多样性差
# episode：整条轨迹任意时刻，会取到"已经达成"的平凡目标
GOAL_STRATEGY = 'future'

# ------------------------------------------------------------------ 评估
# 固定评估集：同一批 (起点=原点, 目标) 组合，所有算法 / 所有 seed / 所有评估点共用。
# 指标以**成功率**为主——稀疏奖励下回报几乎全是 -100 或 0，看不出差别。
EVAL_SEED = 10000
EVAL_GOALS = 20
EVAL_INTERVAL = 100   # 每多少个训练回合评估一次

SEED = 0
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
