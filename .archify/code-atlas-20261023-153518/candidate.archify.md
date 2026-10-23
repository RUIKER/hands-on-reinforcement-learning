# 强化学习代码地图：每个文件与每个代码块 {#code-atlas}

这份代码地图覆盖 11 个算法目录、112 个 Python 文件和 541 个代码块条目；每个文件按导入与常量、类、方法、函数、入口拆开，逐块说明输入、输出和关键步骤。

- 01–03 是价值型方法：Q 表、DQN、Double DQN 与 Dueling DQN。
- 04–08 是策略梯度方法：REINFORCE、Actor-Critic、TRPO 与 PPO。
- 09–11 是连续控制方法：DDPG、TD3 与 SAC，均使用 replay buffer。
- 03–11 共用 config、environment、agent、networks、trainer、main 的拆分方式；01 与 02 是简化结构。

@answer scope="11 个算法目录 · 当前工作区文件 · 行号为当前文件行号"
@meta locale=zh-CN quality_profile=showcase

## 代码库如何组织？ {#overview span=12}

@facts id=atlas-facts
| 算法目录 | 11 |
| Python 文件 | 112 |
| 代码块条目 | 541 |
| 阅读分支 | 3 |

@tree id=repo-tree title="01–11 目录结构"
| root | 强化学习代码库 | - |
| a01 | 01_Cliff Walking | root |
| a02 | 02_DQN | root |
| a03 | 03_Double DQN、Dueling DQN | root |
| a04 | 04_策略梯度算法_Reinforce | root |
| a05 | 05_Mountain Car | root |
| a06 | 06_Mountain Car PyTorch | root |
| a07 | 07_TRPO | root |
| a07_cart | CartPole | a07 |
| a07_pend | Pendulum | a07 |
| a08 | 08_PPO | root |
| a08_cart | CartPole | a08 |
| a08_pend | Pendulum | a08 |
| a09 | 09_DDPG | root |
| a09_cart | CartPole | a09 |
| a09_pend | Pendulum | a09 |
| a10 | 10_TD3 | root |
| a10_pend | Pendulum | a10 |
| a11 | 11_SAC | root |
| a11_pend | Pendulum | a11 |

| 目录 | 分支 | 入口 |
| --- | --- | --- |
| `01_Cliff Walking` | 价值型 | `train.py` |
| `02_DQN` | 价值型 | `practice.py` |
| `03_Double DQN、Dueling DQN` | 价值型 | `main.py` |
| `04_策略梯度算法_Reinforce` | 策略梯度 | `main.py` |
| `05_Mountain Car` | 策略梯度 | `main.py` |
| `06_Mountain Car PyTorch` | 策略梯度 | `main.py` |
| `07_TRPO` | 策略梯度 | `*/main.py` |
| `08_PPO` | 策略梯度 | `*/main.py` |
| `09_DDPG` | 连续控制 | `*/main.py` |
| `10_TD3` | 连续控制 | `Pendulum/main.py` |
| `11_SAC` | 连续控制 | `Pendulum/main.py` |

@callout level=note title="读取范围"
按 2026-10-23 的当前工作区读取；01–11 的重命名与文档尚未提交，行号只对应当前文件，不对应 Git HEAD。

## 01_Cliff Walking：表格型方法如何落到文件？ {#algo01 span=12}

实现 Cliff Walking 表格型强化学习，包含 Sarsa 与 Q-learning 两种 TD 控制算法、地图环境及统一训练脚本。 本目录共 6 个 Python 文件、26 个代码块条目。

**`01_Cliff Walking/config.py`** · 10 行 · 集中定义 Cliff Walking 训练使用的地图尺寸、动作数、回合数和超参数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0001} | 导入与常量 | 1-10 | 无 import；定义地图宽 12、高 4、4 个动作、3000 回合，以及学习率 0.1、折扣因子 0.9、探索率 0.1 和打印间隔 20。 |

**`01_Cliff Walking/environment.py`** · 56 行 · 实现 Cliff Walking 网格环境的状态、动作转移、悬崖/终点奖励和文本渲染。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0002} | 导入与常量 | 1-2 | 导入 random 和 time，供环境中的随机行为与渲染帧间隔使用。 |
| 类 {#blk0003} | `Env` | 5-56 | Env 保存地图宽高与智能体坐标，提供渲染、单步转移和复位接口，供训练循环交互。 |
| 方法 {#blk0004} | `Env.__init__` | 8-14 | 接收 length、height，保存地图尺寸，并将智能体坐标 (x, y) 初始化为起点 (0, 0)。 |
| 方法 {#blk0005} | `Env.render` | 16-29 | 按行构造地图字符（S 起点、x 悬崖、T 终点、. 空地），用 o 标出 (x, y)，打印后光标上移并按 frames 休眠形成动画。 |
| 方法 {#blk0006} | `Env.step` | 31-51 | 按动作 0/1/2/3 对应 (0,1)/(0,-1)/(-1,0)/(1,0) 更新坐标并裁剪边界；默认奖励 -1，落在第 0 行非起点处终止，非终点视为悬崖奖励 -100，返回 (reward, [x, y], terminal)。 |
| 方法 {#blk0007} | `Env.reset` | 53-56 | 将智能体坐标重置到起点 (0, 0)，不返回状态。 |

**`01_Cliff Walking/main.py`** · 5 行 · 程序入口，从 train 导入 cliff_walk 并在直接运行时启动训练。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0008} | 导入与常量 | 1-1 | 从 train 模块导入 cliff_walk 训练函数。 |
| 入口 {#blk0009} | if __name__ == '__main__' | 4-5 | 仅在脚本被直接执行时调用 cliff_walk()，被导入时不启动训练。 |

**`01_Cliff Walking/q_learning.py`** · 49 行 · 实现表格型 Q-learning 的 Q 表、epsilon-greedy 动作选择和时序差分更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0010} | 导入与常量 | 1-3 | 导入 random 以及 config 中的固定探索率 EXPLORATION_RATE。 |
| 类 {#blk0011} | `QTable` | 5-49 | QTable 用一维列表存储 Q(s, a)，负责 epsilon-greedy 选动作、取状态最大 Q 值以及 Q-learning 更新。 |
| 方法 {#blk0012} | `QTable.__init__` | 8-15 | 按 actions × length × height 创建全零一维 Q 表，并保存动作数、地图尺寸、学习率 alpha 和折扣因子 gamma。 |
| 方法 {#blk0013} | `QTable._index` | 17-19 | 把动作 a 与状态 (x, y) 映射为一维下标 a\*height\*length + x\*length + y。 |
| 方法 {#blk0014} | `QTable._epsilon` | 21-23 | 返回 config 中的固定探索率 EXPLORATION_RATE。 |
| 方法 {#blk0015} | `QTable.take_action` | 25-33 | 以 epsilon 概率随机选择动作，否则读取 (x, y) 下所有动作的 Q 值并返回最大值对应动作，并列时取第一个。 |
| 方法 {#blk0016} | `QTable.max_q` | 35-37 | 收集状态 (x, y) 在全部动作上的 Q 值并返回最大值。 |
| 方法 {#blk0017} | `QTable.update` | 39-49 | 读取 Q(s0, a)；非终止时用 r + gamma\*max_a'Q(s1, a')、终止时用 r 作为目标，按 alpha 做增量更新；next_action 参数未使用。 |

**`01_Cliff Walking/sarsa.py`** · 57 行 · 实现表格型 SARSA 的 Q 表、epsilon-greedy 动作选择和 on-policy 时序差分更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0018} | 导入与常量 | 1-3 | 导入 random 以及 config 中的固定探索率 EXPLORATION_RATE。 |
| 类 {#blk0019} | `SarsaTable` | 6-57 | SarsaTable 用一维列表存储 Q(s, a)，提供 epsilon-greedy 选动作与使用实际下一动作的 SARSA 更新。 |
| 方法 {#blk0020} | `SarsaTable.__init__` | 9-16 | 按 actions × length × height 创建全零一维 Q 表，并保存动作数、地图尺寸、学习率 alpha 和折扣因子 gamma。 |
| 方法 {#blk0021} | `SarsaTable._index` | 18-20 | 把动作 action 与状态 (x, y) 映射为一维下标 action\*height\*length + x\*length + y。 |
| 方法 {#blk0022} | `SarsaTable._epsilon` | 22-24 | 返回 config 中的固定探索率 EXPLORATION_RATE。 |
| 方法 {#blk0023} | `SarsaTable.take_action` | 26-37 | 以 epsilon 概率随机选择动作，否则读取状态 (x, y) 的全部动作 Q 值并返回最大值对应动作。 |
| 方法 {#blk0024} | `SarsaTable.update` | 39-57 | 读取 Q(s, a)；终止时目标为 reward，非终止时要求 next_action 并用 reward + gamma\*Q(s', next_action) 计算目标，再按 alpha 更新。 |

**`01_Cliff Walking/train.py`** · 47 行 · 串联环境与表格算法，执行多回合 epsilon-greedy 训练并定期打印回合累计奖励。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0025} | 导入与常量 | 1-13 | 导入 config 超参数、Env 以及 QTable（默认别名为 AlgorithmTable）；SARSA 的导入语句被注释，切换后即可训练 SARSA。 |
| 函数 {#blk0026} | `cliff_walk` | 16-47 | 创建环境与算法表，循环 NUM_EPISODES 个回合；每步选择动作、调用 env.step，非终止时预选下一动作，调用 table.update，累计奖励并推进状态；每 PRINT_INTERVAL 回合打印分数，回合结束后 env.reset。 |

## 02_DQN：最小 DQN 练习由哪些块组成？ {#algo02 span=12}

提供 DQN 的经验回放、Q 网络和智能体更新实现，并附带在线/离线训练循环、滑动平均与优势估计工具。 本目录共 2 个 Python 文件、23 个代码块条目。

**`02_DQN/practice.py`** · 76 行 · 汇集 DQN 练习组件：经验回放池、单隐藏层 Q 网络和 DQN 智能体的动作选择与学习更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0027} | 导入与常量 | 1-9 | 导入 random、gymnasium、numpy、collections、tqdm、torch、torch.nn.functional、matplotlib 和 rl_utils，为回放池、网络与可视化准备依赖。 |
| 类 {#blk0028} | `ReplayBuffer` | 11-25 | ReplayBuffer 用 deque 保存固定容量的经验五元组，支持追加、随机采样和读取当前大小。 |
| 方法 {#blk0029} | `ReplayBuffer.__init__` | 13-14 | 用最大长度为 capacity 的 collections.deque 创建先进先出回放池，超出容量时自动丢弃最旧经验。 |
| 方法 {#blk0030} | `ReplayBuffer.add` | 16-17 | 把 (state, action, reward, next_state, done) 五元组追加到回放池尾部。 |
| 方法 {#blk0031} | `ReplayBuffer.sample` | 19-22 | 随机抽取 batch_size 条经验并用 zip(\*transitions) 解包，返回 numpy 状态/下一状态数组以及动作、奖励、done 元组。 |
| 方法 {#blk0032} | `ReplayBuffer.size` | 24-25 | 返回回放池当前保存的经验条数。 |
| 类 {#blk0033} | `Qnet` | 27-36 | Qnet 是单隐藏层多层感知机，将状态映射为每个离散动作的 Q 值。 |
| 方法 {#blk0034} | `Qnet.__init__` | 29-32 | 注册 state_dim→hidden_dim 的 fc1 与 hidden_dim→action_dim 的 fc2 两个线性层。 |
| 方法 {#blk0035} | `Qnet.forward` | 34-36 | 对输入先做 fc1 线性变换和 ReLU 激活，再经 fc2 输出各动作的 Q 值。 |
| 类 {#blk0036} | `DQN` | 38-76 | DQN 维护在线网络、目标网络与 Adam 优化器，提供 epsilon-greedy 动作选择和基于 TD 目标的批量更新。 |
| 方法 {#blk0037} | `DQN.__init__` | 40-49 | 保存动作维度、gamma、epsilon、目标网络更新周期和 device；构建在线/目标 Qnet 并用 Adam 优化在线网络参数，更新计数器 count 初始化为 0。 |
| 方法 {#blk0038} | `DQN.take_action` | 51-57 | 以 epsilon 概率随机返回一个动作，否则将 state 转为 float tensor 送入 q_net，并返回 Q 值最大的动作编号。 |
| 方法 {#blk0039} | `DQN.update` | 59-76 | 把 transition_dict 批量转为 tensor（当前实现中 states 实际取自 rewards 字段），用 gather 取 Q(s,a)，用目标网络计算下一状态最大 Q，构造 r + gamma\*maxQ\*(1-done)，计算 MSE 后反向传播并更新参数；每 target_update 次把在线网络参数复制到目标网络并递增 count。 |

**`02_DQN/rl_utils.py`** · 90 行 · 提供 DQN/策略梯度常用的经验回放、滑动平均、在线与离线训练循环及优势计算工具。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0040} | 导入与常量 | 1-5 | 导入 tqdm、numpy、torch、collections 和 random，供训练进度、数值计算、回放池与采样使用。 |
| 类 {#blk0041} | `ReplayBuffer` | 7-20 | ReplayBuffer 用 deque 保存经验五元组，支持 add、随机 sample 和 size 查询。 |
| 方法 {#blk0042} | `ReplayBuffer.__init__` | 8-9 | 按给定 capacity 创建有最大长度的 deque，作为先进先出经验池。 |
| 方法 {#blk0043} | `ReplayBuffer.add` | 11-12 | 将 (state, action, reward, next_state, done) 追加到回放池。 |
| 方法 {#blk0044} | `ReplayBuffer.sample` | 14-17 | 随机抽取 batch_size 条经验并用 zip(\*transitions) 解包，返回 numpy 状态/下一状态数组以及动作、奖励、done 元组。 |
| 方法 {#blk0045} | `ReplayBuffer.size` | 19-20 | 返回回放池中的经验条数。 |
| 函数 {#blk0046} | `moving_average` | 22-28 | 先算序列前缀和，再用差分求中间完整窗口的均值，并分别用累积和计算首尾不完整窗口的均值，最后拼接返回滑动平均序列。 |
| 函数 {#blk0047} | `train_on_policy_agent` | 30-54 | 把 num_episodes 均分为 10 轮并用 tqdm 显示进度；每回合重置环境，循环 take_action 与 env.step 收集 transition_dict 和累计回报，回合结束后调用 agent.update，记录 return_list 并每 10 回合显示最近 10 次平均回报。 |
| 函数 {#blk0048} | `train_off_policy_agent` | 56-78 | 同样分 10 轮训练；每步把交互结果写入 replay_buffer，当 buffer 大小超过 minimal_size 时采样 batch_size 条组成 transition_dict 调用 agent.update，同时记录每回合回报并每 10 回合显示最近均值。 |
| 函数 {#blk0049} | `compute_advantage` | 81-89 | 将 td_delta 转为 numpy 后逆序递推 advantage = gamma\*lambda\*advantage + delta，再反转顺序并转为 float tensor 返回。 |

## 03_Double DQN、Dueling DQN：三种价值变体如何共用 pipeline？ {#algo03 span=12}

在 Pendulum-v1 离散化动作上统一实现 DQN、Double DQN 与 Dueling DQN，用 config 切换算法、共享经验回放与训练流程。 本目录共 9 个 Python 文件、38 个代码块条目。

**`03_Double DQN、Dueling DQN/agents.py`** · 89 行 · 统一 DQN agent：按 dqn_type 选择普通或 Dueling 网络，并实现 epsilon-greedy 与 TD 更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0050} | 导入与常量 | 1-5 | 引入 numpy 负责随机探索、torch 与 F 负责张量和 MSE 损失，并从 networks 导入 Qnet 与 DuelingQnet。 |
| 类 {#blk0051} | `DQN` | 8-76 | DQN/Double DQN/Dueling DQN 共用的 agent 类，内部由 dqn_type 决定网络结构和下一状态价值的计算方式。 |
| 方法 {#blk0052} | `DQN.__init__` | 10-25 | 按 dqn_type 选择 DuelingQnet 或 Qnet 构建在线网络与目标网络，把在线权重同步给目标网络，创建 Adam 优化器并保存 gamma、epsilon、target_update、device 等超参。 |
| 方法 {#blk0053} | `DQN.take_action` | 27-35 | epsilon-greedy 策略：以 epsilon 概率用 np.random.randint 随机选动作，否则把 state 转成张量送入 q_net，取 argmax 对应动作索引。 |
| 方法 {#blk0054} | `DQN.max_q_value` | 37-39 | 把单个 state 转为 [1, state_dim] 张量输入 q_net，返回所有动作 Q 值中的最大值，供训练时记录 Q 值曲线。 |
| 方法 {#blk0055} | `DQN.update` | 41-76 | 把 transition_dict 转成张量并用 gather 取 Q(s,a)；DoubleDQN 由在线网络 argmax 选动作、目标网络评估该动作，其余变体直接用目标网络 max；再以 r + gamma\*Q'\*(1-done) 为 TD 目标做 MSE 反传，并按 count % target_update 同步目标网络。 |
| 函数 {#blk0056} | `build_agent` | 79-89 | 工厂函数：把别名 VanillaDQN 归一为 DQN，校验算法名只在 DQN/DoubleDQN/DuelingDQN 之内（否则抛 ValueError），返回对应 dqn_type 的 DQN 实例。 |

**`03_Double DQN、Dueling DQN/config.py`** · 29 行 · 集中定义算法选择、超参数、环境名与运行设备的配置文件。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0057} | 导入与常量 | 1-28 | 导入 torch 并定义全局配置：ALGORITHM 手动选择 DQN/DoubleDQN/DuelingDQN，另有 LR、GAMMA、EPSILON、TARGET_UPDATE、NUM_EPISODES、HIDDEN_DIM、BUFFER_SIZE、MINIMAL_SIZE、BATCH_SIZE、SEED、ENV_NAME='Pendulum-v1'、ACTION_DIM 以及按 CUDA 可用性选择的 DEVICE。 |

**`03_Double DQN、Dueling DQN/environment.py`** · 13 行 · 创建 gymnasium 环境，并提供离散动作到连续动作区间的映射。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0058} | 导入与常量 | 1-1 | 导入 gymnasium 用于构造环境。 |
| 函数 {#blk0059} | `make_environment` | 4-7 | 调用 gym.make 创建环境，从 observation_space.shape[0] 读出状态维度，原样带回 action_dim，返回 (env, state_dim, action_dim)。 |
| 函数 {#blk0060} | `dis_to_con` | 10-13 | 把离散动作索引按 action_dim-1 等分，线性映射到 env.action_space 的 [low[0], high[0]] 连续区间，供 Pendulum 执行。 |

**`03_Double DQN、Dueling DQN/main.py`** · 48 行 · 程序入口：装配环境、回放池与 agent，调用训练与绘图流程。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0061} | 导入与常量 | 1-11 | 导入 random/numpy/torch 三个随机源、config 配置，以及 agents.build_agent、environment.make_environment、trainer.train_DQN、visualization.plot_training_results。 |
| 函数 {#blk0062} | `set_seed` | 14-17 | 分别设置 random、numpy、torch 的随机种子，保证实验可复现。 |
| 函数 {#blk0063} | `main` | 20-44 | 按顺序设置种子、创建环境、构造 ReplayBuffer，用 config 中各项超参调用 build_agent 建 agent，再交给 train_DQN 训练，最后把回报列表、max Q 值列表与算法/环境名传给 plot_training_results 绘图。 |
| 入口 {#blk0064} | if __name__ == '__main__' | 47-48 | 脚本直接运行时调用 main()。 |

**`03_Double DQN、Dueling DQN/networks.py`** · 34 行 · 定义普通 Q 网络与 Dueling 结构的 Q 网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0065} | 导入与常量 | 1-2 | 导入 torch 构建网络层，导入 F 使用 relu。 |
| 类 {#blk0066} | `Qnet` | 5-16 | 普通 DQN 网络：输入状态，输出每个离散动作的 Q 值。 |
| 方法 {#blk0067} | `Qnet.__init__` | 7-10 | 定义两层全连接：fc1 从 state_dim 到 hidden_dim，fc2 从 hidden_dim 到 action_dim。 |
| 方法 {#blk0068} | `Qnet.forward` | 12-16 | 先对 fc1 输出做 relu，再经 fc2 得到形状 [batch, action_dim] 的 Q 值矩阵。 |
| 类 {#blk0069} | `DuelingQnet` | 19-34 | Dueling DQN 网络：共享特征层后分叉估计状态价值 V(s) 与动作优势 A(s,a)。 |
| 方法 {#blk0070} | `DuelingQnet.__init__` | 21-25 | 定义共享层 fc1、优势分支 fc_A(hidden_dim→action_dim) 和价值分支 fc_V(hidden_dim→1)。 |
| 方法 {#blk0071} | `DuelingQnet.forward` | 27-34 | features=relu(fc1(x)) 后分别算 advantage 与 value，输出 value + advantage - advantage.mean(dim=1, keepdim=True) 作为 Q 值。 |

**`03_Double DQN、Dueling DQN/practice.py`** · 5 行 · 极简启动脚本，直接复用 main.py 的流程。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0072} | 导入与常量 | 1-1 | 从 main 模块导入 main 函数，本身不定义常量。 |
| 入口 {#blk0073} | if __name__ == '__main__' | 4-5 | 直接运行 main() 开始训练。 |

**`03_Double DQN、Dueling DQN/rl_utils.py`** · 90 行 · 通用强化学习工具：经验回放池、滑动平均、通用训练循环与优势估计。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0074} | 导入与常量 | 1-5 | 导入 tqdm 进度条、numpy、torch、collections（deque）与 random（采样），供回放池和通用训练循环使用。 |
| 类 {#blk0075} | `ReplayBuffer` | 7-20 | 基于 collections.deque 的定长经验回放池，支持写入、随机采样与读取大小。 |
| 方法 {#blk0076} | `ReplayBuffer.__init__` | 8-9 | 用 collections.deque(maxlen=capacity) 创建容量受限的缓冲区。 |
| 方法 {#blk0077} | `ReplayBuffer.add` | 11-12 | 把 (state, action, reward, next_state, done) 五元组追加进缓冲区，满容量时自动丢弃最旧样本。 |
| 方法 {#blk0078} | `ReplayBuffer.sample` | 14-17 | random.sample 抽取 batch_size 条经验，用 zip(\*transitions) 拆成五列，states/next_states 转 np.array 后连同 action、reward、done 一起返回。 |
| 方法 {#blk0079} | `ReplayBuffer.size` | 19-20 | 返回缓冲区当前保存的样本数量。 |
| 函数 {#blk0080} | `moving_average` | 22-28 | 借助 np.cumsum 的差分计算窗口均值，并单独处理首尾不足整窗的部分，返回与输入等长的平滑序列。 |
| 函数 {#blk0081} | `train_on_policy_agent` | 30-54 | 按 10 段 tqdm 跑同策略训练，每回合收集包含 states/actions/rewards/dones 的整条轨迹后调用 agent.update；采用旧版 gym 的 env.reset()/step 返回值约定，返回每回合回报列表。 |
| 函数 {#blk0082} | `train_off_policy_agent` | 56-78 | 按 10 段 tqdm 跑异策略训练，每步写入 replay_buffer，缓冲区超过 minimal_size 后采样 batch 组装 transition_dict 调用 agent.update，返回每回合回报列表。 |
| 函数 {#blk0083} | `compute_advantage` | 81-89 | 把 td_delta detach 成 numpy 后从后往前按 advantage = gamma\*lmbda\*advantage + delta 累加，反转后转成 torch.float 张量返回（GAE 形式）。 |

**`03_Double DQN、Dueling DQN/trainer.py`** · 53 行 · DQN 系列的统一训练循环，含 Pendulum 动作离散化与 Q 值记录。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0084} | 导入与常量 | 1-4 | 导入 numpy 与 tqdm，并从 environment 导入 dis_to_con 以把离散动作映射为连续动作。 |
| 函数 {#blk0085} | `train_DQN` | 7-53 | 按 10 段 tqdm 跑 num_episodes：每步 agent.take_action 选离散动作，用 0.005/0.995 指数平滑记录 max_q_value，经 dis_to_con 转连续动作送入 Pendulum，把经验存入 replay_buffer，缓冲超过 minimal_size 时采样 batch 调 agent.update；返回 (return_list, max_q_value_list)。 |

**`03_Double DQN、Dueling DQN/visualization.py`** · 22 行 · 绘制平滑后的回合回报曲线与 max Q 值曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0086} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 用于绘图，并导入 rl_utils 取滑动平均函数。 |
| 函数 {#blk0087} | `plot_training_results` | 6-22 | 先对 return_list 做窗口 5 的 moving_average 并画出回报-回合曲线且 show；再按帧画 max_q_value_list，加 y=0（橙虚线）与 y=10（红虚线）参考线后再次 show，标题为“算法 on 环境”。 |

## 04_策略梯度算法_Reinforce：REINFORCE 的文件分工？ {#algo04 span=12}

在 CartPole-v1 上实现 REINFORCE 蒙特卡洛策略梯度，按模块拆分为策略网络、agent、同策略训练循环与绘图。 本目录共 9 个 Python 文件、38 个代码块条目。

**`04_策略梯度算法_Reinforce/agents.py`** · 39 行 · REINFORCE agent：按策略网络输出概率采样动作，并用整条轨迹的折扣回报更新策略。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0088} | 导入与常量 | 1-3 | 导入 torch、numpy，并从 networks 导入 PolicyNet。 |
| 类 {#blk0089} | `REINFORCE` | 5-39 | 封装 REINFORCE 策略梯度算法，持有策略网络、优化器与折扣因子。 |
| 方法 {#blk0090} | `REINFORCE.__init__` | 7-11 | 将 PolicyNet 构建到指定 device，用 Adam 包装网络参数并保存 gamma 与 device。 |
| 方法 {#blk0091} | `REINFORCE.take_action` | 13-18 | 把 state 转成 float32 张量并 unsqueeze 成 [1, state_dim] 送入策略网络，用 Categorical(probs).sample() 采样动作并返回 int 索引。 |
| 方法 {#blk0092} | `REINFORCE.update` | 20-39 | 逆序遍历整条 episode 的奖励，逐步累积折扣回报 G = gamma\*G + r，对每步计算 loss = -log_prob \* G 并 backward 累积梯度，循环结束后 optimizer.step() 一次性更新。 |

**`04_策略梯度算法_Reinforce/config.py`** · 12 行 · 集中定义 REINFORCE 的超参数、环境名与运行设备。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0093} | 导入与常量 | 1-11 | 导入 torch 并定义 LR=1e-3、GAMMA=0.98、NUM_EPISODES=1000、HIDDEN_DIM=128、ENV_NAME='CartPole-v1'、SEED=0，以及按 CUDA 可用性选择的 DEVICE。 |

**`04_策略梯度算法_Reinforce/environment.py`** · 7 行 · 创建 gymnasium 环境并读取状态与动作维度。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0094} | 导入与常量 | 1-1 | 导入 gymnasium 用于构造环境。 |
| 函数 {#blk0095} | `make_environment` | 3-7 | gym.make 创建环境，从 observation_space.shape[0] 取 state_dim、从 action_space.n 取离散动作数 action_dim，返回 (env, state_dim, action_dim)。 |

**`04_策略梯度算法_Reinforce/main.py`** · 32 行 · 程序入口：装配 CartPole 环境与 REINFORCE agent，执行训练和绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0096} | 导入与常量 | 1-8 | 导入 numpy、random、torch 三个随机源，以及 config、make_environment、plot_returns、train_on_policy_agent 和 REINFORCE。 |
| 函数 {#blk0097} | `set_seed` | 10-13 | 分别设置 random、numpy、torch 的随机种子以保证结果可复现。 |
| 函数 {#blk0098} | `main` | 15-29 | 设置种子后创建环境，用 config 的 HIDDEN_DIM/LR/GAMMA/DEVICE 构建 REINFORCE，调用 train_on_policy_agent 训练 config.NUM_EPISODES 个回合，再用 plot_returns 画回报曲线并关闭环境。 |
| 入口 {#blk0099} | if __name__ == '__main__' | 31-32 | 脚本直接运行时调用 main()。 |

**`04_策略梯度算法_Reinforce/networks.py`** · 13 行 · 定义输出离散动作概率分布的策略网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0100} | 导入与常量 | 1-2 | 导入 torch 构建网络层，导入 F 使用 relu 与 softmax。 |
| 类 {#blk0101} | `PolicyNet` | 4-13 | 两层全连接策略网络，输入状态、输出各动作的概率分布。 |
| 方法 {#blk0102} | `PolicyNet.__init__` | 6-9 | 定义 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→action_dim) 两层线性层。 |
| 方法 {#blk0103} | `PolicyNet.forward` | 11-13 | 对输入做 relu(fc1(x))，再经 fc2 得到 logits，并在 dim=1 上 softmax 输出可学习的多项分布概率。 |

**`04_策略梯度算法_Reinforce/practice.py`** · 101 行 · 单文件练习脚本：内联策略网络、REINFORCE 与 CartPole 训练循环。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0104} | 导入与常量 | 1-7 | 导入 gymnasium、torch 与 F、numpy、matplotlib.pyplot、tqdm 以及 rl_utils，供单文件内的网络、训练与进度展示使用。 |
| 类 {#blk0105} | `PolicyNet` | 9-18 | 与 networks.py 中同名的策略网络副本，输出离散动作概率。 |
| 方法 {#blk0106} | `PolicyNet.__init__` | 11-14 | 定义 fc1(state_dim→hidden_dim) 和 fc2(hidden_dim→action_dim) 两层全连接。 |
| 方法 {#blk0107} | `PolicyNet.forward` | 16-18 | relu 过 fc1 后接 fc2，再对 dim=1 做 softmax 得到动作概率。 |
| 类 {#blk0108} | `REINFORCE` | 20-50 | 单文件版 REINFORCE agent，与 agents.py 接口相同但更新实现略有差异。 |
| 方法 {#blk0109} | `REINFORCE.__init__` | 22-26 | 构建 policy_net 到 device，用 Adam 建优化器，保存 gamma 与 device。 |
| 方法 {#blk0110} | `REINFORCE.take_action` | 28-33 | 把 state 转成 [1, state_dim] 张量输入策略网络，按 Categorical 分布采样并返回动作索引 int。 |
| 方法 {#blk0111} | `REINFORCE.update` | 35-50 | 逆序累积折扣回报 G，用 log(policy_net(state).gather(1, action)) 计算 -log_prob\*G 并逐步 backward，最后 optimizer.step()；与 agents.py 的差别是用 gather 代替 Categorical.log_prob。 |
| 入口 {#blk0112} | 模块级训练脚本 | 52-101 | 无 __main__ 守卫的顶层脚本：定义学习率/回合数等常量，用旧版 gym API（env.seed、reset 单返回值、step 四返回值）创建 CartPole-v0，构建 REINFORCE 后跑 10×100 个回合，每回合调用 update，每 10 回合打印近 10 回合平均回报；import 该文件即会执行训练。 |

**`04_策略梯度算法_Reinforce/rl_utils.py`** · 64 行 · 通用工具库：经验回放池、滑动平均、异策略训练循环与优势估计。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0113} | 导入与常量 | 1-5 | 导入 tqdm、numpy、torch、collections 与 random，为本文件的工具函数提供依赖。 |
| 类 {#blk0114} | `ReplayBuffer` | 7-20 | 基于 collections.deque 的定长经验回放池。 |
| 方法 {#blk0115} | `ReplayBuffer.__init__` | 8-9 | 用 collections.deque(maxlen=capacity) 创建容量受限的缓冲区。 |
| 方法 {#blk0116} | `ReplayBuffer.add` | 11-12 | 把 (state, action, reward, next_state, done) 五元组追加进缓冲区，超出容量时丢弃最旧样本。 |
| 方法 {#blk0117} | `ReplayBuffer.sample` | 14-17 | random.sample 抽 batch_size 条经验并用 zip 拆成五列，states/next_states 转 numpy 数组后返回。 |
| 方法 {#blk0118} | `ReplayBuffer.size` | 19-20 | 返回缓冲区当前样本数。 |
| 函数 {#blk0119} | `moving_average` | 22-28 | 用 np.cumsum 的差分求窗口均值，并单独拼接首尾不足整窗的部分，输出与输入等长的平滑序列。 |
| 函数 {#blk0120} | `train_off_policy_agent` | 30-52 | 按 10 段 tqdm 跑异策略训练，每步写入 replay_buffer，超过 minimal_size 后采样 batch 调 agent.update，返回每回合回报列表（本目录的 REINFORCE 未使用该函数）。 |
| 函数 {#blk0121} | `compute_advantage` | 55-63 | 把 td_delta detach 为 numpy，逆序按 gamma\*lmbda 衰减累加得到优势，再反转并转回 torch.float 张量。 |

**`04_策略梯度算法_Reinforce/trainer.py`** · 40 行 · REINFORCE 的单个回合一次更新的训练循环。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0122} | 导入与常量 | 1-2 | 导入 numpy 用于统计近 10 回合平均回报，导入 tqdm 显示全局进度条。 |
| 函数 {#blk0123} | `train_on_policy_agent` | 4-40 | 用单个 tqdm 跑 num_episodes：每回合按 gymnasium 的 5 返回值 API 收集 states/actions/rewards/dones，回合结束后调用一次 agent.update，每 10 回合在 postfix 显示近 10 回合平均回报，返回 return_list。 |

**`04_策略梯度算法_Reinforce/visualization.py`** · 14 行 · 绘制 REINFORCE 的回合回报曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0124} | 导入与常量 | 1-2 | 导入 matplotlib.pyplot 绘图，并导入 rl_utils 使用滑动平均。 |
| 函数 {#blk0125} | `plot_returns` | 4-14 | 回合数不少于 9 时用窗口 9 的 moving_average 平滑，否则直接用原始回报；绘制 Episodes-Returns 曲线，标题固定为“REINFORCE on CartPole-v1”，最后 show。 |

## 05_Mountain Car：线性 Actor-Critic 如何拆文件？ {#algo05 span=12}

NumPy 手写的线性 actor-critic：用随机 RBF 特征编码二维状态，线性 Gaussian 策略加线性状态价值估计，逐步 TD 误差同时更新 actor 与 critic，在 MountainCarContinuous-v0 上训练并把曲线与统计存到 output。 本目录共 8 个 Python 文件、36 个代码块条目。

**`05_Mountain Car/code.py`** · 7 行 · 兼容入口：把 main.main 暴露为 code.py 的运行入口。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0126} | 导入与常量 | 1-3 | 模块 docstring 说明这是 actor-critic pipeline 的兼容入口，唯一依赖是从 main 模块导入 main 函数。 |
| 入口 {#blk0127} | if __name__ == '__main__' | 6-7 | 仅当脚本被直接执行时调用 main()，行为与运行 main.py 完全一致。 |

**`05_Mountain Car/config.py`** · 12 行 · 集中存放实验超参数，保证实验可复现。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0128} | 导入与常量 | 1-12 | 无 import，仅用模块级常量定义环境名 MountainCarContinuous-v0、NUM_EPISODES=50、折扣 0.95、policy/value 学习率 0.001 与 0.1、RBF 采样 10000 点与 100 个分量、四组 gamma、平滑窗口 10 和随机种子 42，供 main 导入。 |

**`05_Mountain Car/environment.py`** · 48 行 · Gymnasium 环境适配层，用 Protocol 描述接口并统一 reset/step 的返回值。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0129} | 导入与常量 | 1-6 | 导入 numpy、gymnasium、gymnasium.spaces.Box，以及 typing 的 Any、Optional、Protocol、cast，用于环境创建与静态类型标注。 |
| 类 {#blk0130} | `MountainCarEnvironment` | 9-21 | 以 Protocol 描述环境接口：声明 observation_space、action_space、spec 三个属性与 reset/step/close 的签名，仅用于类型检查，不提供实现。 |
| 方法 {#blk0131} | `MountainCarEnvironment.reset` | 14-15 | 声明 reset(\*, seed) 返回 (observation, info) 二元组的签名，函数体为省略号占位。 |
| 方法 {#blk0132} | `MountainCarEnvironment.step` | 17-18 | 声明 step(action) 返回 (observation, reward, terminated, truncated, info) 五元组的签名，函数体为省略号占位。 |
| 方法 {#blk0133} | `MountainCarEnvironment.close` | 20-21 | 声明 close() 无返回值的签名，函数体为省略号占位。 |
| 函数 {#blk0134} | `make_environment` | 24-30 | 调用 gym.make(name) 创建环境；若传入 seed，则先 reset(seed=seed) 再设置 action_space 的随机种子，最后 cast 成协议类型返回。 |
| 函数 {#blk0135} | `reset_environment` | 33-36 | 调用 env.reset() 并丢弃 info，把 observation 转成 float64 ndarray 返回，屏蔽 Gymnasium 版本差异。 |
| 函数 {#blk0136} | `step_environment` | 39-48 | 调用 env.step(action)，把 observation 转成 float64、reward 转成 float，并原样返回 terminated 与 truncated，便于调用方对截断状态继续 bootstrap。 |

**`05_Mountain Car/estimators.py`** · 88 行 · 线性状态价值估计器与线性高斯策略，只负责预测和参数更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0137} | 导入与常量 | 1-3 | 只导入 numpy，用于权重向量与特征向量的线性运算。 |
| 类 {#blk0138} | `ValueEstimator` | 6-20 | 用线性函数 V(s)=w·φ(s) 近似状态价值，提供预测与基于 TD 误差的增量更新。 |
| 方法 {#blk0139} | `ValueEstimator.__init__` | 9-11 | 按特征维度把所有初始化为 0 的 float64 权重，并保存学习率（默认 0.1）。 |
| 方法 {#blk0140} | `ValueEstimator.predict` | 13-14 | 返回权重与输入特征的 np.dot 结果并转成 float，即当前状态价值估计。 |
| 方法 {#blk0141} | `ValueEstimator.update` | 16-20 | 计算 TD 误差 target-prediction，按 w += learning_rate \* error \* features 更新权重，并把该误差返回给调用方。 |
| 类 {#blk0142} | `GaussianPolicy` | 23-88 | 连续动作的线性高斯策略：均值与 log 标准差都由特征线性组合得到，采样后裁剪到动作区间，并按策略梯度加熵正则更新。 |
| 方法 {#blk0143} | `GaussianPolicy.__init__` | 30-42 | 初始化全零均值权重；standard_deviation 为 None 时 log_std 初值取 0（即 softplus(0)），否则用 log(expm1(σ)) 反解初值；保存学习率、熵系数、动作上下界，并建立 default_rng(seed)。 |
| 方法 {#blk0144} | `GaussianPolicy.mean` | 44-45 | 返回均值权重与特征的点积（float），作为高斯分布的均值。 |
| 方法 {#blk0145} | `GaussianPolicy.log_standard_deviation` | 47-48 | 返回 log_std 权重与特征的点积（float），即标准差的对数表示。 |
| 方法 {#blk0146} | `GaussianPolicy.standard_deviation` | 50-53 | 把 raw 值 clip 到 [-10, 5] 后取 softplus（log1p(exp(raw))）并加 1e-5，得到恒正的标准差，同时避免指数溢出。 |
| 方法 {#blk0147} | `GaussianPolicy.sample` | 55-59 | 用 rng.normal(mean, std) 采样一个连续动作，并 np.clip 到 [action_low, action_high] 后返回。 |
| 方法 {#blk0148} | `GaussianPolicy.update` | 61-88 | 按高斯 log 概率对均值与 log_std 的梯度乘上 advantage，log_std 方向再加 entropy_coefficient \* features；两路更新分别裁剪到 ±0.05 后累加，随后做 nan_to_num 与权重 ±10 / log_std ±5 的裁剪以防止线性策略发散。 |

**`05_Mountain Car/features.py`** · 35 行 · 状态标准化与随机 RBF 特征工程。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0149} | 导入与常量 | 1-6 | 导入 numpy 与 sklearn 的 RBFSampler、FeatureUnion、StandardScaler，用于构建归一化加多尺度 RBF 特征。 |
| 类 {#blk0150} | `RBFStateFeaturizer` | 9-35 | 把二维 observation 转换成长度固定为 len(gammas)\*components 的随机傅里叶 RBF 特征向量。 |
| 方法 {#blk0151} | `RBFStateFeaturizer.__init__` | 12-25 | 按 observation_space 的上下界均匀采样 samples 个状态点拟合 StandardScaler；再为每个 gamma 建一个 n_components=components 的 RBFSampler，用 FeatureUnion 合并后在标准化样本上 fit。 |
| 方法 {#blk0152} | `RBFStateFeaturizer.feature_count` | 28-30 | property：对占位状态 [0.0, 0.0] 做一次 transform 并取长度，得到特征维度，供构造 actor/critic 使用。 |
| 方法 {#blk0153} | `RBFStateFeaturizer.transform` | 32-35 | 把单个 state reshape 成 (1, 2) 并转 float64，先 StandardScaler 标准化再经 FeatureUnion 变换，返回第一行一维特征向量。 |

**`05_Mountain Car/main.py`** · 58 行 · 组装特征器、策略、价值估计器并运行完整训练流程。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0154} | 导入与常量 | 3-15 | 导入 Path 与 numpy，从 config 取出全部超参数，并导入 environment.make_environment、estimators、features、trainer.train 与 visualization.plot_episode_stats。 |
| 函数 {#blk0155} | `main` | 18-54 | 解析脚本同级的 output 目录并创建；用 config 构造带种子的环境、RBFStateFeaturizer、GaussianPolicy（动作界取自 action_space 的 low/high）与 ValueEstimator，调用 train 得到统计后写出 episode_stats.csv 和 run_config.txt，再以 show=False 绘图，finally 中关闭环境。 |
| 入口 {#blk0156} | if __name__ == '__main__' | 57-58 | 直接运行 main.py 时调用 main() 启动训练。 |

**`05_Mountain Car/trainer.py`** · 55 行 · 逐步 TD actor-critic 训练循环与 episode 统计记录。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0157} | 导入与常量 | 3-9 | 导入 dataclass、numpy，以及 environment 的 reset/step 封装与协议类型、estimators 的 GaussianPolicy/ValueEstimator、features 的 RBFStateFeaturizer。 |
| 类 {#blk0158} | `EpisodeStats` | 13-17 | dataclass，保存每个 episode 的总奖励 episode_rewards 与步数 episode_lengths 两个数组，供保存 CSV 与绘图使用。 |
| 函数 {#blk0159} | `train` | 20-55 | 外层按 episodes 循环、内层按 spec.max_episode_steps 逐步执行：特征化状态、策略采样、环境 step；terminated 时 target=reward，truncated 时 target=reward+γV(s')，用 advantage 同时更新 critic 与 policy，累计奖励与步数并按 print_interval 打印，最后返回 EpisodeStats。 |

**`05_Mountain Car/visualization.py`** · 41 行 · 绘制奖励曲线与 episode 长度曲线并保存为 PNG。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0160} | 导入与常量 | 3-10 | 导入 numpy、matplotlib 的 pyplot/Axes/Figure、Path、Optional，以及 trainer.EpisodeStats 作为绘图输入类型。 |
| 函数 {#blk0161} | `plot_episode_stats` | 13-41 | 对奖励做长度为 smoothing_window 的滑动平均（序列过短则用原值），画 2x1 子图展示原始与平滑奖励、每 episode 步数并设标题；若给 output_dir 则保存 training_curves.png，show 为假时关闭 figure，返回 (figure, axes)。 |

## 06_Mountain Car PyTorch：MLP 版本改了哪些文件？ {#algo06 span=12}

PyTorch 版 Mountain Car actor-critic：MLP Gaussian 策略与 MLP 状态价值网络，每个 episode 收集完整轨迹后用 n 步 TD 回报与 Adam 做批量更新，并保存模型、统计与曲线。 本目录共 6 个 Python 文件、23 个代码块条目。

**`06_Mountain Car PyTorch/config.py`** · 22 行 · 集中存放 PyTorch 版训练超参数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0162} | 导入与常量 | 1-22 | 无 import，仅定义常量并用注释解释取值：环境名、200 个 episode、折扣 0.95、actor/critic 学习率 3e-4 与 1e-3、隐藏层 128、熵系数 1e-2、n 步 5、奖励缩放 0.01、种子 42、平滑窗口 10 与打印间隔 10。 |

**`06_Mountain Car PyTorch/environment.py`** · 39 行 · Gymnasium 环境创建与 reset/step 返回值适配。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0163} | 导入与常量 | 1-6 | 导入 typing 的 Optional/Tuple、gymnasium 与 numpy。 |
| 函数 {#blk0164} | `make_environment` | 9-15 | 调用 gym.make(name) 创建环境；若给定 seed，则 reset(seed=seed) 并设置 action_space 的种子后返回环境对象。 |
| 函数 {#blk0165} | `reset_environment` | 18-21 | 调用 environment.reset() 丢弃 info，把 observation 转成 float32 ndarray 返回，与网络输入精度一致。 |
| 函数 {#blk0166} | `step_environment` | 24-39 | 调用 environment.step(action)，返回 (float32 observation, float reward, bool terminated, bool truncated)，保留截断标记以便继续 bootstrap。 |

**`06_Mountain Car PyTorch/main.py`** · 74 行 · 设置随机种子、构建 Actor/Critic 与优化器并运行训练、保存结果。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0167} | 导入与常量 | 3-26 | 导入 Path、random、numpy、torch，从 config 取出全部超参数，并导入 environment、networks 的 Actor/Critic、trainer.train 与 visualization 的绘图函数。 |
| 函数 {#blk0168} | `main` | 29-70 | 先用 RANDOM_SEED 固定 random、numpy、torch 的随机性；创建环境后由其 observation/action 维度构造 Actor 与 Critic 及两个 Adam 优化器，调用 train 训练；随后创建 output 目录，写出 episode_stats.csv、保存 actor_critic.pt 并绘图，finally 中关闭环境。 |
| 入口 {#blk0169} | if __name__ == '__main__' | 73-74 | 直接运行 main.py 时调用 main() 启动训练。 |

**`06_Mountain Car PyTorch/networks.py`** · 64 行 · 定义输出动作被裁剪到 [-1,1] 的高斯策略 Actor 与状态价值 Critic。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0170} | 导入与常量 | 3-7 | 导入 typing.Tuple、torch、torch.nn 与 torch.distributions.Normal。 |
| 类 {#blk0171} | `Actor` | 10-47 | 两层 ReLU MLP 输出高斯均值，log_std 为独立可学习参数；采样后直接 clip 到 [-1,1]（而非 tanh），docstring 说明这样能形成 bang-bang 摆动以爬坡。 |
| 方法 {#blk0172} | `Actor.__init__` | 20-33 | 构建 state→hidden→hidden 的两层 ReLU Sequential 与输出均值的 Linear(hidden, action_size)，并把 log_std 初始化为 0（std=1）的可学习 Parameter。 |
| 方法 {#blk0173} | `Actor.distribution` | 35-39 | 前向得到均值，把 log_std clamp 到 [-5, 1] 后取 exp 得标准差，返回 Normal(mean, std) 分布对象。 |
| 方法 {#blk0174} | `Actor.sample` | 41-47 | 对分布 rsample 得动作并 clamp 到 [-1,1]；log_prob 用未裁剪的 raw_action 按最后一维求和，同时返回 entropy 之和，组成 (action, log_prob, entropy)。 |
| 类 {#blk0175} | `Critic` | 50-64 | 状态价值网络 V(s)：两层 ReLU 隐藏层后接单输出线性层。 |
| 方法 {#blk0176} | `Critic.__init__` | 53-61 | 构建 Linear(state,hidden)→ReLU→Linear(hidden,hidden)→ReLU→Linear(hidden,1) 的 Sequential。 |
| 方法 {#blk0177} | `Critic.forward` | 63-64 | 对输入状态做前向并对最后一维 squeeze(-1)，返回形状 (batch,) 的价值估计。 |

**`06_Mountain Car PyTorch/trainer.py`** · 148 行 · 按 episode 收集轨迹并用 n 步 TD 回报批量更新 actor 与 critic。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0178} | 导入与常量 | 16-24 | 导入 dataclass、Any、numpy、torch、torch.nn，以及 environment 的 reset/step 封装与 networks 的 Actor/Critic；文件头部 docstring 解释为何采用 episode 批量 n 步更新。 |
| 类 {#blk0179} | `EpisodeStats` | 28-30 | dataclass，记录每个 episode 的 rewards 与 lengths 数组，供保存 CSV 与绘图使用。 |
| 函数 {#blk0180} | `normalize_state` | 33-38 | 用 observation_space 的 low/high 把状态按 2\*(s-low)/(high-low)-1 映射到约 [-1,1]，并转成 float32 torch tensor。 |
| 函数 {#blk0181} | `n_step_targets` | 41-67 | 对 episode 内每个时刻 t，向前累加最多 n_step 步的 reward\*reward_scale 并按 γ 折扣；若中途 terminated 则提前停止，否则加上最后一步 γ 折扣后的 V(s_last) 作为 bootstrap，返回逐时刻目标 tensor。 |
| 函数 {#blk0182} | `train` | 70-148 | 每个 episode 在 no_grad 下用 actor 采样动作并收集 state/action/reward/terminated/next_state；结束后算 next_values 与 n 步 targets，以 MSE 更新 critic、以 -(log_prob\*advantage) 减熵项更新 actor（两者都做 1.0 梯度裁剪），按 print_interval 打印并返回 EpisodeStats。 |

**`06_Mountain Car PyTorch/visualization.py`** · 39 行 · 绘制训练奖励与步数曲线并保存 PNG。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0183} | 导入与常量 | 3-9 | 导入 Path、Optional、matplotlib.pyplot、numpy 与 trainer.EpisodeStats。 |
| 函数 {#blk0184} | `plot_episode_stats` | 12-39 | 序列足够长时对奖励做滑动平均并计算偏移量，否则用原值；画 2x1 子图展示原始/平滑奖励与步数，设标题；若给 output_dir 则保存 training_curves.png，最后关闭 figure。 |

## 07_TRPO：离散与连续两套 TRPO 如何组织？ {#algo07 span=12}

TRPO（信赖域策略优化）的两个独立实现：CartPole 用 Categorical 离散策略，Pendulum 用 Normal 连续策略，均包含 GAE、Hessian-向量积、共轭梯度、KL 约束与线搜索。 本目录共 17 个 Python 文件、81 个代码块条目。

**`07_TRPO/CartPole/agents.py`** · 185 行 · 离散动作 TRPO agent，实现优势估计、Hessian-向量积、共轭梯度、线搜索与策略/价值更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0185} | 导入与常量 | 1-8 | 导入 copy、numpy、torch、torch.nn.functional，以及本地 rl_utils 和 networks 中的 PolicyNet、ValueNet；无模块级常量。 |
| 类 {#blk0186} | `TRPO` | 11-185 | 封装 CartPole 上的 TRPO：actor 输出离散动作概率，critic 回归 TD 目标，策略更新通过自然梯度方向加 KL 信赖域约束完成。 |
| 方法 {#blk0187} | `TRPO.__init__` | 14-25 | 按 state_dim、hidden_dim、action_dim 构建 PolicyNet 与 ValueNet，只为 critic 建 Adam 优化器（actor 由 TRPO 直接改参数），保存 gamma、lmbda、kl_constraint、alpha 与 device。 |
| 方法 {#blk0188} | `TRPO.take_action` | 27-32 | 把单条 state 转成 [1, state_dim] 的 float32 张量，经 actor 得动作概率并构造 Categorical 分布，采样后返回 Python int 动作。 |
| 方法 {#blk0189} | `TRPO.hessian_matrix_vector_product` | 34-48 | 以旧分布与当前策略的平均 KL 为标量，create_graph=True 对 actor 参数求一阶导，拼接后与 vector 点积再求导，返回 detach 后的 H·v。 |
| 方法 {#blk0190} | `TRPO.conjugate_gradient` | 50-71 | 用最多 10 次共轭梯度迭代求解 H^{-1}g，每步调用 Hessian-向量积，残差平方和小于 1e-10 时提前退出，返回近似自然梯度方向 x。 |
| 方法 {#blk0191} | `TRPO._log_probs` | 74-76 | 静态方法：用给定 actor 构造 Categorical 分布，对 actions.squeeze(-1) 取 log_prob 并 unsqueeze(-1) 成列向量。 |
| 方法 {#blk0192} | `TRPO.compute_surrogate_obj` | 78-82 | 用新 actor 的对数概率减去 old_log_probs 取 exp 得 ratio，返回 mean(ratio \* advantage) 作为 surrogate 目标。 |
| 方法 {#blk0193} | `TRPO.line_search` | 84-110 | 把 actor 参数展平为向量，按 alpha^i（i=0..14）沿 max_vector 搜索候选参数；每个候选 deepcopy actor 后写入参数，若 surrogate 提升且平均 KL 小于 kl_constraint 则返回该参数，否则最终回退旧参数。 |
| 方法 {#blk0194} | `TRPO.policy_learn` | 112-139 | 对 surrogate 目标求 actor 梯度并拼接为一维，经共轭梯度得下降方向，用 H·d 计算缩放 sqrt(2\*kl/(d^T H d))，线搜索后把新参数 vector_to_parameters 写回 actor。 |
| 方法 {#blk0195} | `TRPO.update` | 141-185 | 把 transition_dict 的五个字段转成张量，计算 td_target 与 td_delta，调用 rl_utils.compute_advantage 得 GAE 优势；记录 old_log_probs 与旧 Categorical 分布，先用 MSE 更新 critic，再调用 policy_learn 更新策略。 |

**`07_TRPO/CartPole/config.py`** · 16 行 · 集中定义 CartPole 离散 TRPO 的训练超参数与运行设备。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0196} | 导入与常量 | 1-16 | 导入 torch，并定义 NUM_EPISODES=500、HIDDEN_DIM=128、GAMMA=0.98、LMBDA=0.95、CRITIC_LR=1e-2、KL_CONSTRAINT=0.0005、ALPHA=0.5、SEED=0、ENV_NAME='CartPole-v1'，按 CUDA 可用性选择 DEVICE。 |

**`07_TRPO/CartPole/environment.py`** · 15 行 · 创建离散动作 Gym 环境并返回状态维度与动作数量。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0197} | 导入与常量 | 1-4 | 导入 gymnasium（前两行为空行），无模块级常量。 |
| 函数 {#blk0198} | `make_environment` | 6-15 | 调用 gym.make(env_name)，若 action_space 没有 n 属性则抛 TypeError；随后从 observation_space.shape[0] 与 action_space.n 取维度，返回 (env, state_dim, action_dim)。 |

**`07_TRPO/CartPole/main.py`** · 41 行 · CartPole TRPO 实验的入口脚本，负责装配环境、agent 并启动训练与绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0199} | 导入与常量 | 1-10 | 导入 random、numpy、torch，以及本地 config、TRPO、make_environment、train_on_policy_agent、plot_training_results；无模块级常量。 |
| 函数 {#blk0200} | `set_seed` | 13-16 | 依次调用 random.seed、np.random.seed、torch.manual_seed，用同一 seed 固定三处随机源。 |
| 函数 {#blk0201} | `main` | 19-37 | 先设随机种子，用 make_environment 取 env 与状态/动作维度，按 config 超参构造 TRPO，调用 train_on_policy_agent 训练 NUM_EPISODES 局，绘图后 env.close()。 |
| 入口 {#blk0202} | if __name__ == '__main__' | 40-41 | 脚本直接运行时调用 main() 启动训练。 |

**`07_TRPO/CartPole/networks.py`** · 28 行 · 定义 CartPole 使用的离散策略网络与状态价值网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0203} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional，无模块级常量。 |
| 类 {#blk0204} | `PolicyNet` | 5-15 | 两层全连接网络，输入状态，输出每个离散动作的 softmax 概率。 |
| 方法 {#blk0205} | `PolicyNet.__init__` | 8-11 | 建立 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→action_dim) 两个线性层。 |
| 方法 {#blk0206} | `PolicyNet.forward` | 13-15 | 先对 fc1 输出做 ReLU，再经 fc2 并在 dim=1 上 softmax，返回动作概率分布。 |
| 类 {#blk0207} | `ValueNet` | 18-28 | 状态价值网络，输入状态，输出标量 V(s)。 |
| 方法 {#blk0208} | `ValueNet.__init__` | 21-24 | 建立 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→1) 两个线性层。 |
| 方法 {#blk0209} | `ValueNet.forward` | 26-28 | 对 fc1 输出做 ReLU 后过 fc2，返回形状 [batch, 1] 的状态价值。 |

**`07_TRPO/CartPole/practice.py`** · 169 行 · 单文件练习版 TRPO 实现，把网络、agent 与训练依赖集中在一个脚本中。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0210} | 导入与常量 | 1-7 | 导入 torch、numpy、gymnasium、matplotlib.pyplot、F、rl_utils 与 copy；无模块级常量。 |
| 类 {#blk0211} | `PolicyNet` | 9-18 | 与 networks.py 相同的离散策略网络，输出各动作概率。 |
| 方法 {#blk0212} | `PolicyNet.__init__` | 11-14 | 建立 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→action_dim)。 |
| 方法 {#blk0213} | `PolicyNet.forward` | 16-18 | fc1 后接 ReLU，再过 fc2 并在 dim=1 上 softmax 返回概率。 |
| 类 {#blk0214} | `ValueNet` | 21-30 | 状态价值网络，输出标量状态价值。 |
| 方法 {#blk0215} | `ValueNet.__init__` | 23-26 | 建立 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→1)。 |
| 方法 {#blk0216} | `ValueNet.forward` | 28-30 | fc1 后接 ReLU，再经 fc2 输出 [batch, 1] 价值。 |
| 类 {#blk0217} | `TRPO` | 33-169 | 练习版 TRPO agent，直接接收 gym 的 state_space/action_space 对象，流程与模块版一致（surrogate、HVP、CG、线搜索）。 |
| 方法 {#blk0218} | `TRPO.__init__` | 35-48 | 从 state_space.shape[0] 与 action_space.n 取维度，构建 actor/critic 及 critic 的 Adam 优化器，保存 gamma、lmbda、kl_constraint、alpha、device。 |
| 方法 {#blk0219} | `TRPO.take_action` | 50-55 | 把 state 包成 [1, state_dim] 的 float 张量，经 actor 得概率并构造 Categorical 分布，采样后返回 item。 |
| 方法 {#blk0220} | `TRPO.hessian_matrix_vector_product` | 57-72 | 计算新旧 Categorical 分布的平均 KL，create_graph 求一阶导后与 vector 点积，再次求导并拼接，返回 H·v（未做 damping）。 |
| 方法 {#blk0221} | `TRPO.conjugate_gradient` | 74-91 | 最多 10 次共轭梯度迭代求解 H^{-1}g，残差平方和小于 1e-10 时 break，返回解向量。 |
| 方法 {#blk0222} | `TRPO.compute_surrogate_obj` | 93-97 | 用 actor(states).gather(1, actions) 取所选动作对数概率，计算 ratio = exp(logp - old_logp)，返回 mean(ratio \* advantage)。 |
| 方法 {#blk0223} | `TRPO.line_search` | 99-120 | 把参数展平，按 alpha^i 尝试 15 个候选参数，deepcopy actor 并写入候选参数，比较 surrogate 与平均 KL，返回首个满足提升且 KL 小于约束的参数，否则返回原参数。 |
| 方法 {#blk0224} | `TRPO.policy_learn` | 122-140 | 对 surrogate 求 actor 梯度并拼接，用共轭梯度得方向，按 sqrt(2\*kl/(d^T H d)) 缩放后线搜索，最后 vector_to_parameters 写回 actor。 |
| 方法 {#blk0225} | `TRPO.update` | 142-169 | 张量化 transition 的五个字段，计算 td_target、td_delta 并用 rl_utils.compute_advantage（先转 CPU）得 GAE 优势；取 old_log_probs 与旧分布后，用 MSE 更新 critic，再调用 policy_learn。 |

**`07_TRPO/CartPole/rl_utils.py`** · 31 行 · 提供移动平均与 GAE 优势估计两个通用工具函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0226} | 导入与常量 | 1-2 | 导入 numpy 与 torch，无模块级常量。 |
| 函数 {#blk0227} | `moving_average` | 5-19 | 把输入转成 float32 数组，长度小于 window_size 时原样返回；否则用 cumsum 计算中间段窗口均值，两端用半径 1,3,5... 的前缀/后缀累积均值补齐，最后 concatenate。 |
| 函数 {#blk0228} | `compute_advantage` | 22-31 | 将 td_delta 转为一维 numpy，从后向前递推 advantage = gamma\*lmbda\*advantage + delta，反转序列后返回 float32 的 torch 张量。 |

**`07_TRPO/CartPole/trainer.py`** · 41 行 · 实现按整局采样的 on-policy 训练循环。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0229} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0230} | `train_on_policy_agent` | 5-41 | 用 tqdm 循环 num_episodes 局：每局 reset 后反复调用 agent.take_action 与 env.step，直到 terminated 或 truncated，沿途收集 states/actions/next_states/rewards/dones；整局结束后调用一次 agent.update 并记录回报，每 10 局打印最近 10 局均值，最终返回 return_list。 |

**`07_TRPO/CartPole/visualization.py`** · 27 行 · 绘制 CartPole TRPO 训练的原始回报与移动平均回报曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0231} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与本地 rl_utils.moving_average，无模块级常量。 |
| 函数 {#blk0232} | `plot_training_results` | 6-27 | 创建 1x2 子图：左图绘制每局原始回报；右图在样本数不少于 9 时用 moving_average(window=9) 平滑、否则用原始值，设置坐标轴标题后 tight_layout 并 plt.show()。 |

**`07_TRPO/Pendulum/agents.py`** · 205 行 · 连续动作 TRPO agent，使用高斯策略并处理动作边界与奖励缩放。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0233} | 导入与常量 | 1-8 | 导入 copy、numpy、torch、torch.nn.functional，以及本地 rl_utils 与 networks 中的 PolicyNetContinuous、ValueNet；无模块级常量。 |
| 类 {#blk0234} | `TRPOContinuous` | 11-205 | 面向 Pendulum 的连续动作 TRPO：策略为 Normal 分布，KL 按动作维求和，Hessian-向量积带阻尼，配合共轭梯度、信赖域缩放与线搜索更新策略。 |
| 方法 {#blk0235} | `TRPOContinuous.__init__` | 14-36 | 构建 PolicyNetContinuous（传入动作上下界）与 ValueNet，把 action_low/action_high 存为张量，为 critic 建 Adam 优化器，保存 gamma、lmbda、kl_constraint、alpha、reward_shift、reward_scale 与 device。 |
| 方法 {#blk0236} | `TRPOContinuous._distribution` | 38-40 | 调用 actor 得到均值与标准差，返回 torch.distributions.Normal 分布对象。 |
| 方法 {#blk0237} | `TRPOContinuous.take_action` | 42-48 | 把 state 转为 [1, state_dim] 张量，从 Normal 分布采样，再用 clamp 裁剪到 [action_low, action_high]，返回去掉 batch 维的 numpy 动作。 |
| 方法 {#blk0238} | `TRPOContinuous.hessian_matrix_vector_product` | 50-65 | 计算新旧 Normal 分布 KL（动作维求和后取均值），create_graph 求一阶导并与 vector 点积后再求导，返回 H·v 加上阻尼项 damping\*vector（默认 0.1）。 |
| 方法 {#blk0239} | `TRPOContinuous.conjugate_gradient` | 67-87 | 最多 10 次共轭梯度迭代求解 H^{-1}g，每步调用带阻尼的 Hessian-向量积，残差平方和小于 1e-10 时提前退出并返回解向量。 |
| 方法 {#blk0240} | `TRPOContinuous._log_probs` | 90-94 | 静态方法：用 actor 输出的 mean/std 构造 Normal，对 actions 求 log_prob 并在动作维求和，返回列向量对数概率。 |
| 方法 {#blk0241} | `TRPOContinuous.compute_surrogate_obj` | 96-100 | 由新旧对数概率之差取 exp 得 ratio，返回 mean(ratio \* advantage) 作为 surrogate 目标。 |
| 方法 {#blk0242} | `TRPOContinuous.line_search` | 102-126 | 沿 max_vector 按 alpha^i（i=0..14）生成候选参数，deepcopy actor 后写入并计算 KL（动作维求和取均值）与 surrogate，返回首个同时满足目标提升和 KL 小于约束的参数，否则回退旧参数。 |
| 方法 {#blk0243} | `TRPOContinuous.policy_learn` | 128-155 | 对 surrogate 求 actor 梯度并拼接为一维，经共轭梯度得下降方向，用 clamp 后的曲率 d^T H d 计算缩放 sqrt(2\*kl/curvature)，线搜索后用 vector_to_parameters 写回 actor。 |
| 方法 {#blk0244} | `TRPOContinuous.update` | 157-205 | 把 transition_dict 转成张量（动作为 float），先用 (reward+reward_shift)/reward_scale 缩放奖励，再算 td_target、td_delta 与 GAE 优势；记录 old_log_probs 和 detach 后的旧 Normal 分布，以 MSE 更新 critic，最后调用 policy_learn 更新策略。 |

**`07_TRPO/Pendulum/config.py`** · 15 行 · 集中定义 Pendulum 连续 TRPO 的训练超参数与运行设备。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0245} | 导入与常量 | 1-15 | 导入 torch，定义 NUM_EPISODES=2000、HIDDEN_DIM=128、GAMMA=0.9、LMBDA=0.9、CRITIC_LR=1e-2、KL_CONSTRAINT=0.00005、ALPHA=0.5、REWARD_SHIFT=8.0、REWARD_SCALE=8.0、SEED=0、ENV_NAME='Pendulum-v1'，并按 CUDA 可用性选择 DEVICE。 |

**`07_TRPO/Pendulum/environment.py`** · 17 行 · 创建连续动作 Gym 环境并返回状态维度、动作维度及动作上下界。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0246} | 导入与常量 | 1-2 | 导入 gymnasium as gym，无模块级常量。 |
| 函数 {#blk0247} | `make_environment` | 4-17 | 调用 gym.make(env_name)，检查 action_space 是否有非 None 的 shape，不满足则抛 TypeError；随后读取状态维、动作维，并返回 (env, state_dim, action_dim, action_space.low.copy(), action_space.high.copy())。 |

**`07_TRPO/Pendulum/main.py`** · 47 行 · Pendulum TRPO 实验的入口脚本，装配连续动作环境、agent 并启动训练与绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0248} | 导入与常量 | 1-10 | 导入 random、numpy、torch，以及本地 config、TRPOContinuous、make_environment、train_on_policy_agent、plot_training_results；无模块级常量。 |
| 函数 {#blk0249} | `set_seed` | 13-16 | 依次设置 random、numpy 与 torch 的随机种子以固定实验随机性。 |
| 函数 {#blk0250} | `main` | 19-43 | 设种子后解包 make_environment 返回的 5 个值，按 config 超参（含动作边界、reward_shift/scale）构造 TRPOContinuous，训练 NUM_EPISODES 局，绘图并 env.close()。 |
| 入口 {#blk0251} | if __name__ == '__main__' | 46-47 | 脚本直接运行时调用 main() 启动训练。 |

**`07_TRPO/Pendulum/networks.py`** · 34 行 · 定义连续动作策略网络（输出高斯均值与标准差）与状态价值网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0252} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional，无模块级常量。 |
| 类 {#blk0253} | `PolicyNetContinuous` | 5-22 | 连续动作策略网络，输出动作高斯分布的均值与标准差，均值经 tanh 映射到动作区间。 |
| 方法 {#blk0254} | `PolicyNetContinuous.__init__` | 8-16 | 建立 fc1、fc_mu、fc_std 三个线性层，并把动作上下界转成张量后以 register_buffer 保存 action_scale=(high-low)/2 与 action_bias=(high+low)/2。 |
| 方法 {#blk0255} | `PolicyNetContinuous.forward` | 18-22 | fc1 后接 ReLU，分两路输出：mu = bias + scale\*tanh(fc_mu)，std = softplus(fc_std)+1e-5，返回 (mu, std)。 |
| 类 {#blk0256} | `ValueNet` | 25-34 | 状态价值网络，输入状态输出标量 V(s)。 |
| 方法 {#blk0257} | `ValueNet.__init__` | 28-31 | 建立 fc1(state_dim→hidden_dim) 与 fc2(hidden_dim→1)。 |
| 方法 {#blk0258} | `ValueNet.forward` | 33-34 | 将输入经 fc1、ReLU、fc2 后返回 [batch, 1] 状态价值。 |

**`07_TRPO/Pendulum/rl_utils.py`** · 25 行 · 提供移动平均与 GAE 优势估计工具函数（CartPole 版本的简化实现）。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0259} | 导入与常量 | 1-2 | 导入 numpy 与 torch，无模块级常量。 |
| 函数 {#blk0260} | `moving_average` | 5-14 | 把输入转成 float32 数组，长度不足窗口时原样返回；否则用 cumsum 计算中间段窗口均值，两端按半径 1,3,5... 的前后缀累积均值补齐并拼接返回。 |
| 函数 {#blk0261} | `compute_advantage` | 17-25 | 将 td_delta 转一维 numpy 后从后向前递推 advantage = gamma\*lmbda\*advantage + delta，反转并转成 float32 torch 张量返回。 |

**`07_TRPO/Pendulum/trainer.py`** · 35 行 · 实现 Pendulum 的按整局采样 on-policy 训练循环。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0262} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0263} | `train_on_policy_agent` | 5-35 | 用 tqdm 循环 num_episodes 局，每局 reset 后循环执行 agent.take_action 与 env.step 直到 done，积累回报并收集五类 transition 字段；每局结束调用一次 agent.update，每 10 局打印最近 10 局平均回报，最终返回 return_list。 |

**`07_TRPO/Pendulum/visualization.py`** · 21 行 · 绘制 Pendulum TRPO 训练的原始回报与移动平均回报曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0264} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与本地 rl_utils.moving_average，无模块级常量。 |
| 函数 {#blk0265} | `plot_training_results` | 6-21 | 创建 1x2 子图，左图画每局原始回报，右图画 moving_average(return_list, 9) 平滑曲线，设置坐标轴标题后 tight_layout 并 plt.show()。 |

## 08_PPO：离散与连续两套 PPO 如何组织？ {#algo08 span=12}

PPO（近端策略优化）的两个 Gymnasium 实例：CartPole 用离散动作 Categorical 策略，Pendulum 用连续动作 Normal 策略，均以 GAE + PPO-Clip 更新 actor-critic，并按 config/environment/networks/agents/trainer/visualization 拆分为可运行工程。 本目录共 17 个 Python 文件、75 个代码块条目。

**`08_PPO/CartPole/agents.py`** · 113 行 · 实现离散动作 PPO 智能体，负责动作采样与基于 GAE、PPO-Clip 的 actor-critic 更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0266} | 导入与常量 | 1-6 | 导入 numpy、torch、torch.nn.functional，以及本地 rl_utils 和 networks 中的 PolicyNet、ValueNet；本文件没有模块级常量。 |
| 类 {#blk0267} | `PPO` | 9-113 | 离散动作 PPO 智能体，持有 actor（策略网络）、critic（价值网络）及各自的 Adam 优化器，并用 gamma、lmbda、epochs、eps 控制折扣、GAE 与裁剪更新。 |
| 方法 {#blk0268} | `PPO.__init__` | 17-31 | 按 state_dim/hidden_dim/action_dim 构造 PolicyNet 与 ValueNet 并搬到 device，分别为两者建立 Adam 优化器，同时保存 gamma、lmbda、epochs、eps 与 device。 |
| 方法 {#blk0269} | `PPO.take_action` | 33-40 | 把单个 state 转为 float32 张量并 unsqueeze 成 [1, state_dim]，送入 actor 得到动作概率，用 torch.distributions.Categorical 采样并返回 .item() 的整数动作。 |
| 方法 {#blk0270} | `PPO._log_probs` | 43-47 | 静态方法：用 actor 输出构造 Categorical，把形状 [B, 1] 的 actions squeeze 成 [B] 调用 log_prob，再 unsqueeze(-1) 还原为 [B, 1] 以对齐 advantage 与 ratio。 |
| 方法 {#blk0271} | `PPO.update` | 49-113 | 将 transition_dict 的 states/actions/rewards/next_states/dones 转为张量，按 r + gamma\*V(s')\*(1-done) 算 td_target 与 td_delta，用 rl_utils.compute_advantage 得 GAE 并 unsqueeze 成 [T,1]；记录 detach 的 old_log_probs 后循环 epochs 次，计算 ratio 与 clamp 后的保守目标得到 actor_loss，critic 用 MSE 拟合 td_target，两者分别反向传播并 step。 |

**`08_PPO/CartPole/config.py`** · 18 行 · 集中存放 CartPole 实验的全部超参数、环境名与计算设备选择。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0272} | 导入与常量 | 1-18 | 导入 torch，定义 actor/critic 学习率（1e-3、1e-2）、NUM_EPISODES=500、HIDDEN_DIM=128、GAMMA=0.98、LMBDA=0.95、EPOCHS=10、EPS=0.2、SEED=0、ENV_NAME='CartPole-v1'，并按 CUDA 可用性设置 DEVICE。 |

**`08_PPO/CartPole/environment.py`** · 10 行 · 创建 CartPole 环境并校验其动作空间为离散型，返回维度信息。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0273} | 导入与常量 | 1-1 | 仅导入 gymnasium 的 gym 模块，无模块级常量。 |
| 函数 {#blk0274} | `make_environment` | 4-10 | 用 gym.make(env_name) 创建环境，若 action_space 没有 n 属性则抛 TypeError 提示需要离散动作空间；随后从 observation_space 与 action_space 取出 state_dim、action_dim 并连同 env 一起返回。 |

**`08_PPO/CartPole/main.py`** · 43 行 · CartPole 训练入口，负责设种子、装配 PPO 智能体、启动训练并绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0275} | 导入与常量 | 1-10 | 导入 random、numpy、torch，以及本地 config、PPO、make_environment、train_on_policy_agent、plot_training_results；无模块级常量。 |
| 函数 {#blk0276} | `set_seed` | 13-16 | 依次调用 random.seed、np.random.seed、torch.manual_seed 设置同一随机种子，保证实验可复现。 |
| 函数 {#blk0277} | `main` | 19-39 | 设种子后调用 make_environment 得到 env 与维度，env.reset(seed=...) 固定初始状态；用 config 中的超参数构造 PPO，调用 train_on_policy_agent 训练 NUM_EPISODES 个 episode，最后绘制回报曲线并 env.close()。 |
| 入口 {#blk0278} | if __name__ == '__main__' | 42-43 | 脚本直接运行时调用 main() 启动 CartPole 的 PPO 训练。 |

**`08_PPO/CartPole/networks.py`** · 37 行 · 定义离散 PPO 使用的策略网络与状态价值网络结构。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0279} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional，无模块级常量。 |
| 类 {#blk0280} | `PolicyNet` | 5-21 | 两层全连接策略网络，输入状态、输出每个离散动作的概率分布。 |
| 方法 {#blk0281} | `PolicyNet.__init__` | 12-15 | 建立 state_dim→hidden_dim 的 fc1 与 hidden_dim→action_dim 的 fc2 两个线性层。 |
| 方法 {#blk0282} | `PolicyNet.forward` | 17-21 | 对 fc1 输出做 ReLU 再过 fc2，最后在 dim=1 上 softmax，使每行成为合法动作概率分布，形状 [B, action_dim]。 |
| 类 {#blk0283} | `ValueNet` | 24-37 | 状态价值网络，输入状态输出标量 V(s)，不依赖具体动作。 |
| 方法 {#blk0284} | `ValueNet.__init__` | 30-33 | 建立 state_dim→hidden_dim 的 fc1 与 hidden_dim→1 的 fc2。 |
| 方法 {#blk0285} | `ValueNet.forward` | 35-37 | fc1 后接 ReLU 再经 fc2，输出形状 [B, 1] 的状态价值估计。 |

**`08_PPO/CartPole/practice.py`** · 235 行 · 教程原版单文件 PPO 脚本，同时包含离散 CartPole 与连续 Pendulum 两套网络、智能体和顶层训练流程，供对照练习。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0286} | 导入与常量 | 1-7 | 导入旧版 gym、torch、F、numpy、matplotlib.pyplot 与 rl_utils；本文件不使用 config 模块，超参数散落在模块级脚本中。 |
| 类 {#blk0287} | `PolicyNet` | 9-17 | 教程版离散策略网络，两层线性层加 ReLU、softmax 输出动作概率。 |
| 方法 {#blk0288} | `PolicyNet.__init__` | 10-13 | 调用 super 初始化 Module，建立 fc1（state_dim→hidden_dim）与 fc2（hidden_dim→action_dim）。 |
| 方法 {#blk0289} | `PolicyNet.forward` | 15-17 | 先 ReLU(fc1(x))，再对 fc2 输出在 dim=1 做 softmax 得到动作概率。 |
| 类 {#blk0290} | `ValueNet` | 20-28 | 教程版状态价值网络，输出 V(s)。 |
| 方法 {#blk0291} | `ValueNet.__init__` | 21-24 | 建立 fc1（state_dim→hidden_dim）与 fc2（hidden_dim→1）。 |
| 方法 {#blk0292} | `ValueNet.forward` | 26-28 | ReLU(fc1(x)) 后过 fc2，返回形状 [B, 1] 的价值。 |
| 类 {#blk0293} | `PPO` | 31-87 | 教程版离散动作 PPO，采用截断（clip）方式，持有 actor/critic 与两个 Adam 优化器。 |
| 方法 {#blk0294} | `PPO.__init__` | 33-45 | 创建 PolicyNet 与 ValueNet 并搬到 device，分别建立 Adam 优化器，保存 gamma、lmbda、epochs、eps、device 等超参数。 |
| 方法 {#blk0295} | `PPO.take_action` | 47-52 | 用 torch.tensor([state]) 构造 batch 输入 actor 得到概率，Categorical 采样后返回 action.item()。 |
| 方法 {#blk0296} | `PPO.update` | 54-87 | 把 transition_dict 转张量，计算 td_target= r+gamma\*V(s')\*(1-done) 与 td_delta，经 rl_utils.compute_advantage 得优势；用 log(actor(states).gather(1, actions)) 得到 old_log_probs，循环 epochs 次以 ratio、clamp 截断构造 surrogate 并取 min 作为 actor 损失，critic 用 MSE 拟合 td_target，分别反向传播更新。 |
| 导入与常量 {#blk0297} | 模块级 CartPole 训练脚本 | 89-123 | 在模块顶层直接定义 CartPole 超参数（actor_lr=1e-3、critic_lr=1e-2、epochs=10、eps=0.2 等），用 gym.make('CartPole-v0') 建环境并设种子，构造 PPO 后调用 rl_utils.train_on_policy_agent 训练 500 个 episode，随后绘制原始回报与 moving_average(9) 两条曲线。 |
| 类 {#blk0298} | `PolicyNetContinuous` | 125-136 | 连续动作策略网络，输出正态分布的均值与标准差。 |
| 方法 {#blk0299} | `PolicyNetContinuous.__init__` | 126-130 | 建立共享特征层 fc1，以及分别输出均值的 fc_mu 和输出标准差的 fc_std（均为 hidden_dim→action_dim）。 |
| 方法 {#blk0300} | `PolicyNetContinuous.forward` | 132-136 | ReLU 过 fc1 后，mu 用 2.0\*tanh(fc_mu(x)) 限制在 [-2, 2]，std 用 softplus(fc_std(x)) 保证为正，返回 (mu, std)。 |
| 类 {#blk0301} | `PPOContinuous` | 139-200 | 教程版连续动作 PPO，actor 输出 Normal 分布，并对 Pendulum 奖励做平移缩放。 |
| 方法 {#blk0302} | `PPOContinuous.__init__` | 141-154 | 构造 PolicyNetContinuous 与 ValueNet 并搬到 device，分别建立 Adam 优化器，保存 gamma、lmbda、epochs、eps、device。 |
| 方法 {#blk0303} | `PPOContinuous.take_action` | 156-161 | 把 state 包成 [1, state_dim] 张量送入 actor 得 mu、sigma，构造 Normal 分布采样动作，并返回单元素列表 [action.item()]。 |
| 方法 {#blk0304} | `PPOContinuous.update` | 163-200 | 转换 states/actions/rewards/next_states/dones 为张量，先做 rewards=(rewards+8.0)/8.0 的奖励缩放，再算 td_target 与 td_delta 并经 rl_utils.compute_advantage 得优势；以 Normal(mu.detach(), std.detach()) 的 log_prob 作为 old_log_probs，循环 epochs 次计算 ratio 与截断 surrogate 得 actor 损失、critic MSE 损失，并分别反向传播更新。 |
| 导入与常量 {#blk0305} | 模块级 Pendulum 训练脚本 | 202-236 | 在模块顶层定义连续控制超参数（actor_lr=1e-4、critic_lr=5e-3、gamma=0.9、num_episodes=2000 等），用 gym.make('Pendulum-v0') 建环境并按 action_space.shape 取 action_dim，构造 PPOContinuous 训练后绘制原始回报与 moving_average(21) 平滑曲线。 |

**`08_PPO/CartPole/rl_utils.py`** · 31 行 · 提供回报滑动平均与 GAE 优势估计两个通用工具函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0306} | 导入与常量 | 1-2 | 导入 numpy 与 torch，无模块级常量。 |
| 函数 {#blk0307} | `moving_average` | 5-15 | 把输入转成 float32 数组；长度小于 window_size 时原样返回，否则用 cumsum 差分求中段窗口均值，并分别用 radius 处理首尾不足一个窗口的部分，最后 concatenate 成长度与原序列对齐的平滑序列。 |
| 函数 {#blk0308} | `compute_advantage` | 18-31 | 把 td_delta 转 numpy 一维数组，从最后一步向前按 A[t]=td_delta[t]+gamma\*lmbda\*A[t+1] 递推，逆序后转成 float32 的 torch 张量返回形状 [T] 的优势。 |

**`08_PPO/CartPole/trainer.py`** · 46 行 · 实现 on-policy 训练循环：采集完整 episode 后调用智能体 update 并记录回报。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0309} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0310} | `train_on_policy_agent` | 5-46 | 外层循环 num_episodes 个 episode，先 env.reset() 取 state，内层 while 直到 terminated 或 truncated：agent.take_action 采样、env.step 得五元组、把 (s,a,r,s',done) 累积到 transition_dict；episode 结束后调用 agent.update，记录该 episode 总回报，每 10 个 episode 用 tqdm 显示最近 10 次平均回报，最终返回 return_list。 |

**`08_PPO/CartPole/visualization.py`** · 20 行 · 把训练回报画成原始曲线与滑动平均曲线的对比图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0311} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与 rl_utils.moving_average，无模块级常量。 |
| 函数 {#blk0312} | `plot_training_results` | 6-20 | 创建 10x4 画布的 1x2 子图：左图绘制逐 episode 原始回报，右图绘制 moving_average(return_list, 9) 的平滑回报，均标注 Episodes/Returns 与标题，tight_layout 后 plt.show() 显示。 |

**`08_PPO/Pendulum/agents.py`** · 124 行 · 实现连续动作 PPO 智能体，使用 Normal 策略、动作区间映射与奖励缩放完成 actor-critic 更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0313} | 导入与常量 | 1-6 | 导入 numpy、torch、torch.nn.functional，以及本地 rl_utils 和 networks 中的 PolicyNetContinuous、ValueNet；无模块级常量。 |
| 类 {#blk0314} | `PPOContinuous` | 9-124 | 连续动作 PPO 智能体，actor 输出 Normal 分布参数，critic 估计 V(s)，共享 PPO-Clip 目标与 GAE，并保存动作上下界与奖励平移缩放系数。 |
| 方法 {#blk0315} | `PPOContinuous.__init__` | 16-41 | 用 action_low/action_high 构造 PolicyNetContinuous 与 ValueNet 并搬到 device，建立两个 Adam 优化器，把动作上下界存为 device 上的 float32 张量，并保存 gamma、lmbda、epochs、eps、reward_shift、reward_scale。 |
| 方法 {#blk0316} | `PPOContinuous._distribution` | 44-47 | 静态方法：调用 actor(states) 得到 mean 与 std，返回 torch.distributions.Normal(mean, std)，两者形状均为 [B, action_dim]。 |
| 方法 {#blk0317} | `PPOContinuous.take_action` | 49-58 | 把单个 state 转成 [1, state_dim] 张量，用 _distribution 构造 Normal 后采样动作，squeeze 并 detach 后返回 numpy 数组；刻意不 clamp，以保持采样动作与 log_prob 一致。 |
| 方法 {#blk0318} | `PPOContinuous._log_probs` | 61-64 | 静态方法：用 _distribution 对 states 构造 Normal，对 actions 求 log_prob 后在动作维 sum 并 keepdim，得到联合动作的 [B, 1] 对数概率。 |
| 方法 {#blk0319} | `PPOContinuous.update` | 66-124 | 把 transition_dict 的 states/actions（float 而非 long）/rewards/next_states/dones 转张量；按 (reward+shift)/scale 缩放奖励后计算 td_target 与 td_delta，经 rl_utils.compute_advantage 得 GAE 并 unsqueeze 成 [T,1]；记录 detach 的 old_log_probs 后循环 epochs 次，用 ratio 与 clamp 后的 surrogate 取 min 得 actor 损失、MSE 拟合 td_target 得 critic 损失，分别反向传播并 step。 |

**`08_PPO/Pendulum/config.py`** · 19 行 · 集中存放 Pendulum 连续控制实验的超参数、环境名与设备选择。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0320} | 导入与常量 | 1-19 | 导入 torch，定义 ACTOR_LR=1e-4、CRITIC_LR=5e-3、NUM_EPISODES=2000、HIDDEN_DIM=128、GAMMA=0.9、LMBDA=0.9、EPOCHS=10、EPS=0.2、REWARD_SHIFT=8.0、REWARD_SCALE=8.0、SEED=0、ENV_NAME='Pendulum-v1'，并按 CUDA 可用性设置 DEVICE。 |

**`08_PPO/Pendulum/environment.py`** · 16 行 · 创建 Pendulum 环境并校验其动作空间为 Box 连续型，返回维度与动作上下界。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0321} | 导入与常量 | 1-1 | 仅导入 gymnasium 的 gym 模块，无模块级常量。 |
| 函数 {#blk0322} | `make_environment` | 4-16 | 用 gym.make(env_name) 创建环境，若 action_space 无 shape 或 shape 为 None 则抛 TypeError 提示需要 Box 动作空间；随后取 state_dim、action_dim，并复制 action_space.low/high 作为动作上下界，连同 env 一起返回。 |

**`08_PPO/Pendulum/main.py`** · 49 行 · Pendulum 训练入口，装配带动作上下界与奖励缩放的连续 PPO 并启动训练与绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0323} | 导入与常量 | 1-10 | 导入 random、numpy、torch，以及本地 config、PPOContinuous、make_environment、train_on_policy_agent、plot_training_results；无模块级常量。 |
| 函数 {#blk0324} | `set_seed` | 13-16 | 依次设置 random、numpy、torch 的随机种子以保证实验可复现。 |
| 函数 {#blk0325} | `main` | 19-45 | 设种子后调用 make_environment 取回 env、维度与 action_low/high，env.reset(seed=...) 后按 config 的超参数（含 REWARD_SHIFT、REWARD_SCALE）构造 PPOContinuous，训练 NUM_EPISODES 个 episode，绘制回报曲线并关闭环境。 |
| 入口 {#blk0326} | if __name__ == '__main__' | 48-49 | 脚本直接运行时调用 main() 启动 Pendulum 的连续 PPO 训练。 |

**`08_PPO/Pendulum/networks.py`** · 42 行 · 定义连续 PPO 的正态策略网络（含动作区间映射）与状态价值网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0327} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional，无模块级常量。 |
| 类 {#blk0328} | `PolicyNetContinuous` | 5-29 | 连续动作策略网络，输出正态分布的均值与标准差，并把 tanh 输出映射到环境动作区间。 |
| 方法 {#blk0329} | `PolicyNetContinuous.__init__` | 12-21 | 建立 fc1、fc_mu、fc_std 三个线性层，并把 action_low/high 换算成 register_buffer 的 action_scale=(high-low)/2 与 action_bias=(high+low)/2。 |
| 方法 {#blk0330} | `PolicyNetContinuous.forward` | 23-29 | ReLU 过 fc1 得特征，均值 mu = bias + scale\*tanh(fc_mu(features)) 落入动作区间，标准差 std = softplus(fc_std(features))+1e-5 保证为正，返回 (mu, std)。 |
| 类 {#blk0331} | `ValueNet` | 32-42 | 连续 PPO 使用的状态价值网络，只评价状态本身，输出 V(s)。 |
| 方法 {#blk0332} | `ValueNet.__init__` | 35-38 | 建立 state_dim→hidden_dim 的 fc1 与 hidden_dim→1 的 fc2。 |
| 方法 {#blk0333} | `ValueNet.forward` | 40-42 | fc1 后接 ReLU 再经 fc2，返回形状 [B, 1] 的状态价值。 |

**`08_PPO/Pendulum/rl_utils.py`** · 27 行 · 提供回报滑动平均与 GAE 优势估计工具函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0334} | 导入与常量 | 1-2 | 导入 numpy 与 torch，无模块级常量。 |
| 函数 {#blk0335} | `moving_average` | 5-15 | 将序列转为 float32 数组，长度不足 window_size 时直接返回；否则用 cumsum 差分求中段窗口均值，并用 radius 计算首尾不足窗口的均值，拼接成与原序列等长的平滑结果。 |
| 函数 {#blk0336} | `compute_advantage` | 18-27 | 把 td_delta 展平为一维 numpy 数组，从末尾向前按 A[t]=td_delta[t]+gamma\*lmbda\*A[t+1] 递推，逆序后转成 float32 torch 张量返回形状 [T] 的优势序列。 |

**`08_PPO/Pendulum/trainer.py`** · 38 行 · 实现连续 PPO 的 on-policy 训练循环，采集完整 episode 后更新并统计回报。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0337} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0338} | `train_on_policy_agent` | 5-38 | 对每个 episode 先 env.reset() 取 state，循环直到 terminated/truncated：agent.take_action 返回连续动作向量、env.step 得五元组，并把转移累积到 transition_dict；episode 结束后 agent.update 更新，记录 episode_return，每 10 个 episode 用 tqdm 显示最近 10 次平均回报，最后返回 return_list。 |

**`08_PPO/Pendulum/visualization.py`** · 20 行 · 绘制 Pendulum 训练的原始回报与窗口 21 的滑动平均回报。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0339} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与 rl_utils.moving_average，无模块级常量。 |
| 函数 {#blk0340} | `plot_training_results` | 6-20 | 在 1x2 子图中分别绘制逐 episode 的原始回报和 moving_average(return_list, 21) 的平滑回报，设置坐标轴标签与标题后 tight_layout 并 plt.show()。 |

## 09_DDPG：标准版与 CartPole 教学版如何组织？ {#algo09 span=12}

DDPG 的 CartPole 与 Pendulum 两套教学实现：Actor-Critic 双网络加双 target 网络、软更新、经验回放，并配套噪声、训练循环与绘图模块。 本目录共 19 个 Python 文件、96 个代码块条目。

**`09_DDPG/CartPole/agents.py`** · 78 行 · CartPole 版 DDPG 智能体，封装四个网络、优化器、探索噪声以及 critic/actor 的更新步骤。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0341} | 导入与常量 | 1-6 | 导入 numpy、torch 与 F，并从 networks 引入 Actor/Critic、从 noise 引入 NormalNoise/OUNoise；模块内无自定义常量。 |
| 类 {#blk0342} | `DDPG` | 9-78 | 车杆任务的 DDPG 适配器类：actor 输出一维连续信号，由环境层再映射成左右两个离散动作。 |
| 方法 {#blk0343} | `DDPG.__init__` | 12-33 | 接收状态/动作维度、动作上下界、学习率、gamma、tau 与噪声参数；构建 actor/critic 及各自 target 网络并复制参数，创建两个 Adam 优化器，按 noise_type 选择 OUNoise 或 NormalNoise，非法取值抛 ValueError。 |
| 方法 {#blk0344} | `DDPG.reset_noise` | 35-36 | 每回合开始时调用噪声对象的 reset，清除 OU 噪声的历史状态。 |
| 方法 {#blk0345} | `DDPG.take_action` | 38-46 | 把状态转成 (1, state_dim) 张量，no_grad 下经 actor 得到动作；explore 为真时叠加噪声采样，最后裁剪到 [-1, 1] 并返回 numpy 数组。 |
| 方法 {#blk0346} | `DDPG.soft_update` | 48-53 | 对 target 与 online 网络的同名参数逐一对齐，执行 target = (1-tau)\*target + tau\*online 的原地软更新。 |
| 方法 {#blk0347} | `DDPG.update` | 55-78 | 把回放批次转为张量，用 target_actor/target_critic 计算 q_targets = r + gamma\*Q'\*(1-done)；先以 MSE 更新 critic，再以 -mean(Q(s, actor(s))) 更新 actor，最后对两对网络各做一次软更新。 |

**`09_DDPG/CartPole/config.py`** · 18 行 · CartPole DDPG 的超参数配置模块，集中定义学习率、回合数、回放池与噪声设置。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0348} | 导入与常量 | 1-18 | 导入 torch 后定义 actor_lr=3e-4、critic_lr=3e-3、NUM_EPISODES=500、HIDDEN_DIM=64、GAMMA=0.98、TAU=0.005、BUFFER_SIZE/MINIMAL_SIZE/BATCH_SIZE、NOISE_TYPE='normal'、NOISE_SIGMA、OU_THETA、SEED、ENV_NAME='CartPole-v1'，并按 CUDA 可用性选择 DEVICE。 |

**`09_DDPG/CartPole/environment.py`** · 17 行 · 创建 CartPole 环境，并为 DDPG 提供一维连续动作代理空间及其离散化转换。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0349} | 导入与常量 | 1-2 | 导入 numpy 与 gymnasium；无模块级常量。 |
| 函数 {#blk0350} | `make_environment` | 5-12 | 以 env_name 创建 gym 环境，校验动作空间为两个离散动作（否则 TypeError），返回 env、状态维度、动作维度 1 以及 [-1.0] / [1.0] 的动作上下界。 |
| 函数 {#blk0351} | `continuous_to_discrete` | 15-17 | 取连续动作第一个分量，≥0 映射为 1，否则为 0，作为 CartPole 的左右推力动作。 |

**`09_DDPG/CartPole/main.py`** · 40 行 · CartPole DDPG 的运行入口，负责设置随机种子、装配环境/智能体/回放池、调用训练并绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0352} | 导入与常量 | 1-11 | 导入 random、numpy、torch 以及本目录的 config、DDPG、make_environment、ReplayBuffer、train_off_policy_agent、plot_training_results。 |
| 函数 {#blk0353} | `set_seed` | 14-17 | 同时设置 random、numpy 与 torch 的随机种子，保证实验可复现。 |
| 函数 {#blk0354} | `main` | 20-36 | 设定种子后创建 CartPole 环境，用 config 中的参数实例化 DDPG 与 ReplayBuffer，调用 train_off_policy_agent 得到每回合回报，随后绘图并关闭环境。 |
| 入口 {#blk0355} | if __name__ == '__main__' | 39-40 | 脚本直接运行时调用 main()。 |

**`09_DDPG/CartPole/networks.py`** · 35 行 · 定义 CartPole DDPG 使用的确定性 Actor 网络与 Q 值 Critic 网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0356} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional 作为 F；无模块级常量。 |
| 类 {#blk0357} | `Actor` | 5-19 | 确定性策略网络，输出映射到动作上下界之间的连续动作。 |
| 方法 {#blk0358} | `Actor.__init__` | 8-15 | 构建两层全连接（state_dim→hidden_dim→len(action_low)），并把动作缩放 (high-low)/2 与偏移 (high+low)/2 注册为 buffer。 |
| 方法 {#blk0359} | `Actor.forward` | 17-19 | 对状态做 ReLU 隐层后接 tanh 得到 [-1,1] 输出，再按 action_bias + action_scale\*tanh 映射到动作区间。 |
| 类 {#blk0360} | `Critic` | 22-35 | Q(s,a) 价值网络，接收状态与动作并输出标量价值。 |
| 方法 {#blk0361} | `Critic.__init__` | 25-29 | 第一层输入维度为 state_dim+action_dim，第二层为 hidden_dim→hidden_dim，最后 fc_out 输出 1 维。 |
| 方法 {#blk0362} | `Critic.forward` | 31-35 | 沿特征维拼接状态与动作，经过两层 ReLU 隐层后由 fc_out 返回 Q 值。 |

**`09_DDPG/CartPole/noise.py`** · 34 行 · 提供 DDPG 探索用的高斯噪声和 Ornstein-Uhlenbeck 噪声两种实现。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0363} | 导入与常量 | 1 | 仅导入 numpy；无模块级常量。 |
| 类 {#blk0364} | `NormalNoise` | 4-15 | 每一步独立采样的高斯噪声生成器。 |
| 方法 {#blk0365} | `NormalNoise.__init__` | 7-9 | 保存动作维度和噪声标准差 sigma。 |
| 方法 {#blk0366} | `NormalNoise.reset` | 11-12 | 空实现，仅为与其他噪声类保持统一接口。 |
| 方法 {#blk0367} | `NormalNoise.sample` | 14-15 | 返回 sigma \* randn(action_dim) 的独立高斯噪声向量。 |
| 类 {#blk0368} | `OUNoise` | 18-34 | 带时间相关性的 Ornstein-Uhlenbeck 噪声，维护跨步的噪声状态。 |
| 方法 {#blk0369} | `OUNoise.__init__` | 21-26 | 保存 action_dim、sigma、theta、mu，并把内部状态初始化为零向量。 |
| 方法 {#blk0370} | `OUNoise.reset` | 28-29 | 将内部状态原地清零，用于每回合重新开始。 |
| 方法 {#blk0371} | `OUNoise.sample` | 31-34 | 按 state += theta\*(mu-state) + sigma\*randn 更新噪声状态，并返回其副本。 |

**`09_DDPG/CartPole/rl_utils.py`** · 34 行 · 提供 CartPole DDPG 的经验回放池和滑动平均工具函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0372} | 导入与常量 | 1-4 | 导入 collections、random 与 numpy；无模块级常量。 |
| 类 {#blk0373} | `ReplayBuffer` | 7-22 | 基于 deque 的定长 off-policy 经验回放池。 |
| 方法 {#blk0374} | `ReplayBuffer.__init__` | 10-11 | 用 capacity 创建 maxlen 受限的 deque 作为存储。 |
| 方法 {#blk0375} | `ReplayBuffer.add` | 13-14 | 把 (state, action, reward, next_state, done) 五元组追加进缓冲区，超出容量时自动丢弃最旧样本。 |
| 方法 {#blk0376} | `ReplayBuffer.sample` | 16-19 | 用 random.sample 无放回抽取 batch_size 条转移，zip 拆解后转成 numpy 数组的五个返回值。 |
| 方法 {#blk0377} | `ReplayBuffer.size` | 21-22 | 返回当前缓冲区内的样本条数。 |
| 函数 {#blk0378} | `moving_average` | 25-34 | 对序列做指定窗口的滑动平均；长度不足时原样返回，否则用累积和计算中段，并单独处理首尾两段后拼接输出。 |

**`09_DDPG/CartPole/trainer.py`** · 38 行 · 实现 CartPole 上带连续动作代理器的 DDPG off-policy 训练循环。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0379} | 导入与常量 | 1-4 | 导入 numpy、tqdm 以及本目录的 continuous_to_discrete，用于动作离散化。 |
| 函数 {#blk0380} | `train_off_policy_agent` | 7-38 | 外层按 num_episodes 循环（可传 seed 逐回合重置环境），每回合先 reset_noise；内层用 agent 连续动作换算成 0/1 动作执行 env.step，缓冲连续动作与 terminated 标志（截断处仍 bootstrap），缓冲区达到 minimal_size 后每步采样 batch 调用 agent.update，最后返回每回合回报列表并用 tqdm 显示最近 10 回合均值。 |

**`09_DDPG/CartPole/visualization.py`** · 20 行 · 绘制 CartPole DDPG 训练回报曲线与滑动平均曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0381} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与 rl_utils 的 moving_average。 |
| 函数 {#blk0382} | `plot_training_results` | 6-20 | 创建 1x2 子图：左侧画每回合原始回报，右侧画窗口为 window_size 的滑动平均，最后 tight_layout 并 plt.show 显示。 |

**`09_DDPG/Pendulum/agents.py`** · 89 行 · Pendulum 版 DDPG 智能体，封装四个网络、噪声选择以及 critic/actor 的梯度更新与软更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0383} | 导入与常量 | 1-6 | 导入 numpy、torch、F，以及本目录 networks 的 Actor/Critic 与 noise 的两种噪声类；无模块级常量。 |
| 类 {#blk0384} | `DDPG` | 9-89 | 标准 DDPG 智能体实现，包含确定性 actor、Q critic 及各自 target 网络。 |
| 方法 {#blk0385} | `DDPG.__init__` | 12-37 | 构建 actor/critic 与 target 网络并将参数复制到 target，创建两个 Adam 优化器，保存 gamma/tau/device/action_dim，根据 noise_type 选择 OUNoise 或 NormalNoise，其余取值报 ValueError。 |
| 方法 {#blk0386} | `DDPG.reset_noise` | 39-40 | 调用噪声对象的 reset，在每回合开始时重置噪声状态。 |
| 方法 {#blk0387} | `DDPG.take_action` | 42-52 | 将状态转为 (1, state_dim) 张量并 no_grad 前向 actor；explore 为真时加噪声，最后用 actor 的 action_bias±action_scale 裁剪，保证动作落在环境允许区间。 |
| 方法 {#blk0388} | `DDPG.soft_update` | 54-60 | 对 target 与 online 参数的每一对执行原地 target = (1-tau)\*target + tau\*online 的软更新。 |
| 方法 {#blk0389} | `DDPG.update` | 62-89 | 批次转张量后，用 target 网络在 no_grad 下算 Bellman 目标；先 MSE 更新 critic，再以 -mean(Q(s, actor(s))) 更新 actor，最后软更新 actor 与 critic 的 target 网络。 |

**`09_DDPG/Pendulum/config.py`** · 18 行 · Pendulum DDPG 的超参数配置文件。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0390} | 导入与常量 | 1-18 | 导入 torch 后定义 actor_lr、critic_lr、NUM_EPISODES=200、HIDDEN_DIM、GAMMA、TAU、缓冲区与批大小、NOISE_TYPE/SIGMA、OU_THETA、SEED、ENV_NAME='Pendulum-v1'，并按 CUDA 可用性设置 DEVICE。 |

**`09_DDPG/Pendulum/environment.py`** · 17 行 · 创建 Pendulum 连续控制环境并返回其状态/动作维度与动作上下界。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0391} | 导入与常量 | 1 | 仅导入 gymnasium；无模块级常量。 |
| 函数 {#blk0392} | `make_environment` | 4-17 | 按 env_name 创建环境，校验动作空间具有 shape（否则抛 TypeError），读取 observation_space 与 action_space 的维度，并返回 env、state_dim、action_dim 以及动作空间的 low/high 副本。 |

**`09_DDPG/Pendulum/main.py`** · 41 行 · Pendulum DDPG 的运行入口，装配环境、智能体、回放池并启动训练与绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0393} | 导入与常量 | 1-11 | 导入 random、numpy、torch 及本目录的 config、DDPG、make_environment、ReplayBuffer、训练函数与绘图函数。 |
| 函数 {#blk0394} | `set_seed` | 14-17 | 统一设置 random、numpy 与 torch 的随机种子以复现实验。 |
| 函数 {#blk0395} | `main` | 20-37 | 设置种子并创建 Pendulum 环境，再用 seed 重置一次；用 config 参数构造 DDPG 和 ReplayBuffer，调用 train_off_policy_agent 获得回报列表，绘制结果后关闭环境。 |
| 入口 {#blk0396} | if __name__ == '__main__' | 40-41 | 脚本直接执行时调用 main()。 |

**`09_DDPG/Pendulum/networks.py`** · 37 行 · 定义 Pendulum DDPG 的确定性策略网络与状态-动作价值网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0397} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional；无模块级常量。 |
| 类 {#blk0398} | `Actor` | 5-20 | 确定性策略 mu(s)，输出环境动作范围内的连续动作。 |
| 方法 {#blk0399} | `Actor.__init__` | 8-15 | 构造 state_dim→hidden_dim→len(action_low) 的两层网络，并将动作缩放与偏移注册为 buffer 以便随设备迁移。 |
| 方法 {#blk0400} | `Actor.forward` | 17-20 | ReLU 隐层后接 tanh 限幅到 [-1,1]，再经 action_bias + action_scale\*tanh 映射到动作区间。 |
| 类 {#blk0401} | `Critic` | 23-37 | Q(s,a) 网络，同时接收状态与动作并输出标量价值。 |
| 方法 {#blk0402} | `Critic.__init__` | 26-30 | 三层全连接：state_dim+action_dim→hidden_dim、hidden_dim→hidden_dim、hidden_dim→1。 |
| 方法 {#blk0403} | `Critic.forward` | 32-37 | 拼接状态与动作后过两层 ReLU 隐层，由 fc_out 输出每个状态-动作对的 Q 值。 |

**`09_DDPG/Pendulum/noise.py`** · 41 行 · 提供 Pendulum DDPG 探索使用的高斯噪声与 OU 噪声。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0404} | 导入与常量 | 1 | 仅导入 numpy；无模块级常量。 |
| 类 {#blk0405} | `NormalNoise` | 4-16 | 逐步独立重采样的高斯探索噪声。 |
| 方法 {#blk0406} | `NormalNoise.__init__` | 7-9 | 保存动作维度与标准差 sigma。 |
| 方法 {#blk0407} | `NormalNoise.reset` | 11-13 | 空实现，仅为统一接口，因为高斯噪声无跨时间状态。 |
| 方法 {#blk0408} | `NormalNoise.sample` | 15-16 | 返回 sigma \* randn(action_dim) 的噪声向量。 |
| 类 {#blk0409} | `OUNoise` | 19-41 | Ornstein-Uhlenbeck 噪声，按文档字符串中的递推式产生时间相关的探索扰动。 |
| 方法 {#blk0410} | `OUNoise.__init__` | 26-31 | 保存 action_dim、sigma、theta、mu，并把内部噪声状态初始化为零向量。 |
| 方法 {#blk0411} | `OUNoise.reset` | 33-35 | 每回合开始时把内部状态填零，清除上一条轨迹的噪声。 |
| 方法 {#blk0412} | `OUNoise.sample` | 37-41 | 计算漂移 theta\*(mu-state) 与随机项 sigma\*randn，更新 state 后返回其副本。 |

**`09_DDPG/Pendulum/practice.py`** · 112 行 · 单文件版 DDPG 练习脚本，把网络、智能体、超参数与训练流程全部写在同一文件内。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0413} | 导入与常量 | 1-9 | 导入 random、gym、numpy、tqdm、torch 及其 nn/F、matplotlib.pyplot 以及本目录的 rl_utils；无模块级常量定义（常量集中在文件末尾）。 |
| 类 {#blk0414} | `PolicyNet` | 11-20 | 确定性策略网络，用 tanh 乘以 action_bound 输出有界动作。 |
| 方法 {#blk0415} | `PolicyNet.__init__` | 12-16 | 构建 state_dim→hidden_dim→action_dim 的两层全连接，并保存标量 action_bound 作为动作最大值。 |
| 方法 {#blk0416} | `PolicyNet.forward` | 18-20 | ReLU 隐层后接 tanh 并乘以 action_bound，返回环境可接受的连续动作。 |
| 类 {#blk0417} | `QValueNet` | 23-34 | 状态-动作价值网络，输出标量 Q 值。 |
| 方法 {#blk0418} | `QValueNet.__init__` | 24-28 | 构造输入维度 state_dim+action_dim 的三层全连接网络，末端输出 1 维。 |
| 方法 {#blk0419} | `QValueNet.forward` | 30-34 | 沿特征维拼接状态 x 与动作 a，经两层 ReLU 后由 fc_out 输出 Q 值。 |
| 类 {#blk0420} | `DDPG` | 36-86 | 练习用 DDPG 智能体，内含 actor/critic 及 target 网络、双优化器与噪声探索逻辑。 |
| 方法 {#blk0421} | `DDPG.__init__` | 38-53 | 创建 PolicyNet/QValueNet 与其 target 副本并复制参数，建立两个 Adam 优化器，保存 gamma、sigma、tau、action_dim 与 device。 |
| 方法 {#blk0422} | `DDPG.take_action` | 55-60 | 把状态包成 (1, state_dim) 张量前向 actor 取标量动作，再叠加 sigma\*randn 的高斯噪声用于探索，返回 numpy 标量。 |
| 方法 {#blk0423} | `DDPG.soft_update` | 62-64 | 逐个参数执行 target = (1-tau)\*target + tau\*online 的软更新。 |
| 方法 {#blk0424} | `DDPG.update` | 66-86 | 从 transition_dict 取五类数据并整理形状，用 target 网络计算 q_targets；先 MSE 更新 critic，再以 -mean(Q(s,actor(s))) 更新 actor，最后对 actor/critic 的 target 网络各做软更新。 |
| 导入与常量 {#blk0425} | 模块级常量与脚本初始化 | 88-112 | 定义 actor_lr、critic_lr、num_episodes、hidden_dim、gamma、tau、buffer_size、minimal_size、batch_size、sigma 与 device；随后创建 Pendulum-v0 环境、设置各库种子、构建回放池与 agent，并直接调用 rl_utils.train_off_policy_agent 执行训练（文件无 __main__ 保护）。 |

**`09_DDPG/Pendulum/rl_utils.py`** · 40 行 · 提供 Pendulum DDPG 的经验回放池与滑动平均计算工具。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0426} | 导入与常量 | 1-4 | 导入 collections、random 与 numpy；无模块级常量。 |
| 类 {#blk0427} | `ReplayBuffer` | 7-28 | 基于 deque 的 off-policy 经验回放池。 |
| 方法 {#blk0428} | `ReplayBuffer.__init__` | 10-11 | 以 capacity 作为 maxlen 创建 deque 存储。 |
| 方法 {#blk0429} | `ReplayBuffer.add` | 13-14 | 追加一条 (state, action, reward, next_state, done) 转移，满容量时自动淘汰最旧数据。 |
| 方法 {#blk0430} | `ReplayBuffer.sample` | 16-25 | 随机无放回抽取 batch_size 条转移，拆包后转换为五个 numpy 数组返回。 |
| 方法 {#blk0431} | `ReplayBuffer.size` | 27-28 | 返回缓冲区当前样本数。 |
| 函数 {#blk0432} | `moving_average` | 31-40 | 序列长度不足 window_size 时原样返回；否则用累积和计算中间段均值，并分别计算开头与结尾的逐步平均后拼接为完整序列。 |

**`09_DDPG/Pendulum/trainer.py`** · 46 行 · 实现 Pendulum DDPG 的 off-policy 训练循环，负责交互、回放与按步更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0433} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条；无模块级常量。 |
| 函数 {#blk0434} | `train_off_policy_agent` | 5-46 | 按 num_episodes 循环，每回合先 agent.reset_noise 并按 seed+episode 重置环境；内层用 take_action 采样连续动作执行 env.step，把 terminated（而非 truncated）作为 done 存入回放池以便截断处继续 bootstrap，缓冲区达到 minimal_size 后每步采样并调用 agent.update，最终返回每回合回报列表并用 tqdm 显示最近十回合均值。 |

**`09_DDPG/Pendulum/visualization.py`** · 20 行 · 绘制 Pendulum DDPG 的原始回报与滑动平均回报曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0435} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot 与 rl_utils.moving_average。 |
| 函数 {#blk0436} | `plot_training_results` | 6-20 | 建立 1x2 子图分别绘制每回合回报与 window_size 窗口的滑动平均，设置坐标轴标签与标题后 tight_layout 并 show。 |

## 10_TD3：双 critic 与延迟更新落在哪里？ {#algo10 span=12}

TD3（两个 critic 取最小值、target policy smoothing、延迟策略更新）在 Pendulum-v1 上的模块化实现，由 config 超参数、网络、噪声、回放池、训练/评估循环和可视化组成。 本目录共 10 个 Python 文件、59 个代码块条目。

**`10_TD3/Pendulum/agents.py`** · 213 行 · 定义 TD3 智能体，封装六个网络、优化器、动作选择、target 计算与单批更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0437} | 导入与常量 | 1-7 | 导入 numpy、torch 和 torch.nn.functional，并从 networks 引入 Actor/Critic、从 noise 引入 GaussianNoise 与 smooth_target_action；模块内无常量定义。 |
| 类 {#blk0438} | `TD3` | 9-213 | 封装 actor、target_actor、两个 online critic、两个 target critic、三个 Adam 优化器、动作边界与延迟更新计数，对外提供动作选择、target 计算和批量更新接口。 |
| 方法 {#blk0439} | `TD3.__init__` | 24-68 | 按 state_dim/hidden_dim/action_dim 构建 Actor 与两套 Critic，并把 online 参数复制给 target；创建三个 Adam 优化器，保存 gamma、tau、policy_delay、target_noise、noise_clip，把动作上下界转成设备上的张量并计算半幅 action_scale。 |
| 方法 {#blk0440} | `TD3.reset_noise` | 70-72 | 调用 GaussianNoise.reset()；高斯噪声没有历史状态，该接口只是与 DDPG 的训练循环保持统一。 |
| 方法 {#blk0441} | `TD3.take_action` | 74-94 | 把状态转成 [1, state_dim] 张量送入 actor 得到确定性动作；explore=True 时叠加一次高斯噪声采样，最后把动作裁剪到 [action_low, action_high] 并返回 NumPy 数组。 |
| 方法 {#blk0442} | `TD3.soft_update` | 96-103 | 对 online_net 与 target_net 的参数逐对执行 Polyak 平均 target ← (1-tau)·target + tau·online，使用 data 原地运算以避免把 target 纳入计算图。 |
| 方法 {#blk0443} | `TD3.calc_target` | 105-135 | 在 no_grad 下由 target_actor 给出下一状态动作，经 smooth_target_action 加截断噪声，再用两个 target critic 的较小 Q 值计算 r + gamma·(1-done)·min(Q1',Q2')。 |
| 方法 {#blk0444} | `TD3.update` | 137-213 | 先把 batch 转成张量，用共享的 clipped double-Q target 对两个 critic 各做 MSE 回归并分别 backward/step；每 policy_delay 次调用再临时冻结 critic，以 -Q1(s,actor(s)).mean() 更新 actor，随后软更新 actor 与两个 critic 的 target，返回 critic_loss 和 actor_loss。 |

**`10_TD3/Pendulum/config.py`** · 67 行 · 集中存放 TD3 实验的全部超参数、环境名、随机种子与设备选择。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0445} | 导入与常量 | 1-67 | 仅导入 torch，然后定义 actor/critic 学习率 1e-3、200 个 episode、隐藏层 128、gamma 0.99、tau 0.005、policy_delay 2、target_noise 0.2 与 noise_clip 0.5、探索噪声 0.1、buffer 10 万与 batch 100、评估 20 回合及评估种子 10000，并给出 SEED、ENV_NAME='Pendulum-v1' 与自动选择 cuda/cpu 的 DEVICE。 |

**`10_TD3/Pendulum/environment.py`** · 23 行 · 创建 Pendulum 环境并返回状态/动作维度与连续动作边界。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0446} | 导入与常量 | 1-2 | 导入 gymnasium，无模块级常量。 |
| 函数 {#blk0447} | `make_environment` | 4-23 | 调用 gym.make 创建环境，若 action_space 不是 gym.spaces.Box 则抛出 TypeError；否则读取 observation/action 维度，并用 copy() 返回动作下界与上界，防止后续原地修改影响环境本身。 |

**`10_TD3/Pendulum/main.py`** · 77 行 · TD3 实验入口：固定种子、组装智能体与回放池、训练、确定性评估并绘图。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0448} | 导入与常量 | 1-11 | 导入 random、numpy、torch、config，以及本目录的 TD3、make_environment、ReplayBuffer、evaluate_policy、train_off_policy_agent 和 plot_training_results。 |
| 函数 {#blk0449} | `set_seed` | 14-18 | 分别调用 random.seed、np.random.seed 和 torch.manual_seed 固定同一随机种子，保证实验可复现。 |
| 函数 {#blk0450} | `main` | 21-73 | 设置种子并创建环境，按 config 参数实例化 TD3 与 ReplayBuffer；调用 train_off_policy_agent 得到每回合回报，再用 evaluate_policy 以独立评估种子跑 20 回合，打印最后 10 回合均值与评估统计，最后绘图并 env.close()。 |
| 入口 {#blk0451} | if __name__ == '__main__' | 76-77 | 脚本直接运行时调用 main()。 |

**`10_TD3/Pendulum/networks.py`** · 60 行 · 定义 TD3 使用的确定性 Actor 与 Q(s,a) Critic 网络结构。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0452} | 导入与常量 | 1-2 | 导入 torch 与 torch.nn.functional，无模块级常量。 |
| 类 {#blk0453} | `Actor` | 5-37 | 确定性策略 mu(s)：两层 128 维隐藏层后经 tanh 压缩到 [-1,1]，再用注册为 buffer 的 action_scale/action_bias 映射到环境动作范围。 |
| 方法 {#blk0454} | `Actor.__init__` | 13-28 | 按 state_dim→hidden_dim→hidden_dim→action_dim 建立三层线性网络，并把 (high-low)/2 与 (high+low)/2 用 register_buffer 保存，使其随 to(device) 迁移且不参与训练。 |
| 方法 {#blk0455} | `Actor.forward` | 30-37 | 输入状态依次经过 fc1、fc2 的 ReLU 和 fc_out，再取 tanh 得到 [-1,1] 动作，返回 action_bias + action_scale \* squashed。 |
| 类 {#blk0456} | `Critic` | 40-60 | Q(s,a) 网络，把状态与动作拼接后经两层隐藏层输出一个标量 Q 值；TD3 会独立实例化两个这样的网络作为 Q1/Q2。 |
| 方法 {#blk0457} | `Critic.__init__` | 48-53 | 第一层输入维度为 state_dim + action_dim，第二层保持 hidden_dim，输出层为 1 维。 |
| 方法 {#blk0458} | `Critic.forward` | 55-60 | 沿第 1 维拼接 states 与 actions，经 ReLU(fc1)、ReLU(fc2) 后由 fc_out 输出形状 [batch_size, 1] 的 Q 值。 |

**`10_TD3/Pendulum/noise.py`** · 57 行 · 提供独立高斯探索噪声与 target policy smoothing 的截断噪声函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0459} | 导入与常量 | 1-2 | 导入 numpy 与 torch，无模块级常量。 |
| 类 {#blk0460} | `GaussianNoise` | 5-28 | 无状态的高斯探索噪声：每步按 sigma 重新采样，配合确定性 actor 形成行为策略。 |
| 方法 {#blk0461} | `GaussianNoise.__init__` | 18-20 | 保存动作维度 action_dim 与噪声标准差 sigma。 |
| 方法 {#blk0462} | `GaussianNoise.reset` | 22-24 | 空实现（pass），因为独立高斯噪声没有需要重置的时间相关状态。 |
| 方法 {#blk0463} | `GaussianNoise.sample` | 26-28 | 返回 sigma \* np.random.randn(action_dim) 的 float32 数组，避免与 torch float32 张量相加时发生类型提升。 |
| 函数 {#blk0464} | `smooth_target_action` | 31-57 | 对 target actor 的动作按 randn_like 加噪声并乘以 action_scale，把噪声截断到 ±noise_clip·action_scale，最后将 actions+noise 裁剪到 [action_low, action_high] 后返回。 |

**`10_TD3/Pendulum/practice.py`** · 225 行 · 把 TD3 全部流程压缩进单个脚本的教学版本，便于对照 Spinning Up 与单步调试三项改进。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0465} | 导入与常量 | 1-38 | 文件 docstring 说明这是单文件教学版；导入 collections、random、gymnasium、numpy、torch、F 与 tqdm，并在模块级写入与 config 相同的超参数（学习率、200 episodes、gamma、tau、policy_delay、噪声、buffer、batch）、ENV_NAME 与 DEVICE。 |
| 类 {#blk0466} | `Actor` | 41-58 | 用 nn.Sequential 搭建两层隐藏层的确定性策略，tanh 后用 buffer 中的 scale/bias 映射到动作范围。 |
| 方法 {#blk0467} | `Actor.__init__` | 42-55 | 构建 Linear-ReLU-Linear-ReLU-Linear 序列，并注册 (high-low)/2 与 (high+low)/2 两个 buffer。 |
| 方法 {#blk0468} | `Actor.forward` | 57-58 | 返回 bias + scale \* tanh(net(states))。 |
| 类 {#blk0469} | `Critic` | 61-73 | Q(s,a) 网络，用 Sequential 把拼接后的状态动作映射为标量 Q 值。 |
| 方法 {#blk0470} | `Critic.__init__` | 62-70 | 输入层维度为 state_dim+action_dim，两个隐藏层为 hidden_dim，输出 1 维。 |
| 方法 {#blk0471} | `Critic.forward` | 72-73 | 沿 dim=1 拼接 states 与 actions 后送入 net，返回 Q 值。 |
| 类 {#blk0472} | `ReplayBuffer` | 76-97 | 基于 deque(maxlen=capacity) 的回放池，采样时直接返回设备上的张量，供单文件版 TD3.update 使用。 |
| 方法 {#blk0473} | `ReplayBuffer.__init__` | 77-78 | 创建容量为 capacity 的 collections.deque，满后自动丢弃最旧经验。 |
| 方法 {#blk0474} | `ReplayBuffer.add` | 80-81 | 把一条完整 transition 追加进 buffer。 |
| 方法 {#blk0475} | `ReplayBuffer.sample` | 83-94 | 随机采样 batch_size 条经验并 zip 拆分，转成 float32 张量放到 DEVICE，其中 rewards 与 dones 用 view(-1,1) 变成列向量。 |
| 方法 {#blk0476} | `ReplayBuffer.__len__` | 96-97 | 返回当前 buffer 中的经验条数，供训练循环判断是否达到 minimal_size。 |
| 类 {#blk0477} | `TD3` | 100-186 | 单文件版 TD3，直接在类内使用模块级超参数与 DEVICE，维护 actor/两个 critic 及三个 target 网络和对应优化器。 |
| 方法 {#blk0478} | `TD3.__init__` | 101-124 | 创建 Actor、两个 Critic 及各自 target 并同步初始参数，建立三个 Adam 优化器，保存动作上下界与 action_scale，并把 update_count 初始化为 0。 |
| 方法 {#blk0479} | `TD3.take_action` | 126-138 | 状态张量化后经 actor 得到动作，explore=True 时加上 EXPLORATION_SIGMA 的高斯采样，再裁剪到动作边界并转为 NumPy。 |
| 方法 {#blk0480} | `TD3.soft_update` | 140-143 | 使用模块级 TAU 对 online 与 target 网络参数做 Polyak 原地平均。 |
| 方法 {#blk0481} | `TD3.update` | 145-186 | 先在 no_grad 下构造带 target smoothing 的 clipped double-Q target，对两个 critic 求 MSE 并分别更新；再每 POLICY_DELAY 次用 -Q1(s,actor(s)).mean() 更新 actor，并软更新 actor 与两个 critic 的 target。 |
| 函数 {#blk0482} | `main` | 189-221 | 固定三个随机种子，创建 Pendulum-v1、TD3 与 ReplayBuffer；逐 episode 交互并把 (s,a,r,s',terminated) 写入 buffer，buffer 达到 MINIMAL_SIZE 后每步更新一次，训练结束打印最后 10 回合平均回报并关闭环境。 |
| 入口 {#blk0483} | if __name__ == '__main__' | 224-225 | 脚本直接运行时调用 main()。 |

**`10_TD3/Pendulum/rl_utils.py`** · 53 行 · 提供 off-policy 经验回放池与回报曲线移动平均工具。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0484} | 导入与常量 | 1-4 | 导入 collections、random 与 numpy，无模块级常量。 |
| 类 {#blk0485} | `ReplayBuffer` | 7-35 | 以 deque(maxlen=capacity) 保存 (s, a, r, s', done)，支持追加、随机均匀采样和查询当前大小。 |
| 方法 {#blk0486} | `ReplayBuffer.__init__` | 15-16 | 创建容量为 capacity 的 deque，超容量时自动丢弃最旧 transition。 |
| 方法 {#blk0487} | `ReplayBuffer.add` | 18-19 | 把 (state, action, reward, next_state, done) 追加到 buffer。 |
| 方法 {#blk0488} | `ReplayBuffer.sample` | 21-32 | 用 random.sample 均匀抽取 batch_size 条经验，zip 拆成五元组并统一转成 float32 NumPy 数组返回。 |
| 方法 {#blk0489} | `ReplayBuffer.size` | 34-35 | 返回 len(self.buffer)，即当前经验条数。 |
| 函数 {#blk0490} | `moving_average` | 38-53 | 把序列转成 float32 数组；若长度不足 window_size 则原样返回，否则用累积和计算中段滑动平均，并对首尾用逐步增大的窗口补齐，最后拼接返回。 |

**`10_TD3/Pendulum/trainer.py`** · 85 行 · 实现 TD3 的 off-policy 训练循环与无噪声确定性策略评估。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0491} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0492} | `train_off_policy_agent` | 5-58 | 按 num_episodes 循环，用 seed+episode 重置环境；每步调用 agent.take_action(explore=True) 与环境交互，把 done 记为 terminated or truncated，但只把 terminated 写入 buffer，buffer 达到 minimal_size 后每步调用 agent.update；每 10 回合更新 tqdm 后置信息并返回每回合回报列表。 |
| 函数 {#blk0493} | `evaluate_policy` | 61-85 | 以 explore=False 的确定性策略运行 num_episodes 个回合，每个回合可用 seed+episode 固定初始状态，累加奖励后返回 float32 的回合回报数组。 |

**`10_TD3/Pendulum/visualization.py`** · 28 行 · 绘制 TD3 训练回报及移动平均曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0494} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot，并从 rl_utils 引入 moving_average。 |
| 函数 {#blk0495} | `plot_training_results` | 6-28 | 创建 1×2 子图，左图绘制逐回合原始回报，右图绘制 window_size=9 的移动平均，设置坐标轴与含环境名的标题后 tight_layout 并 show。 |

## 11_SAC：最大熵与自动温度落在哪里？ {#algo11 span=12}

SAC（双 critic 取最小值、重参数化高斯策略、自动熵温度调节）在 Pendulum-v1 上的模块化实现，包含奖励重塑、回放池、训练循环与曲线绘制。 本目录共 9 个 Python 文件、46 个代码块条目。

**`11_SAC/Pendulum/agents.py`** · 126 行 · 定义 SAC 智能体，封装随机策略、双 critic 及 target、自动温度参数与三类更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0496} | 导入与常量 | 1-5 | 导入 numpy、torch 与 torch.nn.functional，并从 networks 引入 Critic 与 GaussianPolicy；模块内无常量定义。 |
| 类 {#blk0497} | `SAC` | 8-126 | 维护 GaussianPolicy、两个 online critic、两个 target critic、actor/critic/alpha 三个优化器以及奖励重塑参数，实现随机动作采样、soft Bellman target 计算和单批更新。 |
| 方法 {#blk0498} | `SAC.__init__` | 15-45 | 构建 GaussianPolicy 与两套 Critic，把 online critic 参数复制给 target；建立三个 Adam 优化器，把可学习 log_alpha 初始化为 log(0.01)，记录 target_entropy、tau、gamma、reward_shift 与 reward_scale。 |
| 方法 {#blk0499} | `SAC.take_action` | 47-54 | 在 no_grad 下把单个状态转成 [1, state_dim] 张量送入随机策略采样一次动作（同时丢弃 log_prob），squeeze 后返回 NumPy 动作；策略本身带随机性，不需额外探索噪声。 |
| 方法 {#blk0500} | `SAC.calc_target` | 56-70 | 在 no_grad 下对 next_states 重参数化采样得到 a' 与 log pi(a'\|s')，用两个 target critic 的较小 Q 值减去 exp(log_alpha)·log pi 得到 soft Q，再返回 r + gamma·(1-done)·soft Q。 |
| 方法 {#blk0501} | `SAC.soft_update` | 72-78 | 对 online 与 target 网络参数逐对做 Polyak 平均 target ← (1-tau)·target + tau·online，使用 data 原地修改。 |
| 方法 {#blk0502} | `SAC.update` | 80-126 | 把 batch 转成张量并按 (r+shift)/scale 重塑奖励；先计算无梯度的 soft TD target，对两个 critic 分别做 MSE 更新；再用重参数化采样求 (alpha·log pi - min(Q1,Q2)).mean() 更新 actor，用 -(log_alpha·(log pi+target_entropy)) 更新温度参数，最后软更新两个 target critic。 |

**`11_SAC/Pendulum/config.py`** · 21 行 · 集中存放 SAC 实验的学习率、训练规模、折扣、回放池、奖励重塑与设备等超参数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0503} | 导入与常量 | 1-21 | 仅导入 torch，随后定义 actor/alpha 学习率 3e-4、critic 学习率 3e-3、100 个 episode、隐藏层 128、gamma 0.99、tau 0.005、buffer 10 万与 batch 64、REWARD_SHIFT 与 REWARD_SCALE 均为 8、SEED=0、ENV_NAME='Pendulum-v1' 以及自动选择 cuda/cpu 的 DEVICE。 |

**`11_SAC/Pendulum/environment.py`** · 17 行 · 创建 Pendulum 环境并返回状态/动作维度与连续动作边界。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0504} | 导入与常量 | 1-2 | 导入 gymnasium，无模块级常量。 |
| 函数 {#blk0505} | `make_environment` | 4-17 | 调用 gym.make 创建环境，若 action_space 没有 shape 属性则抛出 TypeError 提示需要 Box 连续动作空间；否则返回 env、state_dim、action_dim 以及 copy 后的动作上下界。 |

**`11_SAC/Pendulum/main.py`** · 42 行 · SAC 实验入口：设置种子、组装智能体与回放池、训练并绘制回报曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0506} | 导入与常量 | 1-11 | 导入 random、numpy、torch、config，以及本目录的 SAC、make_environment、ReplayBuffer、train_off_policy_agent 与 plot_training_results。 |
| 函数 {#blk0507} | `set_seed` | 14-17 | 用 random.seed、np.random.seed、torch.manual_seed 固定 Python、NumPy 与 PyTorch 的随机种子。 |
| 函数 {#blk0508} | `main` | 20-38 | 设置种子并创建环境，按 config 的参数实例化 SAC（target_entropy 传 -action_dim），建立 ReplayBuffer 后调用 train_off_policy_agent 训练，最后绘制回报曲线并 env.close()。 |
| 入口 {#blk0509} | if __name__ == '__main__' | 41-42 | 脚本直接运行时调用 main()。 |

**`11_SAC/Pendulum/networks.py`** · 69 行 · 定义 SAC 的重参数化高斯策略网络与 Q(s,a) Critic 网络。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0510} | 导入与常量 | 1-3 | 导入 torch、torch.nn.functional 以及 torch.distributions.Normal，无模块级常量。 |
| 类 {#blk0511} | `GaussianPolicy` | 6-46 | 随机高斯策略：共享隐藏层后输出均值与 log 标准差，经重参数化采样、tanh 压缩和线性缩放得到合法动作，并返回修正后的对数概率。 |
| 方法 {#blk0512} | `GaussianPolicy.__init__` | 13-22 | 构建 state_dim→hidden_dim 的共享层和 fc_mu、fc_log_std 两个输出头，并把 (high-low)/2 与 (high+low)/2 注册为 buffer 以便随设备迁移。 |
| 方法 {#blk0513} | `GaussianPolicy.forward` | 24-46 | 对状态求均值与截断到 [-20,2] 的 log_std 构造 Normal；用 rsample 得到 pre_tanh，经 tanh 与 action_bias/action_scale 映射为动作，同时减去 log(action_scale·(1-tanh²)+1e-6) 修正密度，按动作维度求和后返回 (actions, log_probs)。 |
| 类 {#blk0514} | `Critic` | 49-69 | Q(s,a) 网络，将状态与动作拼接后经两层隐藏层输出标量 Q 值；SAC 独立实例化两个以减少高估。 |
| 方法 {#blk0515} | `Critic.__init__` | 56-60 | 第一层输入维度为 state_dim+action_dim，第二层为 hidden_dim，输出层为 1 维。 |
| 方法 {#blk0516} | `Critic.forward` | 62-69 | 沿 dim=1 拼接 states 与 actions，经 ReLU(fc1)、ReLU(fc2) 后由 fc_out 返回 [batch_size, 1] 的 Q 值。 |

**`11_SAC/Pendulum/practice.py`** · 196 行 · 单文件版 SAC 教学脚本，把网络、智能体、训练与绘图写在同一文件中并直接执行。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0517} | 导入与常量 | 1-9 | 导入 random、gym、numpy、tqdm、torch、F、Normal、matplotlib 以及本目录的 rl_utils；模块级超参数与实验代码放在文件末尾。 |
| 类 {#blk0518} | `PolicyNetContinuous` | 11-30 | 连续动作随机策略：fc1 提取特征后由 fc_mu/fc_std 给出高斯参数，softplus 保证标准差为正，rsample 后经 tanh 乘 action_bound 输出动作及 log_prob。 |
| 方法 {#blk0519} | `PolicyNetContinuous.__init__` | 12-17 | 建立 state_dim→hidden_dim 的 fc1，以及输出 action_dim 维的 fc_mu 与 fc_std，并保存标量动作上界 action_bound。 |
| 方法 {#blk0520} | `PolicyNetContinuous.forward` | 19-30 | ReLU 后算 mu 与 softplus(std) 构造 Normal，rsample 得 normal_sample，tanh 后减去 log(1-tanh(action)²+1e-7) 修正 log_prob，再乘 action_bound 返回动作与逐维 log_prob。 |
| 类 {#blk0521} | `QValueNetContinuous` | 33-44 | Q(s,a) 网络，把状态动作拼接后经两层隐藏层输出标量价值。 |
| 方法 {#blk0522} | `QValueNetContinuous.__init__` | 34-38 | 输入层维度为 state_dim+action_dim，隐藏层为 hidden_dim，输出 1 维。 |
| 方法 {#blk0523} | `QValueNetContinuous.forward` | 40-44 | 沿 dim=1 拼接 x 与 a，经 ReLU(fc1)、ReLU(fc2) 后由 fc_out 输出 Q 值。 |
| 类 {#blk0524} | `SACContinuous` | 46-149 | 单文件版连续动作 SAC，维护策略网络、两个 Q 网络及 target、可学习 log_alpha 与三个优化器。 |
| 方法 {#blk0525} | `SACContinuous.__init__` | 48-80 | 创建 PolicyNetContinuous、两个 QValueNetContinuous 及对应 target 并同步初始参数，建立 actor/critic 优化器；把 log_alpha 设为可求导张量并记录 target_entropy、gamma、tau 与 device。 |
| 方法 {#blk0526} | `SACContinuous.take_action` | 82-85 | 把状态包成张量送入 actor 取第一个返回值（动作），用 item() 取出标量并以列表形式返回，适配 Pendulum 单维动作。 |
| 方法 {#blk0527} | `SACContinuous.calc_target` | 87-95 | 对 next_states 采样得到 next_actions 与 log_prob，令 entropy=-log_prob，用两个 target critic 的较小值加 alpha·entropy 构成 next_value，再算 r + gamma·(1-done)·next_value。 |
| 方法 {#blk0528} | `SACContinuous.soft_update` | 97-101 | 用 copy_ 把 target 参数更新为 target·(1-tau) + online·tau。 |
| 方法 {#blk0529} | `SACContinuous.update` | 103-149 | 从 transition_dict 取五元组转成张量并对奖励做 (r+8)/8 重塑；计算 td_target 后分别对两个 Q 网络做 MSE 更新，再以 -alpha·entropy - min(Q1,Q2) 更新 actor、以 (entropy-target_entropy)·alpha 更新 log_alpha，最后软更新两个 target critic。 |
| 入口 {#blk0530} | 模块级训练脚本 | 151-196 | 模块导入时即执行：创建 Pendulum-v0、读取维度与 action_bound、设置各随机种子与超参数，构建 ReplayBuffer 和 SACContinuous，调用 rl_utils.train_off_policy_agent 训练后分别绘制原始回报与 9 窗口移动平均并 show。 |

**`11_SAC/Pendulum/rl_utils.py`** · 41 行 · 提供 SAC 使用的 off-policy 回放池与回报移动平均函数。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0531} | 导入与常量 | 1-4 | 导入 collections、random 与 numpy，无模块级常量。 |
| 类 {#blk0532} | `ReplayBuffer` | 7-28 | 以 deque(maxlen=capacity) 存储 (s, a, r, s', terminal)，支持追加、随机采样与大小查询。 |
| 方法 {#blk0533} | `ReplayBuffer.__init__` | 10-11 | 创建容量为 capacity 的 deque，满后自动移除最旧经验。 |
| 方法 {#blk0534} | `ReplayBuffer.add` | 13-15 | 把 (state, action, reward, next_state, done) 追加进 buffer。 |
| 方法 {#blk0535} | `ReplayBuffer.sample` | 17-25 | 用 random.sample 均匀抽取 batch_size 条经验，zip 拆成五元组并各转成 NumPy 数组返回。 |
| 方法 {#blk0536} | `ReplayBuffer.size` | 27-28 | 返回 len(self.buffer)，即当前经验条数。 |
| 函数 {#blk0537} | `moving_average` | 31-41 | 转成 float32 数组；长度不足 window_size 时原样返回，否则用累积和计算中段滑动平均，并用逐步增大的窗口补齐首尾后拼接返回。 |

**`11_SAC/Pendulum/trainer.py`** · 39 行 · 实现 SAC 的 off-policy 训练循环，逐回合交互并在回放池足够后更新。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0538} | 导入与常量 | 1-2 | 导入 numpy 与 tqdm 进度条，无模块级常量。 |
| 函数 {#blk0539} | `train_off_policy_agent` | 5-39 | 按 num_episodes 循环，用 seed+episode（或默认）重置环境；每步调用 agent.take_action(state) 与环境交互，把 done 记为 terminated or truncated 但只将 terminated 写入 buffer，buffer 达到 minimal_size 后每步调用 agent.update；每 10 回合更新 tqdm 后置回报信息并返回回报列表。 |

**`11_SAC/Pendulum/visualization.py`** · 20 行 · 绘制 SAC 训练回报及移动平均曲线。

| 类型 | 代码块 | 行 | 逻辑 |
| --- | --- | --- | --- |
| 导入与常量 {#blk0540} | 导入与常量 | 1-3 | 导入 matplotlib.pyplot，并从 rl_utils 引入 moving_average。 |
| 函数 {#blk0541} | `plot_training_results` | 6-20 | 创建 1×2 子图，左图绘制逐回合原始回报，右图绘制 window_size=9 的移动平均，设置坐标轴与含环境名的标题后 tight_layout 并 show。 |

