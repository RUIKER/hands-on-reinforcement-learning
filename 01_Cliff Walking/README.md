# Cliff Walking：Q-learning 与 SARSA

在手写网格环境中比较两种 TD 控制方法。Q-learning 对下一状态取最大 Q 值；SARSA 使用实际执行的下一动作估计目标。

## 运行

仅依赖 Python 标准库。从仓库根目录执行：

```bash
python "01_Cliff Walking/train.py" --algorithm q-learning --episodes 3000 --seed 0
python "01_Cliff Walking/train.py" --algorithm sarsa --episodes 3000 --seed 0
```

`main.py` 与 `train.py` 接受相同参数。默认算法为 Q-learning；其余超参见 `config.py`。训练返回逐回合回报，并按配置间隔打印，未将探索策略下的训练回报作为确定性评估。

## 文件与接口

| 文件 | 职责 |
| --- | --- |
| `environment.py` | 状态 `[行, 列]`、动作转移与奖励 |
| `q_learning.py` / `sarsa.py` | Q 表、epsilon-greedy 动作选择和 TD 更新 |
| `train.py` | 配置装配、回合采样、参数更新、命令行入口 |
| `config.py` | 地图尺寸、学习率、折扣因子和探索率 |

动作编号以 `Env.step` 为准：0/1 沿列向右/左，2/3 沿行向起点行/远离起点行。到达终点奖励 −1，掉入悬崖奖励 −100，两者均结束回合。**标准 Cliff Walking 通常在掉落后返回起点并继续回合；本环境采用终止式变体，结果不宜直接比较。**
