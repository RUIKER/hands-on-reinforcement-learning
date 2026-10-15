# Reinforcement Learning 实验代码

这是一个强化学习学习项目，包含从表格型方法到深度强化学习和策略梯度方法的实现。代码主要使用 Python、Gymnasium、NumPy、PyTorch 和 Matplotlib。

## 项目目录

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| `Cliff Walking` | Cliff Walking 上的 Q-learning 和 SARSA | `train.py` |
| `DQN` | DQN 练习代码和经验回放 | `practice.py` |
| `Double DQN、Dueling DQN` | DQN、Double DQN、Dueling DQN 统一 pipeline | `main.py` |
| `策略梯度算法_Reinforce` | CartPole-v1 上的 REINFORCE | `main.py` |
| `Mountain Car` | 基于 RBF 特征的 Actor-Critic | `main.py` |
| `Mountain Car PyTorch` | 基于 PyTorch 的 Mountain Car Actor-Critic | `main.py` |

每个实验目录尽量独立管理自己的 `requirements.txt`，建议在对应目录中安装依赖和运行代码。

## 环境准备

推荐使用 Python 3.10 或更高版本，并为项目创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

进入具体实验目录后安装依赖，例如：

```bash
cd "Double DQN、Dueling DQN"
python -m pip install -r requirements.txt
python main.py
```

## 运行示例

### Cliff Walking

该项目只使用 Python 标准库：

```bash
cd "Cliff Walking"
python train.py
```

在 `train.py` 中切换 `q_learning.QTable` 和 `sarsa.SarsaTable` 可以对比 Q-learning 与 SARSA。

### Double DQN、Dueling DQN

```bash
cd "Double DQN、Dueling DQN"
python -m pip install -r requirements.txt
python main.py
```

在 `config.py` 中修改：

```python
ALGORITHM = 'DQN'
```

可选值为 `DQN`、`DoubleDQN` 和 `DuelingDQN`。

### REINFORCE

```bash
cd "策略梯度算法_Reinforce"
python -m pip install -r requirements.txt
python main.py
```

### Mountain Car

```bash
cd "Mountain Car"
python -m pip install -r requirements.txt
python main.py
```

### Mountain Car PyTorch

```bash
cd "Mountain Car PyTorch"
python -m pip install -r requirements.txt
python main.py
```

## 说明

- `经验总结.md` 文件记录每个实验的算法原理、实现细节、排错过程和后续改进方向。
- `.venv`、缓存文件和训练生成的输出默认不会提交到 Git。
- Gymnasium 项目使用新版 `reset()` 和 `step()` 接口，旧版 Gym 示例可能需要额外适配。
