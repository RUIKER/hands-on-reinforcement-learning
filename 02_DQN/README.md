# 02 DQN（最小练习实现）

本目录保留 DQN 的最小练习代码，用于理解三个核心组件：

- Q 网络：输入状态，输出每个离散动作的 Q 值。
- Replay buffer：存储 `(s, a, r, s', done)`，随机采样 batch。
- Target network：延迟同步，稳定 TD 自举目标。

## 文件

| 文件 | 内容 |
| --- | --- |
| `practice.py` | `ReplayBuffer`、`Qnet`、`DQN` 的最小实现 |
| `rl_utils.py` | 旧版训练循环和工具函数，历史参考 |
| `requirements.txt` | 依赖 |

`practice.py` 提供可导入的 DQN 和经验回放组件，不含训练入口。状态字段、目标网络初始化和梯度隔离已修正，并由 `tests/` 中的参数更新测试覆盖；`rl_utils.py` 保留旧接口供历史阅读。

## 完整实验

需要运行完整 DQN、Double DQN、Dueling DQN 对照实验，请使用：

```bash
cd "../03_Double DQN、Dueling DQN"
python main.py
```

## 算法定位

- 前驱：01 表格型 Q-learning。
- 后继：03 Double DQN / Dueling DQN。
- 详细原理、四问和阅读笔记见 `经验总结.md`。
