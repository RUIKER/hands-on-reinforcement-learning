# Mountain Car Actor-Critic with PyTorch

这是原始 Mountain Car 项目的 PyTorch 版本。原版本使用 NumPy 手动更新线性 Actor-Critic；本版本使用 PyTorch 的 MLP、自动求导和 Adam 优化器。

## 运行

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

训练结果会保存到 `output/`：

- `episode_stats.csv`: 每个 episode 的奖励和步数
- `training_curves.png`: 训练曲线
- `actor_critic.pt`: Actor 和 Critic 的模型参数

## 主要实现

- `networks.py`: Actor 和 Critic 神经网络
- `trainer.py`: episode 批量更新的 n 步 TD Actor-Critic
- `environment.py`: Gymnasium 环境适配
- `main.py`: 实验入口和模型保存

## 实现取舍与限制

- **时间截断**：`terminated` 停止 bootstrap，`truncated` 只结束回合采样。
- **更新节奏**：每回合收集完整轨迹，用 n 步 TD 目标批量更新，便于传播较晚出现的终点奖励。
- **奖励缩放**：缩放仅用于训练目标；CSV 与曲线保留原始奖励，避免混淆评估单位。
- **动作裁剪**：采样高斯动作后裁剪至 [-1, 1]，保留 NumPy 版本的探索方式。训练对裁剪动作计算高斯 log probability 是有偏近似，并非严格的变换分布策略梯度。
- **诊断依据**：训练回报、回合长度和成功到达终点应一起检查。仅减少动作惩罚也可能提高回报，不能据此认定已学会爬坡。

历史版本的探索与逐步更新问题见 `经验总结.md`；这些观察限定于当时配置，不能推广为 tanh 策略或 Adam 优化器的一般局限。
