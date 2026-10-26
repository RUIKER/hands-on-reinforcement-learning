# DDPG / TD3 / SAC 对比实验

- 日期：2026-10-25
- 环境：`Pendulum-v1`
- seeds：0, 1, 2（n=3）
- 训练：每个算法 200 回合 × 200 步 = 40000 env steps
- 最终评估：20 回合确定性策略，`EVAL_SEED=10000`（所有算法共用同一组初始状态）
- periodic eval：每 5000 步评估 5 回合（确定性）

## 统一口径说明

- 三个算法都训练到相同 env steps 数，横轴统一为 env steps；
- 训练回报含各自探索噪声，**只用于画训练曲线，不用于算法间比较**；
- 性能数字唯一来源是训练结束后的确定性评估；
- 每个算法保留自己的参考超参数（见下表），没有为了“公平”强行统一网络宽度和学习率；
- SAC 的 `REWARD_SHIFT/REWARD_SCALE` 只作用于 critic 的 Bellman target，`return_list` 仍是环境原始回报。

## 各算法超参数（config.py 快照）

| 算法 | GAMMA | ACTOR_LR | CRITIC_LR | HIDDEN_DIM | BATCH_SIZE | BUFFER_SIZE | 其他 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DDPG | 0.98 | 0.0003 | 0.003 | 64 | 64 | 10000 | NOISE_SIGMA=0.01 |
| TD3 | 0.99 | 0.001 | 0.001 | 128 | 100 | 100000 | EXPLORATION_SIGMA=0.1, POLICY_DELAY=2 |
| SAC | 0.99 | 0.0003 | 0.003 | 128 | 64 | 100000 | REWARD_SHIFT=8.0, REWARD_SCALE=8.0 |

## 结果（3 seeds，中位数 [Q1, Q3]）

| 算法 | 最终评估回报 | IQR | train last-10 中位数 | steps-to-90% 中位数 | wall-clock 中位数 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| DDPG | -150.9 [-347.8, -139.0] | 208.8 | -187.5 | 10000 | 160 |
| TD3 | -122.8 [-123.4, -122.7] | 0.7 | -164.7 | 10000 | 212 |
| SAC | -120.6 [-121.2, -120.4] | 0.9 | -164.8 | 5000 | 487 |

## 图

- `comparison_training_curves.png`：图 1（细线=单 seed，粗线=中位数）
- `comparison_eval_curves.png`：训练过程中的确定性评估曲线
- `comparison_final_eval.png`：最终评估的箱线图 + 单 seed 散点

## 观察（只描述，不做显著性结论）

- 最终评估中位数排序：SAC (-120.6) > TD3 (-122.8) > DDPG (-150.9)
- 达到 90% 阈值（以 9 次运行的最佳最终性能归一化）最快的是 SAC，中位数约 5000 env steps。
- 方差看 IQR 和图中的细线；3 个 seed 下不要对“谁显著更好”下结论。

## 局限

- n=3：做不了 IQM / bootstrap 置信区间，也不应做显著性检验；
- 每个算法用的是仓库里的参考超参数，不是同一套超参数下的公平扫描；
- Pendulum 是教学环境，不是公认 benchmark；这张图用于验证评估协议和实现，不能据此宣称算法优劣；
- wall-clock 包含 periodic eval 的开销，且受 GPU 状态影响，只作参考。

## 复现

```powershell
cd "C:\Users\PC\Desktop\具身智能实习\hands-on-reinforcement-learning"
.venv\Scripts\python.exe experiments\run_compare.py --seeds 0 1 2
```

原始逐回合数据在 `results/raw/`，模型在 `results/models/`。
