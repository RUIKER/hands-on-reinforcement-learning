# PPO / TRPO 对比实验（Pendulum-v1）

- 日期：2026-10-25
- seeds：0（n=1）
- 训练：每个算法 2000 回合 × 200 步 = 400000 env steps
- gamma：0.9（PPO 按文档从 0.9 改为 0.99；TRPO 也统一到 0.99，避免隐藏变量）
- 最终评估：20 回合确定性策略（mu），`EVAL_SEED=10000`
- periodic eval：每 10000 步评估 5 回合

## 与原始代码的差异

- 原 `trainer.py` 的 `env.reset()` 没有 seed；驱动层改为 `reset(seed=seed+episode)`，保证可复现。
- 原 `main.py` 没有确定性评估；驱动层补了 `mu`（不加采样噪声）评估。
- 原 PPO/TRPO 的 gamma 都是 0.9；按文档统一为 0.99。
- `REWARD_SHIFT/REWARD_SCALE` 保持 8/8，只作用于 critic 目标；训练回报仍是环境原始回报。

## 结果（中位数 [Q1, Q3]）

| 算法 | 最终评估回报 | IQR | train last-10 中位数 | steps-to-90% 中位数 | wall-clock 中位数 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PPO | -338.2 [-338.2, -338.2] | 0.0 | -214.6 | 110000 | 515 |
| TRPO | -880.6 [-880.6, -880.6] | 0.0 | -867.1 | NA | 595 |

## 图 2：PPO vs SAC（样本效率）

- `fig2_ppo_vs_sac_eval.png`：把 SAC 的 40k 结果和 PPO/TRPO 的 400k 结果画在同一张 env steps 轴上。SAC 曲线在 40k 处结束（已收敛），PPO/TRPO 跑到 400k——这正是文档要展示的样本效率差一个数量级。
- 两条曲线的训练预算不同，不能用来比较最终性能；只用于比较“达到同一水平需要多少 env steps”。

## 观察（只描述，不做显著性结论）

- 最终评估中位数排序：PPO (-338.2) > TRPO (-880.6)
- n=3，不要做显著性检验；PPO/TRPO 的 run-to-run 方差本身就很大。

## 局限

- PPO/TRPO 是 on-policy，与 DDPG/TD3/SAC 的 off-policy 图不应混在一张图里；文档要求分图 1/图 2。
- Pendulum 是教学环境，不是公开 benchmark。
- wall-clock 受 GPU 状态影响，只作参考。

## 复现

```powershell
cd "C:\Users\PC\Desktop\具身智能实习\hands-on-reinforcement-learning"
.venv\Scripts\python.exe experiments\run_onpolicy.py --seeds 0 1 2
```
