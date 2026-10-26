# experiments：强化学习对比实验（DDPG / TD3 / SAC / PPO / TRPO）

对应 `plan/10_对比实验设计.docx` 的图 1 / 图 2 规格，并附 gamma 诊断。
**不修改任何算法目录下的代码**，只在运行时按各自 `main.py` 的方式组装 agent，
并统一训练/评估协议。

完整的实验结论、结果表和训练性能优化建议见 **[`经验总结.md`](经验总结.md)**。

## 协议

| 项 | 设置 |
|---|---|
| 环境 | `Pendulum-v1` |
| seeds | `0 / 1 / 2`（训练、环境 reset、PyTorch 初始化） |
| 训练回合 | 每个算法 200 回合 × 200 步 = 40k env steps |
| 训练曲线 | 滑动平均窗口 10；细线 = 单 seed，粗线 = 3 seed 中位数 |
| periodic eval | 每 5000 steps，5 回合确定性评估（用于 eval 曲线和 steps-to-90%） |
| 最终评估 | 20 回合确定性评估，`EVAL_SEED=10000`，所有算法共用 |
| 性能口径 | 只用确定性 eval return；训练回报含探索噪声，不用于算法间比较 |

各算法保留自己的参考超参数（DDPG：gamma=0.98、hidden=64；TD3/SAC：gamma=0.99、
hidden=128 等），没有为了“公平”强行统一网络宽度和学习率。SAC 的
`REWARD_SHIFT/REWARD_SCALE` 只作用于 critic 的 Bellman target，训练回报仍是原始环境回报。

## 用法

```powershell
cd "C:\Users\PC\Desktop\具身智能实习\hands-on-reinforcement-learning"
.venv\Scripts\python.exe experiments\run_compare.py --seeds 0 1 2
```

只重画图（不重训）：

```powershell
.venv\Scripts\python.exe experiments\run_compare.py --seeds 0 1 2 --plot-only
```

重跑已有结果：

```powershell
.venv\Scripts\python.exe experiments\run_compare.py --seeds 0 1 2 --force
```

## 输出（`experiments/results/`）

| 文件 | 内容 |
|---|---|
| `raw/{algo}_seed{seed}.json` | 逐回合 steps/returns、periodic eval、最终 eval、超参快照 |
| `models/{algo}_seed{seed}.pt` | actor state_dict（供后续鲁棒性实验复用） |
| `comparison_training_curves.png/.pdf` | 图 1：训练曲线（3 seeds + 中位数） |
| `comparison_eval_curves.png/.pdf` | 训练中的确定性评估曲线 |
| `comparison_final_eval.png/.pdf` | 最终评估箱线图 + 单 seed 散点 |
| `results.csv` | 每次运行的明细（eval mean/median/std/best、steps-to-90%、wall-clock） |
| `results_summary.csv` | 按算法聚合的中位数 / IQR |
| `summary.md` | 协议、超参表、结果、观察与局限 |
| `protocol.json` | 本次实验的完整参数快照 |

## PPO / TRPO（on-policy，图 2）

PPO/TRPO 的原始 `trainer.py` 没有 seed、没有评估函数；驱动层在**不改原代码**的
前提下补了 `reset(seed=seed+episode)` 和确定性 `mu` 评估。按文档要求，PPO 的
`GAMMA` 从 0.9 改为 0.99；TRPO 也统一到 0.99，避免隐藏变量。

```powershell
# PPO + TRPO，各 3 seeds × 2000 回合（400k 步）
.venv\Scripts\python.exe experiments\run_onpolicy.py --seeds 0 1 2

# 只重画图（复用已有 JSON）
.venv\Scripts\python.exe experiments\run_onpolicy.py --seeds 0 1 2 --plot-only
```

输出在 `experiments/results_onpolicy/`：

| 文件 | 内容 |
|---|---|
| `raw/{ppo,trpo}_seed{seed}.json` | 逐回合数据、periodic eval、最终 eval |
| `comparison_*.png/.pdf` | PPO/TRPO 之间的训练/评估/最终评估图 |
| `fig2_ppo_vs_sac_eval.png/.pdf` | 图 2：SAC（40k，已收敛）vs PPO/TRPO（400k） |
| `fig2b_ppo_gamma_compare.png/.pdf` | PPO gamma=0.99（3 seeds）vs gamma=0.9（1 seed 诊断） |
| `gamma_comparison.md` | gamma 对照的详细结论（为什么 0.99 反而退化） |
| `results.csv` / `results_summary.csv` / `summary.md` | 结果表与说明 |

图 2 的 SAC 曲线只到 40k（已经在 5k 步左右收敛），PPO/TRPO 跑到 400k，
这正是文档要展示的“样本效率差一个数量级”；两条曲线的训练预算不同，
只能比较“达到同一水平需要多少 env steps”，不能比较最终性能。

**gamma 对照的重要发现**：PPO gamma=0.9、seed 0 得到 -338.2，和仓库原始记录
-338.146 几乎一致，说明驱动层正确；同一 seed 下 gamma=0.99 只有 -569.7。
其余超参（`CRITIC_LR=5e-3`、`REWARD_SCALE=8`、`EPOCHS=10`）是为 0.9 调的，
单独改 gamma 会让 value target 量级失配。详见 `results_onpolicy/gamma_comparison.md`。
诊断实验的输出在 `experiments/results_onpolicy_gamma09/`，不属于 3-seed 正式表。

## 注意

- 3 个 seed 只够画中位数和 IQR，**不要做显著性检验**，也不要据此宣称算法优劣。
- Pendulum 是教学环境，不是公认 benchmark；这套脚本的价值是跑通评估协议，为后续
  HalfCheetah / Hopper 或 Meta-World 实验复用。
- 单次运行通过 `run_one.py` 执行，可单独调试：

```powershell
.venv\Scripts\python.exe experiments\run_one.py --algo td3 --seed 0 --episodes 10 ^
    --eval-episodes 3 --eval-every 500 --out experiments\results\raw\smoke.json
```
