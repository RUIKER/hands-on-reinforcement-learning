# HalfCheetah SAC（v5）

MuJoCo HalfCheetah 上的 SAC，3 seeds × 1M steps。复用 `11_SAC/Pendulum/agents.py`
的 SAC 类，本地 `networks.py` 把 actor/critic 换成 2×256 MLP（MuJoCo 常用配置）。

## 协议

| 项 | 设置 |
| --- | --- |
| 环境 | `HalfCheetah-v5`（obs 17，action 6，动作范围 [-1,1]） |
| 算法 | SAC，hidden 256，batch 256，buffer 1e6，gamma .99，tau .005 |
| 学习率 | actor / critic / alpha = 3e-4 |
| 训练 | 1M env steps，每步更新一次；前 10k 步只填 buffer |
| 评估 | 每 20k 步 5 回合；最终 20 回合；确定性 `tanh(mu)` |
| seeds | 0 / 1 / 2（agent 可扩到 5） |
| 设备 | CUDA（RTX 5060 Ti） |

## 运行

```bash
cd experiments/seed_experiments/halfcheetah_sac
bash run_all.sh
../../../.venv/Scripts/python.exe aggregate.py
```

输出：`results/raw/sac_s*.json`、`results/learning_curves.png`、
`results/final_eval.png`、`results/summary.csv`、`results/SUMMARY.md`。

## 超参覆盖（供 `ablations/sac_hparams` 使用）

`run_one.py` 支持 `--hidden-dim --actor-lr --critic-lr --alpha-lr --tau --gamma
--batch-size --reward-scale --init-alpha`，默认值来自 `settings.py`；实际使用值写入 JSON 的
`config`。基线 1M run 使用默认值，**初始 alpha 为 0.01**（`11_SAC/Pendulum/agents.py` 写死）。

## 环境说明

- `.venv` 已安装 `gymnasium[mujoco]`（mujoco 3.17）。
- 目录已经改成英文路径后，MuJoCo 可以直接读取 assets；不再需要 ASCII 资产补丁。
- `HalfCheetah-v4` 已 deprecated，统一用 v5。

## 结果（3 seeds × 1M steps，已完成）

| seed | final mean（20 回合） | final median | final best | 训练耗时 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 11083.6 | 11063.9 | 11256.9 | 14047s |
| 1 | 11437.9 | 11451.6 | 11572.6 | 14042s |
| 2 | 10230.8 | 10230.8 | 10562.2 | 14068s |

均值 **10917.4**，中位数 11083.6。曲线在 32–52 万步达到最终值的 80%，
之后缓慢上升/平台，1M 时没有明显崩塌。产物见 `results/SUMMARY.md`、
`results/summary.csv`、`results/learning_curves.png`、`results/final_eval.png`。

注意：final mean 来自 20 个固定评估回合（`EVAL_SEED=10000`），确定性动作
`tanh(mu)`；和 SB3 zoo 的公开数字比较前，先对齐环境版本、评估回合数和动作噪声。

## 你需要自己判断的三件事

Agent 只负责跑数和出图，下面三件事不要直接抄结论：

1. HalfCheetah 的奖励是稠密的，为什么这里不能用 HER？如果硬套 HER，目标是什么？
2. SAC 在 Pendulum 和 HalfCheetah 上的超参（hidden、reward scale、target entropy）
   为什么不同？哪些是环境维度决定的，哪些是调出来的？
3. 学习曲线的波动、最终 return 和 SB3 zoo 的公开值差多少，差异应该归因于
   算法、实现、评估协议还是训练预算？
