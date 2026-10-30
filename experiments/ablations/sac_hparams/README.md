# SAC 超参单因素扫描（HalfCheetah-v5）

用于周日“系统学超参调整”：每次只改一个超参，看样本效率的变化。**这是学习素材，不是调参追高分。**

## 协议

| 项 | 设置 |
| --- | --- |
| 环境 / 算法 | HalfCheetah-v5，SAC（与 `seed_experiments/halfcheetah_sac` 相同实现） |
| 预算 | 100k steps × 3 seeds（0/1/2），每 20k 步评估 5 回合 |
| 基线 | **不重跑**，直接取 1M 基线 run 在 20k 评估网格上的前 100k 步（同代码、同 seed） |
| 对比指标 | 60k / 80k / 100k 三个评估点的均值（降低单点噪声），及 100k 单点 |

| 配置 | 改动 | 想检验的问题 |
| --- | --- | --- |
| `lr1e-4` / `lr1e-3` | actor、critic 学习率 3e-4 → 1e-4 / 1e-3 | 学习率对前期速度与稳定性 |
| `batch64` / `batch1024` | batch 256 → 64 / 1024 | 梯度噪声 vs 单步成本 |
| `gamma0.95` | gamma 0.99 → 0.95 | 视野长度对跑步任务的影响 |
| `alpha1.0` | 初始 alpha 0.01 → 1.0 | 本仓库 SAC 写死 0.01，SB3 默认 1.0 |

## 运行

```bash
cd experiments/ablations/sac_hparams
bash run_sweep.sh            # JOBS=6，已有结果自动跳过
../../../.venv/Scripts/python.exe aggregate_sweep.py
```

输出：`results/raw/*.json`、`results/sweep_table.md`、`results/sweep_results.csv`、
`results/sweep_curves.png`。

## 注意

- 100k 步只反映**样本效率**，不代表 1M 终点；不要把前 100k 的排序外推到最终性能。
- 3 seeds 噪声大；差异小于跨 seed std 的配置不要下结论。
- 单因素扫描看不到超参之间的交互（例如 lr 与 batch、reward scale 与 alpha）。
