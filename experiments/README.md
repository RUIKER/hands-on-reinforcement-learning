# 实验索引与复现

算法实现在各章节目录；本目录负责多种子训练、固定协议评估与结果聚合。表格中的数值来自仓库保留的历史产物，复现时应记录实际依赖版本与设备。

## 主实验

| 目录 | 问题与协议 | 证据入口 |
| --- | --- | --- |
| [five_algorithms](seed_experiments/five_algorithms/README.md) | Pendulum-v1，DDPG / TD3 / SAC / PPO / TRPO，10 seeds；off-policy 40k、on-policy 400k 步 | 固定 20 个起点、确定性评估、逐 seed JSON、配置与版本快照 |
| [her_main](seed_experiments/her_main/README.md) | GridWorld，DDPG 与 DDPG+HER，10 seeds，2000 回合 | 固定目标成功率、失败 seed、逐回合记录 |
| [halfcheetah_sac](seed_experiments/halfcheetah_sac/README.md) | HalfCheetah-v5，SAC，3 seeds × 1M 步 | 最终回报跨 seed 均值 10917.4；每 seed 最终评估 20 回合 |

## 消融与诊断

| 目录 | 控制变量 | 观察范围 |
| --- | --- | --- |
| [her_ratio](ablations/her_ratio/README.md) | HER 比例 0 / 0.2 / 0.5 / 0.8 / 1.0，10 seeds | 此配置下 ratio=1.0 的 10 个 seed 均未达标；0.2–0.8 的差异不足以排序 |
| [her_strategy](ablations/her_strategy/README.md) | future / final / episode，10 seeds | 检查最终成功率，同时识别重标记产生的平凡目标 |
| [sac_hparams](ablations/sac_hparams/README.md) | 学习率、batch、gamma、初始 alpha 等单因素与灵敏度扫描 | 100k 步短预算、3 seeds；区分早期学习速度与最终性能 |

## 执行约定

1. 在仓库根目录创建并激活虚拟环境，安装 `requirements.txt`。MuJoCo 实验额外安装 `python -m pip install "gymnasium[mujoco]>=1.3"`。
2. 按各目录 README 运行训练与聚合。Python 驱动使用当前解释器；Bash 驱动可通过 `PY` 指定解释器，通过 `JOBS` 限制并发。
3. 新协议使用新的结果目录。部分旧 Bash 驱动仅凭 JSON 文件存在决定跳过，复用前需核对 seed、预算和配置；训练成功退出与结果文件可用是两个检查。
4. 保留失败运行的日志和逐种子结果。最终指标、首次达标、训练回报分别报告；不要把不同预算或外部基线的数字直接排成优劣。

完整流程说明与编码约定见 [开发说明](../docs/development.md)。根 README 提供短训练命令；短训练验证管线可运行，不代表达到论文或历史实验性能。
