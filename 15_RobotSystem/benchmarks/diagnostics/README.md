# Panda 实验诊断记录

这里保留从 Windows 实验目录迁移的专用诊断脚本及已完成结果。场景与公共入口见 [Panda 项目](../../README.md)。脚本可通过显式参数或 PANDA_LEGACY_RUN 指定实验原始输出目录。逐步训练日志和原始 PyTorch checkpoint 需由相应运行提供。

| 目录 | 内容 |
| --- | --- |
| warmup_multiseed | 多种子 warm-up 对照驱动及分析 |
| seed42_comparison | 50k SAC/HER 运行驱动与结果比较 |
| precision | 独立小误差目标评估及停滞分析 |
| warmup_seed43 | 单种子 warm-up 对照与比较结果 |
| factorial_recovery | 奖励塑形、动作平滑、重标记及失败恢复诊断 |
| sac_boundaries | 动作、截断、自举和存档边界检查 |
| her_relabel | 对已有轨迹执行 HER 重标记检查 |

运行驱动与分析脚本分开。阅读已完成结果无需重新训练。两个可共享的短验收入口是 [HER 单回合 smoke](../../panda_her_smoke.py) 与 [WSL 闭环验证](../../run_ros2_checks.sh)。
