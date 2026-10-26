"""PPO / TRPO 对比实验编排：3 个 seeds、400k env steps、统一确定性评估。

复用 run_compare.py 里的聚合并图函数，只把训练驱动换成 on-policy 版本。
输出到 experiments/results_onpolicy/，并可选地把现有 SAC 的 40k 结果叠加成图 2。

用法：
    python experiments/run_onpolicy.py --seeds 0 1 2 --episodes 2000 --gamma 0.99
    python experiments/run_onpolicy.py --seeds 0 1 2 --plot-only
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
RUN_ONE = EXP_DIR / "run_one_onpolicy.py"

# 复用 off-policy 版本的聚合/画图函数。
from run_compare import (  # noqa: E402
    COLORS,
    LABELS,
    R_BEST,
    R_RANDOM,
    compute_steps_to_threshold,
    median_iqr,
    normalized,
    plot_eval_curves,
    plot_final_eval,
    plot_training_curves,
    write_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PPO/TRPO (on-policy).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algos", nargs="+", default=["ppo", "trpo"],
                        choices=["ppo", "trpo"])
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Pendulum 每回合 200 步；2000 回合=400k 步")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="文档要求 PPO 用 0.99；TRPO 也统一到 0.99")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--periodic-eval-episodes", type=int, default=5)
    parser.add_argument("--outdir", default=str(EXP_DIR / "results_onpolicy"))
    parser.add_argument("--sac-raw-dir", default=str(EXP_DIR / "results" / "raw"),
                        help="已有 SAC 的逐回合 JSON 目录，用于图 2 叠加")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def run_training(args, raw_dir, model_dir):
    for algo in args.algos:
        for seed in args.seeds:
            out_path = raw_dir / f"{algo}_seed{seed}.json"
            if out_path.exists() and not args.force:
                print(f"[skip] {out_path.name} 已存在（--force 可重跑）", flush=True)
                continue
            cmd = [
                sys.executable, str(RUN_ONE),
                "--algo", algo,
                "--seed", str(seed),
                "--episodes", str(args.episodes),
                "--gamma", str(args.gamma),
                "--eval-episodes", str(args.eval_episodes),
                "--eval-seed", str(args.eval_seed),
                "--eval-every", str(args.eval_every),
                "--periodic-eval-episodes", str(args.periodic_eval_episodes),
                "--out", str(out_path),
                "--model-out", str(model_dir / f"{algo}_seed{seed}.pt"),
            ]
            print(f"\n=== {algo.upper()} seed={seed} ===", flush=True)
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            completed = subprocess.run(cmd, cwd=str(REPO), env=env)
            if completed.returncode != 0:
                raise SystemExit(
                    f"{algo} seed={seed} 训练失败，退出码 {completed.returncode}"
                )


def load_runs(args, raw_dir):
    runs = {}
    missing = []
    for algo in args.algos:
        for seed in args.seeds:
            path = raw_dir / f"{algo}_seed{seed}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            runs[(algo, seed)] = json.loads(path.read_text(encoding="utf-8"))
    if missing:
        raise SystemExit("缺少结果文件：" + ", ".join(missing))
    return runs


def load_sac_runs(args):
    sac_runs = {}
    for seed in args.seeds:
        path = Path(args.sac_raw_dir) / f"sac_seed{seed}.json"
        if path.exists():
            sac_runs[("sac", seed)] = json.loads(
                path.read_text(encoding="utf-8")
            )
    return sac_runs


def write_summary(runs, args, summary, steps_to_threshold, out_dir):
    ranked = sorted(summary, key=lambda row: row["eval_mean_median"], reverse=True)
    lines = [
        "# PPO / TRPO 对比实验（Pendulum-v1）",
        "",
        f"- 日期：2026-10-25",
        f"- seeds：{', '.join(str(seed) for seed in args.seeds)}"
        f"（n={len(args.seeds)}）",
        f"- 训练：每个算法 {args.episodes} 回合 × 200 步 = "
        f"{args.episodes * 200} env steps",
        f"- gamma：{args.gamma}（PPO 按文档从 0.9 改为 0.99；"
        "TRPO 也统一到 0.99，避免隐藏变量）",
        f"- 最终评估：{args.eval_episodes} 回合确定性策略（mu），"
        f"`EVAL_SEED={args.eval_seed}`",
        f"- periodic eval：每 {args.eval_every} 步评估 "
        f"{args.periodic_eval_episodes} 回合",
        "",
        "## 与原始代码的差异",
        "",
        "- 原 `trainer.py` 的 `env.reset()` 没有 seed；驱动层改为 "
        "`reset(seed=seed+episode)`，保证可复现。",
        "- 原 `main.py` 没有确定性评估；驱动层补了 `mu`（不加采样噪声）评估。",
        "- 原 PPO/TRPO 的 gamma 都是 0.9；按文档统一为 0.99。",
        "- `REWARD_SHIFT/REWARD_SCALE` 保持 8/8，只作用于 critic 目标；"
        "训练回报仍是环境原始回报。",
        "",
        "## 结果（中位数 [Q1, Q3]）",
        "",
        "| 算法 | 最终评估回报 | IQR | train last-10 中位数 | "
        "steps-to-90% 中位数 | wall-clock 中位数 (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        steps = (
            f"{row['steps_to_90pct_median']:.0f}"
            if row["steps_to_90pct_median"] is not None else "NA"
        )
        lines.append(
            f"| {LABELS[row['algo']]} | {row['eval_mean_median']:.1f} "
            f"[{row['eval_mean_q1']:.1f}, {row['eval_mean_q3']:.1f}] | "
            f"{row['eval_mean_iqr']:.1f} | {row['train_last10_median']:.1f} | "
            f"{steps} | {row['wall_clock_median']:.0f} |"
        )
    lines += [
        "",
        "## 图 2：PPO vs SAC（样本效率）",
        "",
        "- `fig2_ppo_vs_sac_eval.png`：把 SAC 的 40k 结果和 PPO/TRPO 的 400k "
        "结果画在同一张 env steps 轴上。SAC 曲线在 40k 处结束（已收敛），"
        "PPO/TRPO 跑到 400k——这正是文档要展示的样本效率差一个数量级。",
        "- 但实测 PPO/TRPO 在 400k 仍未达到 SAC 的水平；文档预期的"
        "“SAC 20k ≈ PPO 400k”没有复现。原因见 `gamma_comparison.md`："
        "PPO 的其余超参是为 gamma=0.9 调的，单独改成 0.99 会退化。",
        "- 两条曲线的训练预算不同，不能用来比较最终性能；只用于比较"
        "“达到同一水平需要多少 env steps”。",
        "",
        "## 观察（只描述，不做显著性结论）",
        "",
        "- 最终评估中位数排序："
        + " > ".join(
            f"{LABELS[row['algo']]} ({row['eval_mean_median']:.1f})"
            for row in ranked
        ),
        "- n=3，不要做显著性检验；PPO/TRPO 的 run-to-run 方差本身就很大。",
        "",
        "## 局限",
        "",
        "- PPO/TRPO 是 on-policy，与 DDPG/TD3/SAC 的 off-policy 图不应混在一张图里；"
        "文档要求分图 1/图 2。",
        "- Pendulum 是教学环境，不是公开 benchmark。",
        "- wall-clock 受 GPU 状态影响，只作参考。",
        "",
        "## 复现",
        "",
        "```powershell",
        "cd \"C:\\Users\\PC\\Desktop\\具身智能实习\\hands-on-reinforcement-learning\"",
        ".venv\\Scripts\\python.exe experiments\\run_onpolicy.py --seeds 0 1 2",
        "```",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.outdir)
    raw_dir = out_dir / "raw"
    model_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        run_training(args, raw_dir, model_dir)

    runs = load_runs(args, raw_dir)

    best_final = max(run["metrics"]["eval_mean"] for run in runs.values())
    threshold_norm = 0.9 * normalized(best_final)
    steps_to_threshold = compute_steps_to_threshold(runs, args, threshold_norm)

    rows, summary = write_results(runs, args, steps_to_threshold, out_dir)
    plot_training_curves(
        runs, args, out_dir / "comparison_training_curves.png"
    )
    plot_eval_curves(runs, args, out_dir / "comparison_eval_curves.png")
    plot_final_eval(runs, args, out_dir / "comparison_final_eval.png")
    write_summary(runs, args, summary, steps_to_threshold, out_dir)

    # 图 2：把已有 SAC（40k）叠到 PPO/TRPO（400k）上。
    sac_runs = load_sac_runs(args)
    if len(sac_runs) == len(args.seeds):
        combined = dict(sac_runs)
        combined.update(runs)
        combined_args = argparse.Namespace(
            algos=["sac"] + args.algos,
            seeds=args.seeds,
            eval_every=1,
            periodic_eval_episodes=args.periodic_eval_episodes,
        )
        plot_eval_curves(
            combined, combined_args, out_dir / "fig2_ppo_vs_sac_eval.png"
        )
        print(
            f"[fig2] 叠加了 SAC seeds: "
            f"{sorted(seed for _, seed in sac_runs)}",
            flush=True,
        )
    else:
        print(
            f"[fig2] SAC 结果不完整（{len(sac_runs)}/{len(args.seeds)}），"
            "跳过叠加图",
            flush=True,
        )

    protocol = {
        "seeds": args.seeds,
        "algos": args.algos,
        "episodes": args.episodes,
        "env_steps": args.episodes * 200,
        "gamma": args.gamma,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "eval_every": args.eval_every,
        "periodic_eval_episodes": args.periodic_eval_episodes,
        "threshold_norm": threshold_norm,
        "best_final_eval": best_final,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== summary ===", flush=True)
    for row in summary:
        print(row, flush=True)
    print(f"\n输出目录：{out_dir}", flush=True)


if __name__ == "__main__":
    main()
