"""DDPG / TD3 / SAC 三算法对比实验：训练 3 个 fixed seeds，聚合并画图。

对应文档 plan/10_对比实验设计.docx 的「图 1」规格：
- 横轴 env steps（0–40k），不是 episode；
- 纵轴窗口 10 的滑动平均训练回报；
- 每个算法画 3 个 seed 的原始曲线（细线，alpha=0.3）+ 中位数（粗线）；
- 训练结束用 EVAL_SEED=10000 做 20 回合确定性评估；
- 额外每 5000 步做 5 回合 periodic eval，用于 steps-to-90% 和 eval 曲线。

用法：
    python experiments/run_compare.py --seeds 0 1 2
    python experiments/run_compare.py --seeds 0 1 2 --plot-only   # 只重画图
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP_DIR = Path(__file__).resolve().parent
REPO = EXP_DIR.parent
RUN_ONE = EXP_DIR / "run_one.py"

ALGOS = ["ddpg", "td3", "sac"]
COLORS = {
    "ddpg": "#1f77b4",
    "td3": "#ff7f0e",
    "sac": "#2ca02c",
    "ppo": "#d62728",
    "trpo": "#9467bd",
}
LABELS = {
    "ddpg": "DDPG",
    "td3": "TD3",
    "sac": "SAC",
    "ppo": "PPO",
    "trpo": "TRPO",
}

# Pendulum-v1 的参考刻度（文档第 4 节给出的归一化常数）。
R_RANDOM = -1200.0
R_BEST = -150.0
SMOOTH_WINDOW = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DDPG/TD3/SAC.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algos", nargs="+", default=ALGOS, choices=ALGOS)
    parser.add_argument("--episodes", type=int, default=200,
                        help="每个算法的训练回合数；Pendulum 每回合 200 步，200 回合=40k 步")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=5000,
                        help="periodic eval 间隔（env steps）；0 表示关闭")
    parser.add_argument("--periodic-eval-episodes", type=int, default=5)
    parser.add_argument("--outdir", default=str(EXP_DIR / "results"))
    parser.add_argument("--force", action="store_true", help="已有结果也重新训练")
    parser.add_argument("--plot-only", action="store_true", help="跳过训练，只聚合和画图")
    return parser.parse_args()


def moving_average(values, window_size=SMOOTH_WINDOW):
    """因果滑动平均（trailing window），输出长度与输入一致。

    窗口内的有效点数在序列开头会逐步增加到 window_size；这比直接
    np.convolve(mode='same') 更诚实，不会在开头用未来的点或除以固定窗口。
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.copy()

    cumulative_sum = np.cumsum(np.insert(values, 0, 0))
    indices = np.arange(1, values.size + 1)
    starts = np.maximum(0, indices - window_size)
    counts = indices - starts
    return ((cumulative_sum[indices] - cumulative_sum[starts]) / counts).astype(
        np.float32
    )


def run_training(args, raw_dir, model_dir):
    """按 algo × seed 顺序训练；顺序执行保证 wall-clock 不被并行争抢污染。"""
    for algo in args.algos:
        for seed in args.seeds:
            out_path = raw_dir / f"{algo}_seed{seed}.json"
            if out_path.exists() and not args.force:
                print(f"[skip] {out_path.name} 已存在（--force 可重跑）", flush=True)
                continue
            model_path = model_dir / f"{algo}_seed{seed}.pt"
            cmd = [
                sys.executable, str(RUN_ONE),
                "--algo", algo,
                "--seed", str(seed),
                "--episodes", str(args.episodes),
                "--eval-episodes", str(args.eval_episodes),
                "--eval-seed", str(args.eval_seed),
                "--eval-every", str(args.eval_every),
                "--periodic-eval-episodes", str(args.periodic_eval_episodes),
                "--out", str(out_path),
                "--model-out", str(model_path),
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


def align_curves(curves, steps_list):
    """把不同 seed 的曲线对齐到同一横轴；正常情况下 200×200 步完全一致。"""
    reference = np.asarray(steps_list[0])
    same = all(
        len(steps) == len(reference) and np.array_equal(steps, reference)
        for steps in steps_list
    )
    if same:
        return reference, [np.asarray(curve) for curve in curves]

    max_step = max(int(np.max(steps)) for steps in steps_list)
    grid = np.linspace(0, max_step, 400)
    aligned = [
        np.interp(grid, np.asarray(steps), np.asarray(curve))
        for steps, curve in zip(steps_list, curves)
    ]
    return grid, aligned


def median_iqr(values):
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    q1, q3 = np.percentile(values, [25, 75])
    return median, float(q1), float(q3), float(q3 - q1)


def normalized(value):
    return (value - R_RANDOM) / (R_BEST - R_RANDOM)


def compute_steps_to_threshold(runs, args, threshold_norm):
    rows = {}
    for algo in args.algos:
        for seed in args.seeds:
            run = runs[(algo, seed)]
            points = list(zip(
                run["periodic_eval"]["steps"],
                run["periodic_eval"]["returns"],
            ))
            # 把最终评估也作为一个点，避免 periodic eval 关闭时完全没数据。
            final_step = run["metrics"]["env_steps"]
            final_value = run["metrics"]["eval_mean"]
            if not points or points[-1][0] != final_step:
                points.append((final_step, final_value))
            points.sort()
            hit = next(
                (step for step, value in points
                 if normalized(value) >= threshold_norm),
                None,
            )
            rows[(algo, seed)] = hit
    return rows


def plot_training_curves(runs, args, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=200)

    for algo in args.algos:
        curves, steps_list = [], []
        for seed in args.seeds:
            run = runs[(algo, seed)]
            steps = np.asarray(run["train"]["steps"])
            returns = np.asarray(run["train"]["returns"])
            smoothed = moving_average(returns, SMOOTH_WINDOW)
            curves.append(smoothed)
            steps_list.append(steps)
            ax.plot(
                steps, smoothed, color=COLORS[algo],
                alpha=0.28, linewidth=0.9, zorder=1,
            )
        grid, aligned = align_curves(curves, steps_list)
        median = np.median(np.vstack(aligned), axis=0)
        ax.plot(
            grid, median, color=COLORS[algo], linewidth=2.6,
            label=LABELS[algo], zorder=3,
        )

    ax.axhline(R_RANDOM, color="0.6", linestyle="--", linewidth=0.9, zorder=0)
    ax.axhline(R_BEST, color="0.6", linestyle=":", linewidth=0.9, zorder=0)
    ax.text(
        ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 40000,
        R_RANDOM + 25, "random policy ≈ -1200",
        ha="right", va="bottom", fontsize=7.5, color="0.45",
    )
    ax.text(
        ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 40000,
        R_BEST - 45, "best known ≈ -150",
        ha="right", va="top", fontsize=7.5, color="0.45",
    )

    seed_label = f"{len(args.seeds)} seed" + ("s" if len(args.seeds) > 1 else "")
    ax.set_title(
        f"Pendulum-v1 · DDPG vs TD3 vs SAC ({seed_label})",
        fontsize=13,
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Smoothed training return (window = {SMOOTH_WINDOW})")
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)
    ax.set_xlim(left=0)

    handles = [
        Line2D([0], [0], color=COLORS[algo], lw=2.6, label=LABELS[algo])
        for algo in args.algos
    ]
    handles.append(
        Line2D([0], [0], color="0.4", lw=0.9, alpha=0.4,
               label="individual seeds")
    )
    ax.legend(handles=handles, loc="best", frameon=True, fontsize=9)

    fig.text(
        0.01, 0.005,
        "Thin lines = individual seeds; thick lines = median across seeds. "
        f"{args.episodes} episodes × 200 steps = {args.episodes * 200 // 1000}k env steps. "
        f"Final eval: {args.eval_episodes} deterministic episodes, EVAL_SEED={args.eval_seed}. "
        "Each algorithm keeps its reference hyperparameters (see summary.md).",
        fontsize=7.2, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_eval_curves(runs, args, out_path):
    """periodic 确定性评估曲线（文档第 8 节的 eval 学习曲线）。"""
    if args.eval_every <= 0:
        return
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=200)
    has_data = False

    for algo in args.algos:
        curves, steps_list = [], []
        for seed in args.seeds:
            run = runs[(algo, seed)]
            steps = np.asarray(run["periodic_eval"]["steps"])
            values = np.asarray(run["periodic_eval"]["returns"])
            if len(steps) == 0:
                continue
            has_data = True
            curves.append(values)
            steps_list.append(steps)
            ax.plot(
                steps, values, color=COLORS[algo],
                alpha=0.28, linewidth=0.9, marker="o", markersize=2.5,
            )
        if not curves:
            continue
        grid, aligned = align_curves(curves, steps_list)
        median = np.median(np.vstack(aligned), axis=0)
        ax.plot(grid, median, color=COLORS[algo], linewidth=2.4,
                marker="o", markersize=3.5, label=LABELS[algo])

    if not has_data:
        plt.close(fig)
        return

    ax.axhline(R_RANDOM, color="0.6", linestyle="--", linewidth=0.9)
    ax.axhline(R_BEST, color="0.6", linestyle=":", linewidth=0.9)
    ax.set_title("Pendulum-v1 · deterministic evaluation during training",
                 fontsize=13)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(
        f"Mean deterministic eval return ({args.periodic_eval_episodes} episodes)"
    )
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)
    handles = [
        Line2D([0], [0], color=COLORS[algo], lw=2.4, marker="o",
               label=LABELS[algo])
        for algo in args.algos
    ]
    handles.append(
        Line2D([0], [0], color="0.4", lw=0.9, alpha=0.4,
               label="individual seeds")
    )
    ax.legend(handles=handles, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_final_eval(runs, args, out_path):
    data = [
        [runs[(algo, seed)]["metrics"]["eval_mean"] for seed in args.seeds]
        for algo in args.algos
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=200)
    box = ax.boxplot(
        data, patch_artist=True, widths=0.45, showfliers=False,
        medianprops={"color": "black", "linewidth": 1.6},
    )
    for patch, algo in zip(box["boxes"], args.algos):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.28)
    for algo, values in zip(args.algos, data):
        index = args.algos.index(algo) + 1
        offsets = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            index + offsets, values, s=30, color=COLORS[algo],
            edgecolor="white", linewidth=0.7, zorder=3,
            label="individual seeds",
        )
        ax.annotate(
            f"median {np.median(values):.0f}",
            xy=(index, np.median(values)),
            xytext=(0, 14), textcoords="offset points",
            ha="center", fontsize=8, color="0.3",
        )

    ax.set_xticks(range(1, len(args.algos) + 1))
    ax.set_xticklabels([LABELS[algo] for algo in args.algos])
    ax.set_ylabel(
        f"Deterministic eval return ({args.eval_episodes} episodes)"
    )
    ax.set_title("Final performance after training", fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_results(runs, args, steps_to_threshold, out_dir):
    rows = []
    for algo in args.algos:
        for seed in args.seeds:
            metrics = runs[(algo, seed)]["metrics"]
            rows.append({
                "algo": algo,
                "seed": seed,
                "env_steps": metrics["env_steps"],
                "eval_mean": metrics["eval_mean"],
                "eval_median": metrics["eval_median"],
                "eval_std": metrics["eval_std"],
                "eval_best": metrics["eval_best"],
                "steps_to_90pct": steps_to_threshold.get((algo, seed)),
                "wall_clock_sec": metrics["wall_clock_sec"],
                "train_return_last10": metrics["train_return_last10"],
            })

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            rendered = {
                key: (
                    f"{value:.3f}"
                    if isinstance(value, float) else
                    ("NA" if value is None else value)
                )
                for key, value in row.items()
            }
            writer.writerow(rendered)

    summary = []
    for algo in args.algos:
        eval_values = [runs[(algo, seed)]["metrics"]["eval_mean"]
                       for seed in args.seeds]
        train_values = [runs[(algo, seed)]["metrics"]["train_return_last10"]
                        for seed in args.seeds]
        wall_values = [runs[(algo, seed)]["metrics"]["wall_clock_sec"]
                       for seed in args.seeds]
        step_values = [steps_to_threshold.get((algo, seed)) for seed in args.seeds]
        step_values = [value for value in step_values if value is not None]

        median, q1, q3, iqr = median_iqr(eval_values)
        summary.append({
            "algo": algo,
            "n_seeds": len(args.seeds),
            "eval_mean_median": median,
            "eval_mean_q1": q1,
            "eval_mean_q3": q3,
            "eval_mean_iqr": iqr,
            "train_last10_median": float(np.median(train_values)),
            "wall_clock_median": float(np.median(wall_values)),
            "steps_to_90pct_median": (
                float(np.median(step_values)) if step_values else None
            ),
        })

    with (out_dir / "results_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            rendered = {
                key: (
                    f"{value:.3f}"
                    if isinstance(value, float) else
                    ("NA" if value is None else value)
                )
                for key, value in row.items()
            }
            writer.writerow(rendered)

    return rows, summary


def write_summary(runs, args, rows, summary, steps_to_threshold, out_dir):
    first = runs[(args.algos[0], args.seeds[0])]
    config = first["config"]
    ranked = sorted(summary, key=lambda row: row["eval_mean_median"], reverse=True)
    step_rows = [row for row in summary if row["steps_to_90pct_median"] is not None]
    fastest = min(step_rows, key=lambda row: row["steps_to_90pct_median"]) \
        if step_rows else None

    lines = [
        "# DDPG / TD3 / SAC 对比实验",
        "",
        f"- 日期：2026-10-25",
        f"- 环境：`Pendulum-v1`",
        f"- seeds：{', '.join(str(seed) for seed in args.seeds)}（n={len(args.seeds)}）",
        f"- 训练：每个算法 {args.episodes} 回合 × 200 步 = "
        f"{args.episodes * 200} env steps",
        f"- 最终评估：{args.eval_episodes} 回合确定性策略，"
        f"`EVAL_SEED={args.eval_seed}`（所有算法共用同一组初始状态）",
        f"- periodic eval：每 {args.eval_every} 步评估 "
        f"{args.periodic_eval_episodes} 回合（确定性）",
        "",
        "## 统一口径说明",
        "",
        "- 三个算法都训练到相同 env steps 数，横轴统一为 env steps；",
        "- 训练回报含各自探索噪声，**只用于画训练曲线，不用于算法间比较**；",
        "- 性能数字唯一来源是训练结束后的确定性评估；",
        "- 每个算法保留自己的参考超参数（见下表），没有为了“公平”强行统一网络宽度和学习率；",
        "- SAC 的 `REWARD_SHIFT/REWARD_SCALE` 只作用于 critic 的 Bellman target，"
        "`return_list` 仍是环境原始回报。",
        "",
        "## 各算法超参数（config.py 快照）",
        "",
        "| 算法 | GAMMA | ACTOR_LR | CRITIC_LR | HIDDEN_DIM | BATCH_SIZE | BUFFER_SIZE | 其他 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for algo in args.algos:
        cfg = runs[(algo, args.seeds[0])]["config"]
        extra = []
        if "NOISE_SIGMA" in cfg:
            extra.append(f"NOISE_SIGMA={cfg['NOISE_SIGMA']}")
        if "EXPLORATION_SIGMA" in cfg:
            extra.append(f"EXPLORATION_SIGMA={cfg['EXPLORATION_SIGMA']}")
        if "POLICY_DELAY" in cfg:
            extra.append(f"POLICY_DELAY={cfg['POLICY_DELAY']}")
        if "REWARD_SHIFT" in cfg:
            extra.append(
                f"REWARD_SHIFT={cfg['REWARD_SHIFT']}, "
                f"REWARD_SCALE={cfg['REWARD_SCALE']}"
            )
        lines.append(
            f"| {LABELS[algo]} | {cfg.get('GAMMA')} | {cfg.get('ACTOR_LR')} | "
            f"{cfg.get('CRITIC_LR')} | {cfg.get('HIDDEN_DIM')} | "
            f"{cfg.get('BATCH_SIZE')} | {cfg.get('BUFFER_SIZE')} | "
            f"{', '.join(extra)} |"
        )

    lines += [
        "",
        f"## 结果（{len(args.seeds)} seeds，中位数 [Q1, Q3]）",
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
        "## 图",
        "",
        "- `comparison_training_curves.png`：图 1（细线=单 seed，粗线=中位数）",
        "- `comparison_eval_curves.png`：训练过程中的确定性评估曲线",
        "- `comparison_final_eval.png`：最终评估的箱线图 + 单 seed 散点",
        "",
        "## 观察（只描述，不做显著性结论）",
        "",
        f"- 最终评估中位数排序："
        + " > ".join(
            f"{LABELS[row['algo']]} ({row['eval_mean_median']:.1f})"
            for row in ranked
        ),
    ]
    if fastest is not None:
        lines.append(
            f"- 达到 90% 阈值（以 9 次运行的最佳最终性能归一化）最快的是 "
            f"{LABELS[fastest['algo']]}，中位数约 "
            f"{fastest['steps_to_90pct_median']:.0f} env steps。"
        )
    lines += [
        "- 方差看 IQR 和图中的细线；3 个 seed 下不要对“谁显著更好”下结论。",
        "",
        "## 局限",
        "",
        "- n=3：做不了 IQM / bootstrap 置信区间，也不应做显著性检验；",
        "- 每个算法用的是仓库里的参考超参数，不是同一套超参数下的公平扫描；",
        "- Pendulum 是教学环境，不是公认 benchmark；这张图用于验证评估协议和实现，"
        "不能据此宣称算法优劣；",
        "- wall-clock 包含 periodic eval 的开销，且受 GPU 状态影响，只作参考。",
        "",
        "## 复现",
        "",
        "```powershell",
        "cd \"C:\\Users\\PC\\Desktop\\具身智能实习\\hands-on-reinforcement-learning\"",
        ".venv\\Scripts\\python.exe experiments\\run_compare.py --seeds 0 1 2",
        "```",
        "",
        "原始逐回合数据在 `results/raw/`，模型在 `results/models/`。",
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

    # 阈值：9 次运行中最好的最终评估性能，按 Pendulum 参考刻度归一化后取 90%。
    best_final = max(run["metrics"]["eval_mean"] for run in runs.values())
    threshold_norm = 0.9 * normalized(best_final)
    steps_to_threshold = compute_steps_to_threshold(
        runs, args, threshold_norm
    )

    rows, summary = write_results(runs, args, steps_to_threshold, out_dir)
    plot_training_curves(
        runs, args, out_dir / "comparison_training_curves.png"
    )
    plot_eval_curves(runs, args, out_dir / "comparison_eval_curves.png")
    plot_final_eval(runs, args, out_dir / "comparison_final_eval.png")
    write_summary(runs, args, rows, summary, steps_to_threshold, out_dir)

    protocol = {
        "seeds": args.seeds,
        "algos": args.algos,
        "episodes": args.episodes,
        "env_steps": args.episodes * 200,
        "eval_episodes": args.eval_episodes,
        "eval_seed": args.eval_seed,
        "eval_every": args.eval_every,
        "periodic_eval_episodes": args.periodic_eval_episodes,
        "smooth_window": SMOOTH_WINDOW,
        "normalization": {"random": R_RANDOM, "best": R_BEST},
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
