"""五算法合并对比图：DDPG / TD3 / SAC / PPO / TRPO 放在同一张图里。

- 左图：训练过程中的确定性评估（各 seed 中位数 + min/max 阴影），横轴 env steps 取对数，
  这样 40k 的 off-policy 和 400k 的 on-policy 曲线都能看清；
- 右图：最终 20 回合确定性评估，每个点是一个 seed，竖线是中位数；
- off-policy 实线 / on-policy 虚线 + 不同 marker + 线尾直接标注，颜色不是唯一区分手段。

只读 results/raw 和 results_onpolicy/raw 下已有的 JSON，不重新训练。

用法：
    python experiments/plot_all_algos.py
    python experiments/plot_all_algos.py --ppo-fixed-dir path/to/raw   # 叠加修正版 PPO
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent

ALGOS = ["ddpg", "td3", "sac", "ppo", "trpo"]
OFF_POLICY = {"ddpg", "td3", "sac"}
LABELS = {"ddpg": "DDPG", "td3": "TD3", "sac": "SAC", "ppo": "PPO", "trpo": "TRPO",
          "ppo_fixed": "PPO (fixed)"}
# 固定顺序的分类色；5 色无法单靠颜色区分，因此同时用线型/marker/直接标注。
COLORS = {"ddpg": "#2a78d6", "td3": "#eb6834", "sac": "#1baf7a",
          "ppo": "#eda100", "trpo": "#e87ba4", "ppo_fixed": "#eda100"}
MARKERS = {"ddpg": "o", "td3": "s", "sac": "D", "ppo": "^", "trpo": "v",
           "ppo_fixed": "^"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
R_RANDOM = -1200.0


def parse_args():
    parser = argparse.ArgumentParser(description="Plot all five algorithms together.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--off-raw-dir", default=str(EXP_DIR / "results" / "raw"))
    parser.add_argument("--on-raw-dir", default=str(EXP_DIR / "results_onpolicy" / "raw"))
    parser.add_argument("--ppo-fixed-dir", default=None,
                        help="可选：修正版 PPO 的 raw JSON 目录，叠加为 PPO (fixed)")
    parser.add_argument("--out", default=str(EXP_DIR / "results_all" / "fig_all_algos.png"))
    return parser.parse_args()


def load(raw_dir, algo, seeds):
    runs = []
    for seed in seeds:
        path = Path(raw_dir) / f"{algo}_seed{seed}.json"
        if not path.exists():
            raise SystemExit(f"缺少结果文件：{path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def eval_matrix(runs):
    """把各 seed 的 periodic eval 对齐成 [n_seeds, n_points]。"""
    steps = np.asarray(runs[0]["periodic_eval"]["steps"])
    values = np.asarray([run["periodic_eval"]["returns"][:len(steps)] for run in runs])
    return steps, values


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def main():
    args = parse_args()
    series = []
    for algo in ALGOS:
        raw_dir = args.off_raw_dir if algo in OFF_POLICY else args.on_raw_dir
        series.append((algo, load(raw_dir, algo, args.seeds)))
    if args.ppo_fixed_dir:
        series.append(("ppo_fixed", load(args.ppo_fixed_dir, "ppo", args.seeds)))

    fig, (ax_curve, ax_final) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [2.3, 1]},
        facecolor=SURFACE,
    )

    # ---- 左：评估曲线 ----
    style_axis(ax_curve)
    end_labels = []
    for algo, runs in series:
        steps, values = eval_matrix(runs)
        median = np.median(values, axis=0)
        dashed = algo not in OFF_POLICY and algo != "ppo_fixed"
        # 原 PPO 在叠加修正版时退为背景参照。
        faded = algo == "ppo" and args.ppo_fixed_dir
        alpha = 0.45 if faded else 1.0
        ax_curve.fill_between(steps, values.min(axis=0), values.max(axis=0),
                              color=COLORS[algo], alpha=0.10 * alpha, linewidth=0)
        ax_curve.plot(steps, median, color=COLORS[algo], alpha=alpha,
                      linewidth=2, linestyle="--" if dashed else "-",
                      marker=MARKERS[algo], markersize=5,
                      # on-policy 每 10k 一个点，对数轴右端会挤成一团，只隔点画 marker。
                      markevery=1 if algo in OFF_POLICY else 4,
                      markeredgecolor=SURFACE, markeredgewidth=1,
                      label=LABELS[algo])
        end_labels.append([steps[-1], median[-1], algo])

    # 线尾标注按 y 排开，避免 TD3/SAC 这种几乎重合的线标签相撞。
    for x_end in {item[0] for item in end_labels}:
        group = sorted((item for item in end_labels if item[0] == x_end),
                       key=lambda item: item[1])
        for lower, upper in zip(group, group[1:]):
            if upper[1] - lower[1] < 70:
                upper[1] = lower[1] + 70
    for x_end, y_label, algo in end_labels:
        ax_curve.annotate(LABELS[algo], (x_end, y_label), xytext=(7, 0),
                          textcoords="offset points", va="center",
                          fontsize=9, color=INK, fontweight="bold")

    ax_curve.axhline(R_RANDOM, color=INK_2, linestyle=":", linewidth=1)
    ax_curve.text(4.3e3, R_RANDOM + 20, "random policy ≈ -1200",
                  fontsize=8, color=INK_2, va="bottom")
    ax_curve.axvline(40000, color=GRID, linewidth=1.5, zorder=0)
    ax_curve.text(41000, -1640, "off-policy budget\n(40k)", fontsize=8,
                  color=INK_2, va="bottom")
    ax_curve.set_xscale("log")
    ax_curve.set_xlim(4e3, 7.5e5)
    ax_curve.set_ylim(-1650, 0)
    ax_curve.set_xticks([5e3, 1e4, 2e4, 5e4, 1e5, 2e5, 4e5])
    ax_curve.set_xticklabels(["5k", "10k", "20k", "50k", "100k", "200k", "400k"])
    ax_curve.set_xlabel("Environment steps (log scale)", color=INK_2, fontsize=10)
    ax_curve.set_ylabel("Deterministic eval return (median of seeds)",
                        color=INK_2, fontsize=10)
    ax_curve.set_title("Evaluation during training", color=INK, fontsize=12,
                       loc="left", pad=34)
    legend = ax_curve.legend(loc="lower left", fontsize=9, frameon=False,
                             ncol=len(series), bbox_to_anchor=(0.0, 1.06),
                             handlelength=2.6, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(INK)

    # ---- 右：最终评估 ----
    style_axis(ax_final)
    ax_final.grid(axis="y", visible=False)
    order = sorted(series, key=lambda item: np.median(
        [run["metrics"]["eval_mean"] for run in item[1]]))
    for row, (algo, runs) in enumerate(order):
        finals = np.asarray([run["metrics"]["eval_mean"] for run in runs])
        median = float(np.median(finals))
        ax_final.plot([finals.min(), finals.max()], [row, row],
                      color=COLORS[algo], linewidth=2, alpha=0.35,
                      solid_capstyle="round")
        ax_final.scatter(finals, np.full_like(finals, row), s=46,
                         color=COLORS[algo], marker=MARKERS[algo],
                         edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax_final.plot([median, median], [row - 0.22, row + 0.22],
                      color=INK, linewidth=1.5, solid_capstyle="round", zorder=2)
        ax_final.annotate(f"{median:.0f}", (median, row + 0.24),
                          ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax_final.set_yticks(range(len(order)))
    ax_final.set_yticklabels([LABELS[algo] for algo, _ in order], color=INK,
                             fontsize=10)
    ax_final.set_ylim(-0.6, len(order) - 0.2)
    ax_final.set_xlim(-1650, 0)
    ax_final.set_xticks([-1600, -1200, -800, -400, 0])
    ax_final.set_xlabel("Final eval return (20 ep, per seed)", color=INK_2,
                        fontsize=10)
    ax_final.set_title("Final performance  (| = median)", color=INK,
                       fontsize=12, loc="left", pad=34)

    budgets = "off-policy 40k steps · on-policy 400k steps"
    fig.suptitle(f"Pendulum-v1 · five algorithms · {len(args.seeds)} seeds · {budgets}",
                 color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
