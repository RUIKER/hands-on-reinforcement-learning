"""画 PPO 的 gamma 对照图：gamma=0.99（3 seeds）vs gamma=0.9（1 seed），
并用 SAC 的 40k 曲线作为样本效率参照。

用法：
    python experiments/plot_gamma_compare.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onpolicy-dir",
                        default=str(EXP_DIR / "results_onpolicy"))
    parser.add_argument("--gamma09-dir",
                        default=str(EXP_DIR / "results_onpolicy_gamma09"))
    parser.add_argument("--sac-dir", default=str(EXP_DIR / "results" / "raw"))
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def eval_curve(run):
    return (
        np.asarray(run["periodic_eval"]["steps"], dtype=float),
        np.asarray(run["periodic_eval"]["returns"], dtype=float),
    )


def main():
    args = parse_args()
    onpolicy_dir = Path(args.onpolicy_dir)
    gamma09_dir = Path(args.gamma09_dir)
    sac_dir = Path(args.sac_dir)
    out_path = Path(args.out) if args.out else (
        onpolicy_dir / "fig2b_ppo_gamma_compare.png"
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=200)

    # SAC（40k，已收敛）作为参照。
    sac_curves = []
    for seed in (0, 1, 2):
        path = sac_dir / f"sac_seed{seed}.json"
        if not path.exists():
            continue
        steps, values = eval_curve(load(path))
        sac_curves.append(values)
        ax.plot(steps, values, color="#2ca02c", alpha=0.22, linewidth=0.9)
    if sac_curves:
        steps, _ = eval_curve(load(sac_dir / "sac_seed0.json"))
        ax.plot(
            steps, np.median(np.vstack(sac_curves), axis=0),
            color="#2ca02c", linewidth=2.6, marker="o", markersize=3.5,
            label="SAC (40k, 3 seeds)",
        )

    # PPO gamma=0.99（3 seeds）。
    ppo99_curves = []
    ppo99_steps = None
    for seed in (0, 1, 2):
        path = onpolicy_dir / "raw" / f"ppo_seed{seed}.json"
        if not path.exists():
            continue
        steps, values = eval_curve(load(path))
        ppo99_curves.append(values)
        ppo99_steps = steps
        ax.plot(steps, values, color="#d62728", alpha=0.25, linewidth=0.9)
    if ppo99_curves:
        ax.plot(
            ppo99_steps, np.median(np.vstack(ppo99_curves), axis=0),
            color="#d62728", linewidth=2.6, marker="o", markersize=3.5,
            label="PPO γ=0.99 (3 seeds)",
        )

    # PPO gamma=0.9（seed 0，诊断）。
    path09 = gamma09_dir / "raw" / "ppo_seed0.json"
    if path09.exists():
        run09 = load(path09)
        steps09, values09 = eval_curve(run09)
        ax.plot(
            steps09, values09, color="#ff7f0e", linewidth=2.2,
            linestyle="--", marker="s", markersize=3.2,
            label="PPO γ=0.9 (seed 0, diagnostic)",
        )
        ax.annotate(
            f"final {run09['metrics']['eval_mean']:.0f}",
            xy=(steps09[-1], values09[-1]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=8, color="#b35c00",
        )

    ax.axhline(-1200, color="0.6", linestyle="--", linewidth=0.9)
    ax.axhline(-150, color="0.6", linestyle=":", linewidth=0.9)
    ax.set_xlim(left=0)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean deterministic eval return (5 episodes)")
    ax.set_title("PPO gamma comparison on Pendulum-v1 (SAC as reference)",
                 fontsize=13)
    ax.grid(alpha=0.25)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="0.4", lw=0.9, alpha=0.4,
                          label="individual seeds"))
    ax.legend(handles=handles, loc="best", fontsize=9)

    fig.text(
        0.01, 0.005,
        "Thin lines = individual seeds; thick = median. PPO γ=0.9 is a "
        "single-seed diagnostic, not part of the 3-seed table. "
        "SAC was trained for 40k steps only (already converged).",
        fontsize=7.2, color="0.35",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
