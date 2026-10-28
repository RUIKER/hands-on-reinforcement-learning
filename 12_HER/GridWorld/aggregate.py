"""聚合 results/raw 下的单次实验，输出 CSV、summary.md 和图。

指标（都按回合数，不按步数；一回合最多 50 步）：
- first_success：训练中第一次到达目标（回报 > -50）的回合；
- converge_eval：固定评估集成功率 >= 0.9 且此后不再跌破 0.9 的最早回合；
- converge_return：训练回报 20 回合滑动平均首次 > -10 的回合
  （成功回报约 -4，失败恒为 -50；这个指标不依赖评估，可与教程原版直接比较）；
- final_success：训练结束后在固定 20 个目标上的确定性成功率。
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def converge_return(returns, window=20, threshold=-10.0):
    r = np.asarray(returns, dtype=float)
    if len(r) < window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode='valid')
    hit = np.where(ma > threshold)[0]
    return int(hit[0] + window) if len(hit) else None


def first_success(returns):
    hit = np.where(np.asarray(returns) > -49.5)[0]
    return int(hit[0] + 1) if len(hit) else None


def converge_eval(history, level=0.9):
    episodes = [e for e, _ in history]
    values = [s for _, s in history]
    for i in range(len(values)):
        if all(v >= level for v in values[i:]):
            return int(episodes[i])
    return None


def load(raw_dir):
    runs = []
    for path in sorted(Path(raw_dir).glob('*.json')):
        d = json.loads(path.read_text(encoding='utf-8'))
        d['name'] = path.stem
        d['first_success'] = first_success(d['returns'])
        d['converge_eval'] = converge_eval(d['success_history'])
        d['converge_return'] = converge_return(d['returns'])
        runs.append(d)
    return runs


def group_key(d):
    if not d['her']:
        return 'no HER'
    return 'HER ratio=%.1f %s' % (d['ratio'], d['strategy'])


def median_iqr(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, None
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), float(q1), float(q3)


def fmt(values, n_total):
    med, q1, q3 = median_iqr(values)
    got = len([v for v in values if v is not None])
    if med is None:
        return 'never (0/%d)' % n_total
    tag = '' if got == n_total else ' (%d/%d)' % (got, n_total)
    return '%.0f [%.0f, %.0f]%s' % (med, q1, q3, tag)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default=str(HERE / 'results' / 'raw'))
    parser.add_argument('--out', default=str(HERE / 'results'))
    parser.add_argument('--prefix', default='')
    args = parser.parse_args()

    out = Path(args.out)
    runs = load(args.raw)
    groups = {}
    for d in runs:
        groups.setdefault(group_key(d), []).append(d)

    with (out / (args.prefix + 'runs.csv')).open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['group', 'seed', 'first_success', 'converge_eval',
                    'converge_return', 'final_success', 'wall_clock_sec'])
        for key, items in groups.items():
            for d in sorted(items, key=lambda x: x['seed']):
                w.writerow([key, d['seed'], d['first_success'], d['converge_eval'],
                            d['converge_return'], d['final_success'],
                            round(d['wall_clock_sec'], 1)])

    lines = ['| 设置 | seeds | 首次成功回合 | 收敛回合（评估） | 收敛回合（回报） | 最终成功率 | ≥0.9 seeds | 耗时 s |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key, items in groups.items():
        n = len(items)
        fs = [d['first_success'] for d in items]
        ce = [d['converge_eval'] for d in items]
        cr = [d['converge_return'] for d in items]
        fin = [d['final_success'] for d in items]
        wall = [d['wall_clock_sec'] for d in items]
        good = int(sum(1 for v in fin if v >= 0.9))
        lines.append('| %s | %d | %s | %s | %s | %.2f [%.2f, %.2f] | %d/%d | %.0f |' % (
            key, n, fmt(fs, n), fmt(ce, n), fmt(cr, n),
            np.median(fin), np.percentile(fin, 25), np.percentile(fin, 75),
            good, n, np.median(wall)))
    table = '\n'.join(lines)
    (out / (args.prefix + 'summary_table.md')).write_text(table + '\n', encoding='utf-8')
    print(table)

    # 图：每组成功率曲线，细线 = 单 seed，粗线 = 中位数
    palette = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    for color, (key, items) in zip(palette, groups.items()):
        episodes = [e for e, _ in items[0]['success_history']]
        mat = np.array([[s for _, s in d['success_history']] for d in items])
        for row in mat:
            ax.plot(episodes, row, color=color, alpha=0.18, linewidth=0.8)
        ax.plot(episodes, np.median(mat, axis=0), color=color, linewidth=2.4,
                label='%s (n=%d)' % (key, len(items)))
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success rate on 20 fixed goals')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_title('GridWorld: DDPG with / without HER')
    fig.text(0.01, 0.005, 'Thin = single seed, thick = median across seeds. '
             'Deterministic evaluation on fixed goals.', fontsize=7, color='0.4')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / (args.prefix + 'success_curves.png'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
