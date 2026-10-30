"""聚合 SAC 超参单因素扫描，并与 1M 基线的前 100k 步对比。

指标：评估网格 60k/80k/100k 三点的均值（每点 5 回合），减少单点噪声。
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE_RAW = HERE.parents[1] / 'seed_experiments' / 'halfcheetah_sac' / 'results' / 'raw'
MAX_STEP = 100_000
LATE = (60_000, 80_000, 100_000)
SEEDS = (0, 1, 2)
ORDER = ['baseline', 'lr1e-4', 'lr1e-3', 'batch64', 'batch1024', 'gamma0.95', 'alpha1.0']
LABEL = {
    'baseline': 'baseline (lr3e-4, b256, g.99, a0.01)',
    'lr1e-4': 'lr 1e-4', 'lr1e-3': 'lr 1e-3',
    'batch64': 'batch 64', 'batch1024': 'batch 1024',
    'gamma0.95': 'gamma 0.95', 'alpha1.0': 'init alpha 1.0',
}


def curve(d):
    """截取基线 run 前 100k 步的评估序列。

    基线每 20k 步评估一次，所以截断后正好是 20k/40k/60k/80k/100k 五个点，
    与扫描 run 的评估网格逐点对齐，可以直接比较。
    """
    s = np.array(d['eval']['steps']); r = np.array(d['eval']['returns'])
    keep = s <= MAX_STEP
    return s[keep], r[keep]


def load(name):
    out = {}
    for seed in SEEDS:
        path = (BASE_RAW / f'sac_s{seed}.json') if name == 'baseline' \
            else HERE / 'results' / 'raw' / f'{name}_s{seed}.json'
        if path.exists():
            out[seed] = curve(json.loads(path.read_text(encoding='utf-8')))
    return out


def main():
    # 所有配置都用 LATE 三点的均值做对比：单看 100k 一个点容易被评估噪声影响，
    # 三点平均对学习曲线的抖动更稳健。跨 seed 标准差一并报告，
    # 差异小于标准差的配置不要排序。
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    groups = {n: load(n) for n in ORDER}
    rows, lines = [], [
        '| 配置 | seeds | 60k/80k/100k 均值（3 seeds 均值 ± 跨 seed std） | 100k 单点均值 | 相对基线 |',
        '| --- | ---: | ---: | ---: | ---: |']
    base_late = None
    for n in ORDER:
        runs = groups[n]
        if not runs:
            continue
        late = []
        end = []
        for s, r in runs.values():
            mask = np.isin(s, LATE)
            late.append(r[mask].mean()); end.append(r[-1])
        late, end = np.array(late), np.array(end)
        if n == 'baseline':
            base_late = late.mean()
        rel = '' if base_late is None or n == 'baseline' else '%+.0f%%' % (
            100 * (late.mean() - base_late) / abs(base_late))
        lines.append('| %s | %d | %.0f ± %.0f | %.0f | %s |' % (
            LABEL[n], len(runs), late.mean(), late.std(), end.mean(), rel))
        rows.append([n, len(runs), round(late.mean(), 1), round(late.std(), 1),
                     round(end.mean(), 1)])
    table = '\n'.join(lines)
    (out / 'sweep_table.md').write_text(table + '\n', encoding='utf-8')
    with (out / 'sweep_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['config', 'seeds', 'late_mean', 'late_std', 'end_mean'])
        w.writerows(rows)
    print(table)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    cmap = plt.get_cmap('tab10')
    for i, n in enumerate(ORDER):
        runs = groups[n]
        if not runs:
            continue
        steps = next(iter(runs.values()))[0]
        mat = np.array([r for _, r in runs.values()])
        ax.plot(steps, mat.mean(axis=0), color='black' if n == 'baseline' else cmap(i),
                linewidth=2.6 if n == 'baseline' else 1.8, label=LABEL[n])
        if len(mat) > 1:
            ax.fill_between(steps, mat.min(axis=0), mat.max(axis=0),
                            color='black' if n == 'baseline' else cmap(i), alpha=0.07)
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Deterministic eval return (5 episodes)')
    ax.set_title('SAC HalfCheetah-v5 hyperparameters (mean, min–max over 3 seeds)')
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(out / 'sweep_curves.png', bbox_inches='tight')
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    main()
