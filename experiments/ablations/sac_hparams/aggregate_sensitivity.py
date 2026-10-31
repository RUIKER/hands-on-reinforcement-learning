"""SAC 超参灵敏度图：把 grid.py 里所有配置聚合成“灵敏度地图”。

产出（results/）：
    sensitivity_map.png       每个超参一个小图：横轴=超参取值，纵轴=后期回报（60/80/100k 均值）
    sensitivity_early.png     同上，指标换成早期速度（20/40k 均值）
    sensitivity_tornado.png   每个超参相对基线的最差/最好变化，灰带 = 基线跨 seed 噪声
    sensitivity_table.md/csv  逐配置：均值、相对基线、配对差、t、p、Holm 校正 p、同号 seed 数
    sensitivity_summary.md    逐超参：最差/最好变化、噪声倍数、粗分类

读图规则：
    - 圆点是单个 seed，线是 3 个 seed 的均值，灰带是基线 ± 1 个跨 seed 样本标准差；
    - ★ = 3/3 seed 配对差同号，且平均配对差 > 2 倍基线跨 seed 标准差（“一致且大”）；
      ✕ = 至少一个 seed 发散。注意：n=3（df=2）下对 ~28 个比较做 Holm 校正几乎不可能
      通过，所以 p / Holm p 只在表里作参考，★ 用效应量 + 方向一致性判定；
    - n<3 的点是还没跑完的，只画不检验。
分类是启发式的，只用来决定阅读顺序，不是统计结论。
"""

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from grid import GRID

HERE = Path(__file__).resolve().parent
BASE_RAW = HERE.parents[1] / 'seed_experiments' / 'halfcheetah_sac' / 'results' / 'raw'
RAW = HERE / 'results' / 'raw'
OUT = HERE / 'results'
SEEDS = (0, 1, 2)
MAX_STEP = 100_000
LATE = (60_000, 80_000, 100_000)
EARLY = (20_000, 40_000)


def nanmean(x):
    """忽略 NaN 的均值；全是 NaN（发散）时返回 NaN，不触发 numpy 警告。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def load_run(path):
    """读一个 run，返回 {'late','early','diverged'}；文件不存在返回 None。"""
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding='utf-8'))
    s = np.array(d['eval']['steps'])
    r = np.array(d['eval']['returns'], dtype=float)
    keep = s <= MAX_STEP
    s, r = s[keep], r[keep]
    return {
        'late': nanmean(r[np.isin(s, LATE)]),
        'early': nanmean(r[np.isin(s, EARLY)]),
        'diverged': bool(d.get('diverged_at_step') is not None or np.isnan(r).any()),
    }


def load_config(name):
    """name=None 表示基线（1M run 的前 100k 步）。返回 {seed: run}。"""
    out = {}
    for seed in SEEDS:
        path = (BASE_RAW / f'sac_s{seed}.json') if name is None \
            else RAW / f'{name}_s{seed}.json'
        run = load_run(path)
        if run is not None:
            out[seed] = run
    return out


def holm(pvals):
    """Holm–Bonferroni 校正；输入 {key: p}，输出 {key: p_adj}。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, adj, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def build_records(groups, base, base_std):
    """逐配置统计：均值、与基线的配对差、t、p；再对所有 p 做 Holm 校正。"""
    records, pvals = [], {}
    for g in groups:
        for value, name, _ in g['points']:
            runs = base if name is None else load_config(name)
            seeds = sorted(runs)
            late = np.array([runs[s]['late'] for s in seeds])
            early = np.array([runs[s]['early'] for s in seeds])
            rec = dict(key=g['key'], value=value, name=name or 'baseline', seeds=seeds,
                       late=late, early=early,
                       diverged=sum(runs[s]['diverged'] for s in seeds),
                       t=np.nan, p=np.nan, diff=np.nan, same=0)
            # 只有 3 个 seed 都在、且都没发散时才做配对检验。
            if name is not None and len(seeds) == 3 and np.isfinite(late).all():
                diff = late - np.array([base[s]['late'] for s in seeds])
                rec['diff'] = diff.mean()
                rec['same'] = int(max((diff > 0).sum(), (diff < 0).sum()))
                if diff.std(ddof=1) > 0:
                    rec['t'] = diff.mean() / (diff.std(ddof=1) / np.sqrt(3))
                    rec['p'] = 2 * stats.t.sf(abs(rec['t']), df=2)
                    pvals[rec['name']] = rec['p']
            records.append(rec)
    padj = holm(pvals)
    for r in records:
        r['p_holm'] = padj.get(r['name'], np.nan)
        # 效应量 + 方向一致性：n=3 时 p 值本身不可靠，不拿它当门槛。
        r['sig'] = bool(r['same'] == 3 and np.isfinite(r['diff'])
                        and abs(r['diff']) > 2 * base_std)
    return records


def fmt(x, spec):
    return '' if (x == '' or (isinstance(x, float) and np.isnan(x))) else spec % x


def write_tables(records, base_mean):
    header = ['hyper', 'value', 'config', 'n', 'late_mean', 'late_std', 'rel_pct',
              'paired_diff', 't', 'p', 'p_holm', 'same_sign', 'early_mean', 'diverged']
    rows = []
    for r in records:
        lm = nanmean(r['late'])
        std = float(np.nanstd(r['late'], ddof=1)) if len(r['seeds']) > 1 else float('nan')
        rel = '' if r['name'] == 'baseline' else 100 * (lm - base_mean) / abs(base_mean)
        rows.append([r['key'], r['value'], r['name'], len(r['seeds']), lm, std, rel,
                     r['diff'], r['t'], r['p'], r['p_holm'],
                     '' if r['name'] == 'baseline' else '%d/3' % r['same'],
                     nanmean(r['early']), r['diverged']])
    with (OUT / 'sensitivity_results.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in rows:
            w.writerow([('' if (isinstance(v, float) and np.isnan(v)) else
                         (round(v, 3) if isinstance(v, float) else v)) for v in row])

    md = ['| 超参 | 取值 | 配置 | n | 后期回报 ± std | 相对基线 | 配对差 | t | p | Holm p | 同号 | 早期回报 | 发散 |',
          '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row, r in zip(rows, records):
        star = ' ★' if r['sig'] else ''
        rel = fmt(row[6], '%+.0f%%') + star if row[6] != '' else ''
        md.append('| %s | %g | %s | %d | %s ± %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            row[0], row[1], row[2], row[3], fmt(row[4], '%.0f') or 'NaN',
            fmt(row[5], '%.0f') or '—', rel, fmt(row[7], '%+.0f'), fmt(row[8], '%.2f'),
            fmt(row[9], '%.3f'), fmt(row[10], '%.3f'), row[11],
            fmt(row[12], '%.0f') or 'NaN', '%d/%d' % (row[13], row[3]) if row[13] else '0'))
    (OUT / 'sensitivity_table.md').write_text('\n'.join(md) + '\n', encoding='utf-8')


def write_summary(groups, records, base_mean, base_std, noise_rel, n_base):
    """逐超参摘要 + 返回 tornado 数据。"""
    lines = ['基线后期回报 %.0f ± %.0f（跨 seed 噪声 %.1f%%，n=%d）。' % (
        base_mean, base_std, 100 * noise_rel, n_base), '',
        '| 超参 | 最差点 | 最好点 | 范围/噪声 | ★ 点数 | 发散 run 数 | 粗分类 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    tornado = []
    for g in groups:
        recs = [r for r in records if r['key'] == g['key'] and r['name'] != 'baseline'
                and np.isfinite(nanmean(r['late']))]
        if not recs:
            continue
        rel = {r['name']: (nanmean(r['late']) - base_mean) / abs(base_mean) for r in recs}
        wname, bname = min(rel, key=rel.get), max(rel, key=rel.get)
        worst, best = rel[wname], rel[bname]
        span = (max(0.0, best) - min(0.0, worst)) / noise_rel
        nsig = sum(r['sig'] for r in recs)
        ndiv = sum(r['diverged'] for r in records if r['key'] == g['key'])
        if nsig:
            cls = '敏感（有 ★：3/3 同号且 >2σ）'
        elif max(abs(worst), abs(best)) < noise_rel:
            cls = '钝感（全部在噪声内）'
        else:
            cls = '待定（有超噪声的点，但不满足 3/3 同号且 >2σ）'
        lines.append('| %s | %s %+.0f%% | %s %+.0f%% | %.1f× | %d | %d | %s |' % (
            g['key'], wname, 100 * worst, bname, 100 * best, span, nsig, ndiv, cls))
        tornado.append((g['key'], worst, best, nsig))
    (OUT / 'sensitivity_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    return tornado


def small_multiples(groups, records, metric, base_vals, fname, ylabel):
    """每个超参一张小图：圆点=单 seed，线=均值，灰带=基线±1std，竖线=基线取值。"""
    n, cols = len(groups), 5
    rows_n = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.1 * cols, 3.5 * rows_n), dpi=150)
    axes = np.atleast_1d(axes).ravel()
    bmean, bstd = base_vals.mean(), base_vals.std(ddof=1)
    for ax, g in zip(axes, groups):
        recs = [r for r in records if r['key'] == g['key']]
        gamma = g['key'] == 'gamma'
        # gamma 用有效视野 1/(1-gamma) 做横轴（对数），比直接画 gamma 更能看出间隔。
        xs = [1 / (1 - r['value']) if gamma else r['value'] for r in recs]
        order = np.argsort(xs)
        ax.axhspan(bmean - bstd, bmean + bstd, color='gray', alpha=0.18, zorder=0)
        ax.axhline(bmean, color='gray', linewidth=1, linestyle='--', zorder=0)
        means = []
        for i in order:
            r, x = recs[i], xs[i]
            vals = r[metric]
            ax.scatter([x] * len(vals), vals, s=14, color='#1f77b4', alpha=0.45, zorder=2)
            m = nanmean(vals)
            if np.isfinite(m):
                means.append((x, m))
            if r['diverged']:
                ax.scatter([x], [bmean * 0.02], marker='x', color='red', s=50, zorder=4)
            if r['sig']:
                ax.scatter([x], [m], marker='*', s=140, color='#d62728', zorder=5)
        if means:
            mx, my = zip(*means)
            ax.plot(mx, my, color='#1f77b4', linewidth=1.8, zorder=3)
        ax.axvline(1 / (1 - g['base']) if gamma else g['base'],
                   color='black', linewidth=0.8, alpha=0.5)
        if g['log'] or gamma:
            ax.set_xscale('log')
        ticks = sorted(set(xs))
        ax.set_xticks(ticks)
        ax.set_xticklabels([('%g' % (1 - 1 / t) if gamma else '%g' % t) for t in ticks],
                           fontsize=7)
        ax.minorticks_off()
        ax.set_title(g['title'], fontsize=9)
        ax.grid(alpha=0.25)
    for ax in axes[n:]:
        ax.axis('off')
    for ax in axes[::cols]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle('SAC HalfCheetah-v5 hyperparameter sensitivity (100k steps, 3 seeds; gray band = '
                 'baseline ± 1 std, star = 3/3 same sign and |diff| > 2x baseline std, x = diverged)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / fname, bbox_inches='tight')
    plt.close(fig)


def plot_tornado(tornado, noise_rel):
    tornado = sorted(tornado, key=lambda t: min(0, t[1]) - max(0, t[2]))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(tornado) + 1.8), dpi=150)
    for i, (k, lo, hi, nsig) in enumerate(tornado):
        ax.barh(i, 100 * min(0, lo), color='#d62728' if nsig else '#e8a0a0')
        ax.barh(i, 100 * max(0, hi), color='#2ca02c' if nsig else '#9fd29f')
        ax.text(100 * min(0, lo) - 1, i, '%+.0f%%' % (100 * lo), va='center', ha='right', fontsize=8)
        ax.text(100 * max(0, hi) + 1, i, '%+.0f%%' % (100 * hi), va='center', ha='left', fontsize=8)
    ax.axvspan(-100 * noise_rel, 100 * noise_rel, color='gray', alpha=0.2,
               label='baseline noise ±%.0f%%' % (100 * noise_rel))
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(tornado)))
    ax.set_yticklabels([t[0] for t in tornado])
    ax.invert_yaxis()
    ax.set_xlabel('Change in late return vs baseline (%): worst / best point in the grid')
    ax.set_title('Sensitivity ranking (deep color = has a starred point)', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / 'sensitivity_tornado.png', bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    base = load_config(None)
    base_late = np.array([base[s]['late'] for s in SEEDS if s in base])
    base_early = np.array([base[s]['early'] for s in SEEDS if s in base])
    base_mean, base_std = base_late.mean(), base_late.std(ddof=1)
    noise_rel = base_std / abs(base_mean)

    # utd4 单独成组只是为了调度（最慢，放最后），画图时并入 utd。
    utd4 = next(g for g in GRID if g['key'] == 'utd4')
    groups = []
    for g in GRID:
        if g['key'] == 'utd4':
            continue
        g = dict(g)
        if g['key'] == 'utd':
            g['points'] = g['points'] + utd4['points']
        groups.append(g)

    records = build_records(groups, base, base_std)
    write_tables(records, base_mean)
    tornado = write_summary(groups, records, base_mean, base_std, noise_rel, len(base_late))
    small_multiples(groups, records, 'late', base_late, 'sensitivity_map.png',
                    'Eval return, mean of 60k/80k/100k')
    small_multiples(groups, records, 'early', base_early, 'sensitivity_early.png',
                    'Eval return, mean of 20k/40k')
    if tornado:
        plot_tornado(tornado, noise_rel)
    print('saved', OUT)


if __name__ == '__main__':
    main()
