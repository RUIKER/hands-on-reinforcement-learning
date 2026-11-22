"""Plot the terminal error and speed of the same seven-axis goals before/after compensation."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path, help='Output of panda_delay_compensation.py')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    data = json.loads(args.report.read_text(encoding='utf8'))
    paired = data['cases']['sac_her_bc']['2']['confirm']
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(9.2, 4.25))
    fig.subplots_adjust(left=.10, right=.98, bottom=.17, top=.71, wspace=.13)
    fig.text(.10, .94, 'Can the Panda stop at the target?', fontsize=19, weight='bold')
    fig.text(.10, .865, '40 ms command delay  |  Same policy weights, home pose and 48 confirmation targets',
             fontsize=10, color='#465663')
    for ax, (key, title, color) in zip(axes, [('before', 'Without prediction', '#B3532B'),
                                            ('after', 'With queued-command prediction', '#007B78')]):
        rows = paired[key]
        success = sum(row['outcome'] == 'success' for row in rows)
        ax.add_patch(Rectangle((0, 0), 20, 30, facecolor='#DCEFE4', edgecolor='none', zorder=0))
        ax.axhline(30, color='#749485', ls='--', lw=.9)
        ax.axvline(20, color='#749485', ls='--', lw=.9)
        ax.scatter([row['final_error_m'] * 1000 for row in rows],
                   [row['final_speed_m_s'] * 1000 for row in rows],
                   s=28, color=color, edgecolors='white', linewidths=.55, alpha=.84, zorder=3)
        ax.set(xlim=(0, 23), ylim=(0, 190), xlabel='Terminal position error (mm)')
        ax.set_title(f'{title}  ·  {success}/{len(rows)}', fontsize=10.5, weight='bold', loc='left')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.grid(axis='y', color='#C4CCD0', alpha=.45, linewidth=.5)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Terminal hand speed (mm/s)')
    fig.text(.10, .035, 'Each dot is one episode. Green region: error < 20 mm and speed < 30 mm/s on the same step.',
             fontsize=9, color='#465663')
    fig.savefig(args.output, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
