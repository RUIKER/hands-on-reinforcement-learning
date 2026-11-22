"""Rebuild exact cumulative HER sample attempt counts by seed and window."""
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/factorial_recovery")).resolve()
SEEDS = (1103,1129,1181,1213,1277)
ARMS = ('her','her_shape','her_smooth','her_both')
FIELDS = ('attempts','relabeled','relabeled_success','no_future_goal','already_at_goal','source_failed','original')
WINDOWS = ((0,10000),(10000,20000),(20000,30000))


def main():
    data = {}
    for arm in ARMS:
        data[arm] = {}
        for start,end in WINDOWS:
            output = []
            for seed in SEEDS:
                path=ROOT / f'seed-{seed}' / arm / 'updates.jsonl'
                with path.open(encoding='utf-8') as f:
                    rows=[json.loads(line) for line in f]
                endrow=next(r['her_total'] for r in rows if r['step']==end)
                startrow=next((r['her_total'] for r in rows if r['step']==start),{})
                entry={'seed':seed, **{k:endrow.get(k,0)-startrow.get(k,0) for k in FIELDS}}
                assert entry['attempts'] == entry['relabeled']+entry['no_future_goal']+entry['already_at_goal']+entry['source_failed']
                output.append(entry)
            data[arm][f'{start}-{end}']=output
    print(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
