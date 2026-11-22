"""Exploratory fixed-interval snapshots of raw versus executed behavior actions.

No validation/test data is read. Direction alignment is a heuristic, not a
measure of optimal actions: moving away can legitimately brake or prevent overshoot.
"""
import os
from pathlib import Path
from statistics import mean
import json
import math
R=Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/factorial_recovery")).resolve()
SEEDS=[1103,1129,1181,1213,1277]
ARMS=['her_shape','her_both']
WINDOWS=[(257,10000),(10001,20000),(20001,30000)]
output={'description':'Systematic every-250-step actor snapshots; alignment counts axes with abs(goal−ctrl)>0.01. Sign mismatch counts axes with abs(raw)>0.01 and abs(executed)>0.01; equal seed weighting. Heuristic only.', 'windows':{}}
for lo,hi in WINDOWS:
    window={}
    for arm in ARMS:
        data=[]
        for seed in SEEDS:
            with (R/f'seed-{seed}'/arm/'steps.jsonl').open(encoding='utf-8') as source:
                rows=[json.loads(line) for line in source]
            rows=[r for r in rows if lo<=r['step']<=hi and r['step']%250==0 and r['source']=='actor']
            raw_norm=[math.hypot(*r['raw_action']) for r in rows]
            exec_norm=[math.hypot(*r['executed_action']) for r in rows]
            align,align_raw,opposing=[],[],[]
            for r in rows:
                for axis in (0,1):
                    err=r['observation'][4+axis]-r['observation'][6+axis]
                    if abs(err)>.01:
                        align.append(err*r['executed_action'][axis]>0)
                        align_raw.append(err*r['raw_action'][axis]>0)
                    if abs(r['raw_action'][axis])>.01 and abs(r['executed_action'][axis])>.01:
                        opposing.append(r['raw_action'][axis]*r['executed_action'][axis]<0)
            data.append({'seed':seed,'snapshots':len(rows),'raw_norm_mean':mean(raw_norm),'executed_norm_mean':mean(exec_norm),
                         'alignment_executed':mean(align),'alignment_raw':mean(align_raw),'raw_executed_opposite_fraction':mean(opposing) if opposing else None})
        keys=['raw_norm_mean','executed_norm_mean','alignment_executed','alignment_raw','raw_executed_opposite_fraction']
        window[arm]={'per_seed':data,'seed_means':{k:mean([x[k] for x in data if x[k] is not None]) for k in keys}}
    output['windows'][f'{lo}-{hi}']=window
with (R/'action_alignment_analysis.json').open('x',encoding='utf-8') as f: json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({w:{a:v['seed_means'] for a,v in x.items()} for w,x in output['windows'].items()},indent=2))
