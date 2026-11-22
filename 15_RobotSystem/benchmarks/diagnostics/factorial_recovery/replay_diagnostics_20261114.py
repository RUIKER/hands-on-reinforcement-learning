"""Read-only summary of HER/critic sample diagnostics; prints JSON."""
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/factorial_recovery")).resolve()
SEEDS = (1103, 1129, 1181, 1213, 1277)
ARMS = ('sac', 'her', 'her_shape', 'her_smooth', 'her_both')
GRID = np.array([(x,y) for x in (-.16,-.08,.08,.16) for y in (-.16,-.08,.08,.16)])
WINDOWS = (('early',1,10000), ('mid',10001,20000), ('later',20001,30000), ('late',30001,50000))


def mean_or_none(values):
    return float(np.mean(values)) if len(values) else None


def main():
    out = {}
    for arm in ARMS:
        out[arm] = {}
        for label,lo,hi in WINDOWS:
            per_seed = []
            for seed in SEEDS:
                file = ROOT / f'seed-{seed}' / arm / 'updates.jsonl'
                with file.open(encoding='utf-8') as stream:
                    records = [json.loads(line) for line in stream]
                records = [r for r in records if lo <= r['step'] <= hi]
                targets, original_shaping, relabel_shaping = [], [], []
                samples = relabeled = relabel_success = original_success = 0
                for row in records:
                    for goal,flag,base,shape in zip(row['sample_goals'], row['sample_relabeled'], row['sample_base_rewards'], row['sample_shaping']):
                        samples += 1
                        if flag:
                            relabeled += 1
                            targets.append(goal)
                            relabel_shaping.append(shape)
                            relabel_success += base == 0
                        else:
                            original_shaping.append(shape)
                            original_success += base == 0
                target_array = np.asarray(targets)
                nearest = np.min(np.max(np.abs(target_array[:,None,:]-GRID[None,:,:]),axis=2),axis=1) if targets else np.array([])
                per_seed.append({'seed':seed,'log_steps':[r['step'] for r in records],
                    'logged_updates':len(records),'diagnostic_samples':samples,'relabeled_count':relabeled,
                    'relabeled_success':relabel_success,'original_success':original_success,
                    'relabel_within_0_01_any_training_goal':float(np.mean(nearest < .01)) if len(nearest) else None,
                    'relabel_min_linf_to_training_grid_median':float(np.median(nearest)) if len(nearest) else None,
                    'relabel_abs_max_median':float(np.median(np.max(np.abs(target_array),axis=1))) if len(target_array) else None,
                    'shape_term_original_mean':mean_or_none(original_shaping),
                    'shape_term_relabel_mean':mean_or_none(relabel_shaping),
                    'actor_entropy_median':float(np.median([r['actor_entropy_estimate'] for r in records])),
                    'alpha_median':float(np.median([r['alpha'] for r in records])),
                    'critic1_loss_median':float(np.median([r['critic_1_loss'] for r in records])),
                    'critic1_gradient_median':float(np.median([r['gradient_l2']['critic_1'] for r in records])),
                    'actor_gradient_median':float(np.median([r['gradient_l2']['actor'] for r in records]))})
            out[arm][label] = per_seed
    print(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
