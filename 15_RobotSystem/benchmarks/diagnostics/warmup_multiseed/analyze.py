"""Read-only verification and paired analysis for the preregistered experiment."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import sqrt
import os
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/panda-warmup-multiseed")).resolve()


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def early(path):
    sources = defaultdict(set)
    result = Counter()
    for row in lines(path):
        if row['step'] > 10000:
            break
        sources[row['episode']].add(row['source'])
        if row['outcome'] != 'running' and sources[row['episode']] == {'actor'}:
            result[row['outcome']] += 1
    return dict(result)


def summarize(root):
    protocol = load(root / 'protocol.json')
    statuses = list(lines(root / 'status.jsonl'))
    require(statuses[0]['stage'] == 'protocol_frozen' and
            statuses[0]['sha256'] == digest(root / 'protocol.json'), 'protocol hash mismatch')
    require(statuses[-1]['stage'] == 'job_complete', 'job incomplete')
    require(not any(s['stage'] == 'job_failed' or s.get('exit_code', 0) != 0 for s in statuses),
            'job failure recorded')
    require(protocol['seeds'] == [251, 379, 487, 593, 719] and len(protocol['cases']) == 48,
            'unexpected protocol')
    results = []
    for seed in protocol['seeds']:
        run = {'seed': seed}
        for arm in ('random', 'demo'):
            base = root / f'seed-{seed}' / arm
            conf = load(base / 'train/config.json')
            train = load(base / 'train/summary.json')
            evconf = load(base / 'evaluation/config.json')
            ev = load(base / 'evaluation/summary.json')
            require(conf['seed'] == seed and conf['steps'] == 50000 and conf['her']['enabled'], 'training config')
            require(conf['warmup_policy']['demonstration_steps'] == (152 if arm == 'demo' else 0), 'wrong warmup')
            require(train['status'] == 'PASS' and train['steps'] == 50000 and
                    train['reload_action_max_abs_difference'] == 0, 'training verification')
            require(evconf['cases'] == protocol['cases'] and evconf['delay_steps'] == 0 and
                    evconf['checkpoint_sha256'] == digest(base / 'train/sac_cpu.pt'), 'evaluation config')
            for controller in ('sac', 'position_target'):
                block = ev[controller]
                episodes = block['per_episode']
                require(block['episodes'] == 48 and len(episodes) == 48, 'episode count')
                require([[ep['seed'], ep['goal_rad']] for ep in episodes] == protocol['cases'], 'case mismatch')
                require(block['successes'] == sum(ep['outcome'] == 'success' for ep in episodes), 'success count')
            episodes = ev['sac']['per_episode']
            by_quad = {}
            for label, (sx, sy) in {'++': (1, 1), '-+': (-1, 1), '--': (-1, -1), '+-': (1, -1)}.items():
                subset = [ep for ep in episodes if ep['goal_rad'][0]*sx > 0 and ep['goal_rad'][1]*sy > 0]
                by_quad[label] = {'successes': sum(e['outcome'] == 'success' for e in subset), 'episodes': len(subset)}
            run[arm] = {'successes': ev['sac']['successes'], 'episodes': 48,
                        'outcomes': dict(Counter(e['outcome'] for e in episodes)),
                        'quadrants': by_quad, 'mean_success_steps': ev['sac']['mean_success_steps'],
                        'position_baseline_successes': ev['position_target']['successes'],
                        'early_actor_outcomes': early(base / 'train/steps.jsonl'),
                        'training_wall_seconds': train['rollout_update_s'],
                        'warnings': sum(bool(e.get('physics_warning') or e.get('actual_limit_violation')) for e in episodes)}
        run['difference_percentage_points'] = (run['demo']['successes'] - run['random']['successes']) * 100 / 48
        results.append(run)
    diffs = [r['difference_percentage_points'] for r in results]
    avg = mean(diffs)
    margin = 2.776 * stdev(diffs) / sqrt(5)
    ci = [avg - margin, avg + margin]
    criteria = {'mean_at_least_10pp': avg >= 10,
                'at_least_4_positive_seeds': sum(d > 0 for d in diffs) >= 4,
                'no_seed_below_minus_10pp': min(diffs) >= -10,
                'ci_lower_above_zero': ci[0] > 0}
    return {'runs': results, 'mean_difference_pp': avg, 'paired_t_95_ci_pp': ci,
            'criteria': criteria, 'all_criteria_met': all(criteria.values()),
            'zero_variance_warning': margin == 0}


if __name__ == '__main__':
    print(json.dumps(summarize(ROOT), ensure_ascii=False, indent=2, allow_nan=False))
