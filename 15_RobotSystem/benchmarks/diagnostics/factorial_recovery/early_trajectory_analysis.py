"""Read-only exploratory summary; no validation/test targets read and no training."""
from collections import Counter
from statistics import mean, median
import json
import os
from pathlib import Path
R = Path(os.environ.get("PANDA_LEGACY_RUN", "runs/diagnostics/factorial_recovery")).resolve()
SEEDS = [1103,1129,1181,1213,1277]
ARMS = ['her', 'her_shape', 'her_smooth', 'her_both']
WINDOWS = [(257,10000),(10001,20000),(20001,30000)]

def parse(f):
    with f.open(encoding='utf-8') as s:
        return [json.loads(x) for x in s]

def qmean(values):
    return mean(values) if values else None

out = {'protocol':'Completed actor-only episodes wholly inside each window; equal-weight mean of seed-level statistics. Steps are systematic 250-step snapshots, not random samples or full trajectories.', 'seeds':SEEDS, 'windows':{}}
for lo, hi in WINDOWS:
    stage = {}
    for arm in ARMS:
        details = []
        for seed in SEEDS:
            path = R / f'seed-{seed}' / arm
            episodes = [x for x in parse(path/'episodes.jsonl') if x['step'] <= hi and x['step'] - x['episode_steps'] + 1 >= lo]
            snapshots = [x for x in parse(path/'steps.jsonl') if lo <= x['step'] <= hi and x['step'] % 250 == 0 and x['source']=='actor']
            cnt = Counter(x['outcome'] for x in episodes)
            tv = [x['actor_action_tv']/x['actor_action_pairs'] for x in episodes if x['actor_action_pairs']]
            path_per_step = [x['joint_path_rad']/x['episode_steps'] for x in episodes]
            velocity_per_step = [x['velocity_delta_rad_s']/x['episode_steps'] for x in episodes]
            metrics = {'target_error':[],'command_error':[],'tracking_error':[],'home_distance':[],'progress':[],'speed':[],'action_norm':[]}
            for x in snapshots:
                ob=x['observation']; q=ob[:2]; v=ob[2:4]; g=ob[4:6]; c=ob[6:8]
                err=max(abs(a-b) for a,b in zip(g,q))
                metrics['target_error'].append(err)
                metrics['command_error'].append(max(abs(a-b) for a,b in zip(g,c)))
                metrics['tracking_error'].append(max(abs(a-b) for a,b in zip(c,q)))
                metrics['home_distance'].append(max(map(abs,q)))
                metrics['progress'].append(max(map(abs,g))-err)
                metrics['speed'].append(max(map(abs,v)))
                metrics['action_norm'].append(max(map(abs,x['executed_action'])))
            details.append({'seed':seed,'completed_actor_episodes':len(episodes),'outcomes':dict(cnt),'success_fraction':cnt['success']/len(episodes) if episodes else None,
                            'mean_actor_action_delta':qmean(tv),'mean_joint_path_per_step':qmean(path_per_step), 'mean_speed_delta_per_step':qmean(velocity_per_step),
                            'snapshots':len(snapshots), 'snapshot_means':{k:qmean(v) for k,v in metrics.items()},
                            'snapshot_fraction_max_goal_error_lt_0_03':qmean([v<.03 for v in metrics['target_error']]),
                            'snapshot_fraction_max_home_distance_ge_0_08':qmean([v>=.08 for v in metrics['home_distance']])})
        def avg(k):
            return qmean([x[k] for x in details if x[k] is not None])
        def avg_snap(k):
            return qmean([x['snapshot_means'][k] for x in details if x['snapshot_means'][k] is not None])
        stage[arm]={'per_seed':details, 'seed_mean_success_fraction':avg('success_fraction'), 'total_completed_episodes':sum(x['completed_actor_episodes'] for x in details),
                    'total_successes':sum(x['outcomes'].get('success',0) for x in details),'total_timeouts':sum(x['outcomes'].get('timeout',0) for x in details),'total_failures':sum(x['outcomes'].get('failure',0) for x in details),
                    'seed_mean_actor_action_delta':avg('mean_actor_action_delta'),'seed_mean_joint_path_per_step':avg('mean_joint_path_per_step'),
                    'seed_mean_speed_delta_per_step':avg('mean_speed_delta_per_step'),'seed_mean_snapshot_target_error_rad':avg_snap('target_error'),
                    'seed_mean_snapshot_command_error_rad':avg_snap('command_error'),'seed_mean_snapshot_tracking_error_rad':avg_snap('tracking_error'),
                    'seed_mean_snapshot_progress_rad':avg_snap('progress'), 'seed_mean_snapshot_action_max':avg_snap('action_norm'),
                    'seed_mean_snapshot_near_goal_fraction':avg('snapshot_fraction_max_goal_error_lt_0_03'),
                    'seed_mean_snapshot_far_from_home_fraction':avg('snapshot_fraction_max_home_distance_ge_0_08')}
    out['windows'][f'{lo}-{hi}']=stage
with (R/'early_trajectory_analysis.json').open('x',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False,allow_nan=False)
for w,arms in out['windows'].items():
 print('\nWINDOW',w)
 for name,entry in arms.items():
  print(name,'success',round(entry['seed_mean_success_fraction']*100,2),'n',entry['total_successes'],entry['total_completed_episodes'],'TV',round(entry['seed_mean_actor_action_delta'],4),'path/step',round(entry['seed_mean_joint_path_per_step'],5),'goalerr',round(entry['seed_mean_snapshot_target_error_rad'],4),'progress',round(entry['seed_mean_snapshot_progress_rad'],4),'trackerr',round(entry['seed_mean_snapshot_tracking_error_rad'],4),'near',round(entry['seed_mean_snapshot_near_goal_fraction']*100,1),'far',round(entry['seed_mean_snapshot_far_from_home_fraction']*100,1))
