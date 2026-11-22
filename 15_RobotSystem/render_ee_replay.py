"""Render an offline comparison from the saved seven-joint evaluation."""
import argparse
import json
from pathlib import Path

import mujoco
from render_panda_replay import render_frames, webp_data


LABELS = {
    'ik': 'IK 基线',
    'sac': '原始 SAC',
    'sac_her': '原始 SAC + HER',
    'cartesian_feedback': '笛卡尔反馈控制',
    'bc': '仅行为克隆',
    'sac_her_bc': '改进 SAC + HER',
}


def read(directory, name):
    return json.loads((directory / f'{name}.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', type=Path)
    parser.add_argument('results', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', default='test', choices=('test', 'frozen', 'fresh'))
    parser.add_argument('--legacy', type=Path, help='Include the historical baseline on the frozen targets')
    args = parser.parse_args()
    goals = read(args.results, 'goals')[args.split]
    results = read(args.results, 'evaluation')
    traces = read(args.results, 'trajectories')
    if args.split != 'test':
        results = {name: results[name][args.split] for name in ('sac_her_bc', 'bc', 'cartesian_feedback')}
        traces = {name: traces[name][args.split] for name in results}
    if args.legacy:
        if args.split != 'frozen' or goals != read(args.legacy, 'goals')['test']:
            parser.error('--legacy requires matching frozen goals')
        results.update(read(args.legacy, 'evaluation'))
        traces.update(read(args.legacy, 'trajectories'))
    # Pick a case that shows the most varied outcomes, with stable tie breaking.
    case = max(range(len(goals)), key=lambda i: len({results[k][i]['outcome'] for k in results}))
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    addresses = [model.jnt_qposadr[model.joint(f'joint{i}').id] for i in range(1, 8)]
    clips = []
    for name in results:
        trace = traces[name][case]
        states = []
        for row in trace:
            q = model.key('home').qpos.copy()
            q[addresses] = row['q']
            states.append(q)
        frames = [[i, webp_data(im)] for i, im in render_frames(model, states, 5)]
        clips.append({'name': LABELS[name], 'frames': frames, 'trace': trace, 'result': results[name][case]})
    data = json.dumps({'goal': goals[case], 'case': case + 1, 'goal_count': len(goals),
                       'seed': read(args.results, 'protocol')['training_seed'], 'clips': clips},
                      ensure_ascii=False, separators=(',', ':'))
    html = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Panda 七轴末端到达</title><style>
body{background:#101a20;color:#e9f3f2;font:16px system-ui,Microsoft YaHei;margin:0}main{max-width:1100px;margin:auto;padding:24px}h1{margin:0;font-size:32px}p{color:#b1c6c8}nav{display:flex;gap:8px;flex-wrap:wrap}button{font:inherit;background:#233740;color:#fff;border:1px solid #698086;border-radius:6px;padding:8px 14px;cursor:pointer}button[aria-pressed=true]{background:#76e2bf;color:#10231d}.grid{display:grid;grid-template-columns:1.2fr 1fr;gap:18px;margin-top:16px}.panel{background:#1b2a32;border:1px solid #3d555c;border-radius:8px;padding:12px}img{display:block;width:100%;aspect-ratio:8/5;object-fit:contain}canvas{width:100%;height:auto;aspect-ratio:5/3;background:#122128}input{width:100%;accent-color:#76e2bf}.line{display:flex;align-items:center;gap:14px}.line input{flex:1}strong{color:#76e2bf}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><main><h1>Panda 七轴末端到达</h1><p>同一测试目标 · 固定 home 起点 · 实际仿真轨迹</p><nav id="tabs"></nav><div class="grid"><div class="panel"><img id="pose" alt="Panda 机械臂仿真姿态"><div id="status"></div></div><div class="panel"><strong>末端轨迹 XY 俯视</strong><canvas id="chart" width="500" height="300"></canvas><p>绿色为末端路径，紫色圆点为目标，白色圆点为当前末端。Z 高度和三维误差见下方。</p><div id="coords"></div></div></div><div class="line"><button id="play">播放</button><input id="time" type="range" min="0" value="0"><span id="clock"></span></div><p id="info"></p></main>
<script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent), get=id=>document.getElementById(id);let selected=0,index=0,timer=null;const names=data.clips.map(clip=>clip.name);
function select(n){selected=n;index=0;clearInterval(timer);timer=null;get('play').textContent='播放';for(const [i,b] of [...get('tabs').children].entries())b.setAttribute('aria-pressed',i===n);get('time').max=data.clips[n].trace.length-1;update()}
names.forEach((name,i)=>{const b=document.createElement('button');b.textContent=name;b.onclick=()=>select(i);get('tabs').append(b)});
function update(){const c=data.clips[selected],t=c.trace[index],goal=data.goal;get('time').value=index;let frame=c.frames[0];for(const f of c.frames){if(f[0]<=index)frame=f;else break}get('pose').src=frame[1];const e=Math.hypot(...t.xyz.map((x,i)=>x-goal[i]));get('clock').textContent=t.time.toFixed(2)+' s';get('status').textContent=names[selected]+' · '+(c.result.outcome==='success'?'成功':c.result.outcome==='timeout'?'超时':'仿真失败')+' · '+c.result.steps+' 步';get('coords').textContent='当前 z '+t.xyz[2].toFixed(3)+' m · 目标 z '+goal[2].toFixed(3)+' m · 三维误差 '+(e*100).toFixed(1)+' cm';get('info').textContent='测试目标 '+data.case+' / '+data.goal_count+' · 成功条件为末端距离 < 2 cm 且线速度 < 3 cm/s · 训练 seed '+data.seed;const cv=get('chart'),ctx=cv.getContext('2d'),pts=data.clips.flatMap(c=>c.trace.map(t=>t.xyz)).concat([goal]),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),xmin=Math.min(...xs)-.02,xmax=Math.max(...xs)+.02,ymin=Math.min(...ys)-.02,ymax=Math.max(...ys)+.02,scale=Math.min(430/(xmax-xmin),230/(ymax-ymin));const xy=p=>[250+(p[0]-(xmin+xmax)/2)*scale,150-(p[1]-(ymin+ymax)/2)*scale];ctx.clearRect(0,0,500,300);ctx.strokeStyle='#78e5c2';ctx.lineWidth=3;ctx.beginPath();c.trace.slice(0,index+1).forEach((r,i)=>{const [x,y]=xy(r.xyz);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();for(const [p,col] of [[goal,'#c0aaff'],[t.xyz,'#fff']]){const [x,y]=xy(p);ctx.beginPath();ctx.fillStyle=col;ctx.arc(x,y,7,0,7);ctx.fill()}ctx.fillStyle='#b1c6c8';ctx.font='14px sans-serif';ctx.fillText('x →',445,290);ctx.fillText('y ↑',5,18)}
get('time').oninput=e=>{index=+e.target.value;update()};get('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;get('play').textContent='播放';return}if(index>=data.clips[selected].trace.length-1)index=0;get('play').textContent='暂停';timer=setInterval(()=>{index=Math.min(index+1,data.clips[selected].trace.length-1);update();if(index===data.clips[selected].trace.length-1){clearInterval(timer);timer=null;get('play').textContent='播放'}},40)};select(0)
</script></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html.replace('__DATA__', data), encoding='utf-8')
    print('case', case + 1, 'output', args.output)


if __name__ == '__main__':
    main()
