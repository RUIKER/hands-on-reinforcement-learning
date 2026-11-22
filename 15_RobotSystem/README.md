# Panda 机械臂仿真与策略验证

## 七轴末端位置到达

七轴任务见 [环境与训练代码](panda_ee_reach.py)、[面试问答](Panda七轴末端到达_面试问答.md) 和 [末端轨迹回放生成器](render_ee_replay.py)。任务在 `link0` 坐标系中控制 `hand` 原点的三维位置。每 20 ms 输出七个关节位置目标增量，每轴最多 0.02 rad。观测包括七轴位置、速度、伺服目标，以及末端位置、末端线速度和目标位置，总共 30 维。距离小于 0.02 m 且末端速度小于 0.03 m/s 为成功。仿真失败终止，250 步为时间截断。

固定目标生成种子 61122，采样 home 附近的三维偏移并用位置 IK 检查收敛，前 80 个用于训练，后 24 个留出测试。SAC 与 SAC + HER 共用训练种子、目标集合和网络配置。HER 从同一回合当前或未来有效末端位置采样，并重新计算目标、奖励和终止。对照基线使用阻尼最小二乘 IK 计算关节角，再以相同增量限制执行。实验记录成功、超时和异常，不推断随机初态、障碍或实机表现。

使用上游固定模型和装有 MuJoCo、NumPy、PyTorch 的 Python，训练两个单种子策略并评估，输出目录必须不存在。

```bash
python 15_RobotSystem/panda_ee_reach.py "$PANDA_SCENE" runs/panda-ee-seed43 --steps 50000 --seed 43
```

目录中 `goals.json` 是冻结目标，`sac/actor.npz` 与 `sac_her/actor.npz` 是确定性评估所用权重。运行下面的命令会用这两份权重重新生成目录中的评估和逐步轨迹。

```bash
python 15_RobotSystem/panda_ee_reach.py "$PANDA_SCENE" runs/panda-ee-seed43 --evaluate-only
```

回放还需 Pillow 和可用的 MuJoCo 渲染后端。

```bash
python 15_RobotSystem/render_ee_replay.py "$PANDA_SCENE" runs/panda-ee-seed43 runs/panda-ee-seed43/replay.html
```

回放自动选择三种方法表现最有差异的同一测试目标。曲线为真实记录的末端 XY 投影，Z 和三维误差单独显示。详细评估口径和结果随实验记录保存。

### 2026-11-22 单种子首次对照

在 Windows CPU、MuJoCo 3.17.0、PyTorch 2.10.0+cpu 上，以 seed 43 各训练 50,000 步。两组耗时分别约 188 秒与 214 秒，均完成 49,744 次更新。保存的 [协议、24 个目标、权重与逐步轨迹](results/20261122_seven_axis_reach/) 可使用 `--evaluate-only` 重评估。冻结目标按 IK 收敛筛选，起点均为 home。

| 方法 | 成功 | 超时 | 仿真失败 | 最终距离中位数 |
| --- | ---: | ---: | ---: | ---: |
| IK 位置控制 | 24/24 | 0/24 | 0/24 | 0.77 cm |
| SAC | 0/24 | 0/24 | 24/24 | 119.77 cm |
| SAC + HER | 0/24 | 24/24 | 0/24 | 47.03 cm |

复查显示 SAC 24 例均发生第二关节超过模型限位及 0.001 rad 容差，未出现物理告警。HER 训练共 1,585,800 次有效重标记，其中 4,000 次为虚拟成功；这个计数以训练采样次数为单位，可能重复采样同一转移。结果说明首个单种子策略尚未完成七轴到达任务，训练样本、目标分布及固定初态限制了结论范围。独立目标结果及 [同一目标的离线回放](demo/panda-ee-replay.html) 如实呈现失败与超时。

用 MuJoCo 原生 Python API 加载 `scene.xml`，从 `home` 关键帧重置并验证单关节指令能改变模拟状态。模型来自 [MuJoCo Menagerie 固定提交](https://github.com/google-deepmind/mujoco_menagerie/tree/98a70c569c4c1b6da467a20012f766ee1f8dc521/franka_emika_panda)，模型的 Apache-2.0 许可证随上游目录保留。

轨迹误差、位置基线和超时案例见 [仿真验证笔记](经验总结.md)。

11 月 9 日分终端练习见 [Panda 固定轨迹手动复现](Panda轨迹手动复现.md)。

## 展示与复现入口

- [Panda 轨迹预览](demo/panda-tracking.webp)展示 ROS 2 七轴多航点的实测仿真运动。
- [下载离线动作回放](https://raw.githubusercontent.com/RUIKER/hands-on-reinforcement-learning/main/15_RobotSystem/demo/panda-replay.html)，用浏览器打开。可切换七轴多航点、两轴固定跟踪、用例 30104 成功和用例 30124 超时，播放并拖动到指定仿真时刻。
- [七轴采样](results/20261117_fixed_tracking/tracking.csv)、[冻结目标与闭环验收](results/20261119_panda_policy/)、[SAC/HER 配对结果](results/20261112_paired/)和 [MCAP 录包](results/20261119_rosbag/)可作为对照证据。
- [历史实验与诊断](benchmarks/diagnostics/README.md)包含 warm-up、精度和恢复对照的脚本与结果。
- [WSL 预检](wsl_preflight/README.md)记录仿真、策略和依赖的版本，以及已有短测。

### 从新环境复现

从强化学习仓库根目录执行。Panda 使用固定 Menagerie 提交，离线示例验证版本为 MuJoCo 3.17.0 和 NumPy 2.5.4。

~~~bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
git -C mujoco_menagerie checkout --detach 98a70c569c4c1b6da467a20012f766ee1f8dc521
python -m pip install "mujoco==3.17.0" "numpy==2.5.4"
export PANDA_SCENE="$PWD/mujoco_menagerie/franka_emika_panda/scene.xml"
export PANDA_PYTHON="$(command -v python)"
"$PANDA_PYTHON" -I 15_RobotSystem/panda_smoke.py "$PANDA_SCENE"
"$PANDA_PYTHON" 15_RobotSystem/panda_her_smoke.py
"$PANDA_PYTHON" 15_RobotSystem/panda_fixed_position_baseline.py "$PANDA_SCENE" runs/fixed-tracking
~~~

位置跟踪的输出目录必须尚不存在。Windows PowerShell 可在设置 $env:PANDA_SCENE 后用同样的 Python 参数运行。重建离线回放时还需要 Pillow 与可用的离屏渲染后端。

~~~bash
python -m pip install Pillow
python 15_RobotSystem/render_panda_replay.py "$PANDA_SCENE" runs/panda-replay.html
~~~

WSL ROS 2 双进程检查接受 ROS 环境下能同时导入 rclpy、MuJoCo 和 NumPy 的 Python。脚本在新目录构建消息接口及节点，加载随仓库发布的 NumPy actor，然后运行同一成功和超时用例。

~~~bash
bash 15_RobotSystem/run_ros2_checks.sh "$PANDA_SCENE" runs/ros2-repro /path/to/ros-venv/bin/python
~~~

下方保存各阶段的实验记录。历史输出使用各阶段的参数与依赖，未随仓库归档的原始训练产物使用本机归档标记。

## 额外练习：关节阻尼受控对照

`panda_damping_compare.py` 加载同一固定模型，以 `home` 关键帧重置。基线使用原始 `joint1` 关节阻尼，对照组只将该轴的 `dof_damping` 乘以指定倍数。两组其余模型参数、初始状态、执行器控制和固定输入一致。脚本直接使用 MuJoCo 推进仿真，命令轨迹复用单轴示例的目标值和线性插值规则。

无随机采样。每组分别从同一关键帧运行两次，每次从 0 rad 目标在 0.4 s 内线性到 0.1 rad，再保持到 0.8 s。物理步长 0.002 s。每步先设置位置目标，再推进一步；记录该步开始的控制时间、实际施加的目标与推进后的状态时间和位置。跟踪 RMSE 比较每步末位置和该步施加目标。

在 WSL 中用已有模型和 Python 环境运行。输出目录必须为新目录。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
scene="$PANDA_SCENE"
panda_python="$PANDA_PYTHON"
out="$PWD/runs/damping-$(date +%Y%m%d-%H%M%S)"
"$panda_python" "$project/panda_damping_compare.py" "$scene" "$out" --damping-factor 2
```

输出 `trajectory.csv` 和 `summary.json`。同组两次位置差用于检查固定条件下的重复性。对照组第一次减基线第一次的位置差用于观察阻尼变化下的响应差异。脚本同时报告整段跟踪 RMSE、末帧位置和四个时间点的样本。这个离线练习不包含 ROS2 通信或传感器噪声。

Agent 在 WSL 2、MuJoCo 3.17.0、NumPy 2.5.4 上用 `--damping-factor 1` 检查对照两侧完全一致，再用 `--damping-factor 2` 检查阻尼变化产生可观察差异。本人于 2026-11-08 在相同版本下运行双倍阻尼对照，原始数据、摘要与图分别保存在 [轨迹数据](results/20261108_joint1_damping/trajectory.csv)、[摘要](results/20261108_joint1_damping/summary.json) 和 [对照图](results/20261108_joint1_damping/comparison.png)。

使用带 NumPy 和 Matplotlib 的 Python，从某次实验目录生成图。

```bash
python "$project/panda_damping_plot.py" "$out"
```

输出 `comparison.png` 包括关节位置、两组位置差、速度、阻尼力矩分量、跟踪误差和组内重执行差异。阻尼力矩分量按采样时的 `−dof_damping × qvel` 重建，单位 N·m；执行器力矩没有包含在该量内。图中目标值是在物理步开始时施加的控制目标，状态值取该步结束时的采样。WSL 现有 MuJoCo venv 是否安装 Matplotlib 需单独确认；本次绘图使用 Windows 的独立临时 Python 依赖。

## Panda 正运动学检查

`panda_fk.py` 以模型的 `link0` 为参考坐标系，计算 `hand` 坐标系的位姿。末端选择的是模型中 `hand` 的原点，夹爪指尖或工具中心需要额外的坐标变换。

脚本按名称找到 `link0` 到 `hand` 的连杆与七个关节，从编译后的模型读取每段固定位置、固定旋转、局部关节轴及关节位置，再逐段相乘。计算过程独立组合变换，模型参数来自同一份 MuJoCo 模型。对照值来自 `mj_forward` 后的 body 位姿，两者都转换到 `link0` 坐标系。

在 WSL 中使用已有模型和 MuJoCo 环境运行。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
scene="$PANDA_SCENE"
panda_python="$PANDA_PYTHON"
"$panda_python" "$project/panda_fk.py" "$scene"
"$panda_python" "$project/panda_fk.py" "$scene" \
  --joints 0.1 -0.1 0 -1.57079 0 1.57079 -0.7853
```

默认固定输入共九组，不涉及随机采样。包括 `home`、第一轴加 0.1 rad 且第二轴减 0.1 rad，以及每次仅一轴相对 `home` 增加 0.05 rad 的七组。脚本输出两种位置和 3×3 旋转矩阵，比较位置欧氏误差和旋转矩阵逐元素最大绝对差，二者各自超过 `1e-9` 时退出失败。参数 `--joints` 接收七个有限关节角，单位为 rad。

2026-11-08 在 WSL 2、MuJoCo 3.17.0、NumPy 2.5.4 上完成上述九组 agent 检查，最大位置差为 `4.710e-16 m`，最大旋转矩阵元素差为 `2.665e-15`。对照检查反映同模型下的数值一致性，精度受到模型及输入定义影响。

## Panda 两轴到达任务

`panda_two_joint_env.py` 提供原生 Python 的 `reset()` 和 `step(action)`。`reset()` 恢复 `home`，默认任务目标为 `(0.1, 0.1)` rad。观测依次是 `q1, q2, dq1, dq2, goal1, goal2, ctrl1, ctrl2`。动作是两轴位置命令增量，每轴限于 ±0.02 rad，累加后的命令限制在模型关节和执行器允许范围的交集。其他执行器保留 home 命令。每个动作推进 10 个 2 ms 物理步。

两个受控关节的位置误差都小于 0.01 rad 且速度绝对值都小于 0.05 rad/s 时成功，奖励为 0。其他步奖励为 −1。无效状态、仿真警告或受控关节超出模型限位及 0.001 rad 容差时终止为失败。默认 250 步无成功或失败时记为超时，返回 `truncated=True`。这些是固定场景 smoke 的初始参数。

在 WSL 中沿用上文的 `project`、`scene` 和 `panda_python`，先预测命令目标再运行。

```bash
"$panda_python" "$project/panda_two_joint_env.py" "$scene" \
  --action 0.02 -0.01 --action 0.02 0.02
```

每行是一次 `reset` 或 `step` 的 JSON。两步执行器目标预期分别为 `(0.02, -0.01)` 和 `(0.04, 0.01)` rad。可用 `--goal 0.1 0.05` 改变任务目标后复跑。

2026-11-08，agent 在固定模型、MuJoCo 3.17.0、NumPy 2.5.4 下验证复位与重复执行、命令及增量限幅、非法动作、超时和注入限位异常。成功分支用固定目标，先连续五步施加 `(0.02, 0.02)` rad，再施加零增量直至成功，于第 18 步通过。全部短检查无随机采样。本人于 2026-11-09 在 WSL 完成两步固定动作复现。

<details>
<summary>两步手动复现参考值</summary>

2026-11-09，agent 用上面的命令在 WSL 2 和 Windows 上复跑固定输入。两边均使用 MuJoCo 3.17.0、NumPy 2.5.4 和页首固定版本的模型，无随机采样。下表保留六位小数。

| 阶段 | 实际角度 q（rad） | 实际角速度 dq（rad/s） | 执行器目标 ctrl（rad） | 奖励 |
| --- | --- | --- | --- | --- |
| reset | `(0, 0)` | `(0, 0)` | `(0, 0)` | — |
| 第一步 | `(0.003600, -0.000598)` | `(0.166862, -0.029388)` | `(0.02, -0.01)` | `-1` |
| 第二步 | `(0.010277, 0.001637)` | `(0.307635, 0.142928)` | `(0.04, 0.01)` | `-1` |

三行观测的任务目标均为 `(0.1, 0.1)` rad，两步均为 `running`，`terminated` 与 `truncated` 都为 `false`。在 Windows 上加入 `--goal 0.1 0.05` 并保持两步动作不变，三行中只有观测的 `goal2` 从 `0.1` 变为 `0.05`。任务目标参与到达判定，固定动作通过 `ctrl` 驱动仿真。本人在 WSL 手算并复现了两组目标下的两步输出。

</details>

## Panda 两轴 SAC CPU 检查

`panda_sac_smoke.py` 复用 `11_SAC/Pendulum` 的 SAC 和回放池。策略和回放使用归一化动作 `[-1, 1]²`，送入环境时乘以 0.02，转换为关节目标增量。回合在终止或超时后重置。回放池只把 `terminated` 作为停止 bootstrap 的标记。

11 月 10 日的固定动作练习、源码追踪及正常步/超时步对照见 [SAC 单条样本追踪](SAC单条样本追踪.md)。本人于 11 月 11 日复跑 CPU 短检查，结果与边界见 [经验总结](经验总结.md)；独立口述数据流和目标网络仍待复测。

从项目根目录在 Windows PowerShell 运行。使用 Python 3.12 和隔离的 CPU 环境。已有环境可跳过安装。

```powershell
$cpuEnv = '.venv'
python -m venv $cpuEnv
$pandaPython = Join-Path $cpuEnv 'Scripts\python.exe'
& $pandaPython -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.10.0+cpu'
& $pandaPython -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
$scene = Join-Path $env:LOCALAPPDATA 'EmbodiedAI\PandaSmoke\menagerie-98a70c5\franka_emika_panda\scene.xml'
$out = Join-Path (Split-Path $cpuEnv) ('panda-sac-cpu-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& $pandaPython 15_RobotSystem/panda_sac_smoke.py $scene $out
```

协议固定 seed 42、64 条采样、32 条 batch、一次参数更新。测试将单回合临时限为 16 步，用于覆盖超时重置。输出 `summary.json` 和 `sac_cpu.pt`，其中 checkpoint 包含网络、优化器、温度参数和配置。以相同状态和随机种子对照重载前后的动作。该流程验收接线、更新和保存重载，策略成功率另行评估。

2026-11-08 在 Windows CPU、PyTorch 2.10.0+cpu、MuJoCo 3.17.0、NumPy 2.5.4 上通过。Actor 和两个 Critic 的最大参数变化均约 `3.0e-4`，重载动作最大绝对差为 `0`。

## W3 两轴控制基线和 SAC 训练评估

`panda_reach.py` 的 `train` 和 `evaluate` 共用上述两轴环境。从 home 重置，每条动作推进 20 ms，增量为归一化动作乘 0.02 rad，写入的位置命令再按模型限位裁剪。成功要求两轴位置误差都严格小于 0.01 rad 且绝对速度都严格小于 0.05 rad/s，超过 250 步预算即超时。成功步奖励为 0，其余步为 −1。回放只用 `terminated` 关闭 bootstrap，超时用截断前的下一状态。

| 用途 | 固定目标和随机种子 |
| --- | --- |
| 训练 | 两轴各在 `{-0.16, -0.08, 0.08, 0.16}` rad 取值，共 16 个目标。每回合按 seed 42 均匀抽一个目标。PyTorch 初始化和回放抽样用 seed 42，随机预热动作用 seed 43。 |
| 调试 | seed 101/102，目标依次为 `(0.10, 0.10)`、`(-0.10, 0.10)` rad。 |
| 测试 | seed 201–204，目标依次为 `(0.12, 0.14)`、`(-0.14, 0.12)`、`(0.14, -0.12)`、`(-0.12, -0.14)` rad。 |

评估中只有随机基线使用每个用例的 seed。位置基线每步将 `goal - ctrl` 限幅到 ±0.02 rad，底层仍使用 Panda 的位置执行器。保持 home 发送零增量。SAC 使用 `tanh(mean)` 的确定性动作。可选延迟用队列将增量推迟两个或更多控制步执行，回合开始以零增量填充；观测仍为当前状态。名义结果与延迟结果分别保存。

每次训练保存配置、逐回合日志、checkpoint 和耗时摘要。前 256 步随机预热，之后每个环境步抽 64 条样本更新一次。checkpoint 包含网络、优化器、温度、训练配置和随机状态，可用于评估。精确断点续训还需要回放池和环境状态。评估保存配置、checkpoint 校验和、逐回合摘要和各控制器的逐步 JSONL，可查发出的动作、排队后请求的增量、实际应用的增量、当前和下一状态、命令裁剪及异常。

从仓库根目录用 Windows PowerShell 运行。使用上一节 CPU 环境和同一固定版本的 Panda 模型。每个输出路径需为新目录或新文件。

```powershell
$py = (Get-Command python).Source
$scene = Join-Path $env:LOCALAPPDATA 'EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
$project = '15_RobotSystem'
$run = Join-Path "runs" ("panda-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
& $py "$project/panda_reach.py" train $scene "$run/train" --steps 2048 --seed 42
& $py "$project/panda_reach.py" evaluate $scene "$run/nominal" --split test --checkpoint "$run/train/sac_cpu.pt"
& $py "$project/panda_reach.py" evaluate $scene "$run/delayed" --split test --delay-steps 2 --checkpoint "$run/train/sac_cpu.pt"
& $py "$project/panda_ik.py" $scene --smoke --output "$run/ik.json"
# 使用已安装 Matplotlib 的 Python 从逐步日志绘图
python "$project/panda_reach_plot.py" "$run/nominal/position_target.jsonl" "$run/delayed/position_target.jsonl" "$run/position_target_204.png" --seed 204
```

2026-11-12 Agent 在 Windows、i5-14600KF、PyTorch 2.10.0+cpu（单线程）、MuJoCo 3.17.0 和 NumPy 2.5.4 上执行。短检查采集 2048 步并更新 1792 次，重载前后参考状态的确定性动作完全相同。采样与更新循环耗时 5.29 s，有更新阶段约 343 环境步/s。训练出现八个超时和一个因预算结束的未完回合。固定测试集结果如下。

| 控制器 | 名义成功数 | 40 ms 动作延迟成功数 |
| --- | ---: | ---: |
| 位置目标基线 | 4/4（每例 20 步） | 0/4 |
| 保持 home | 0/4 | 0/4 |
| 随机 | 0/4 | 0/4 |
| SAC 短检查 checkpoint | 0/4 | 0/4 |

延迟下四例均超时。位置基线按当前已执行的 `ctrl` 计算增量，未计入在途命令。seed 204 的日志显示，位置命令越过目标后继续来回调整，末步两轴速度绝对值仍约 0.377 和 0.262 rad/s，超过成功阈值。该固定样本用于诊断，调参使用调试集。SAC 当前通过更新、保存和重载检查，测试目标尚未到达。结果目录为 `<本机路径>`。

### 2026-11-12 单 seed 50,000 步训练和调试集评估

使用上述训练配置和 seed 42，将步数从 2048 改为 50,000。单线程 CPU 采样与更新循环耗时 149.77 s，完成 49,744 次更新，重载前后的确定性动作一致。训练抽到了全部 16 个训练目标，200 个回合均超时，回报均为 −250。评估使用调试集 seed 101、102 和 0 步延迟。位置基线 2/2 到达，均用 18 步。SAC、随机和保持 home 均为 0/2，均超时。

| 调试目标（rad） | SAC 最小两轴最大绝对位置误差 | SAC 回合末两轴实际角度（rad） |
| --- | ---: | --- |
| `(0.10, 0.10)` | 0.1030 rad | `(-0.3993, 0.7586)` |
| `(-0.10, 0.10)` | 0.1004 rad | `(-0.2755, 0.5230)` |

两条 SAC 轨迹均未进入两轴 0.01 rad 的位置误差范围。两个目标的第一轴方向相反，但两次起步都输出了负的第一轴增量。逐步日志没有命令裁剪、仿真警告或实际关节越限。本次训练未采到成功，所有转移奖励均为 −1，接近目标和远离目标的未成功动作获得相同即时奖励。缺少成功奖励和距离区分是有证据支持的学习障碍假设；奖励设计与探索、超参数及其他训练因素的贡献尚未用对照实验分离。此次 checkpoint 仅用于调试集。

#### 对已有结果的补充诊断

只读取上述 checkpoint、已有训练和调试日志，另从 home 对固定目标 `(0.1, 0.1)` 做两次单步检查，无训练和参数更新。命令为零时，步末最大轴位置误差约为 `0.099996 rad`；命令增量 `(0.02, 0.02) rad` 时，步末最大轴位置误差约为 `0.096382 rad`。两步均未达到终止条件，即时奖励都是 `−1`。这展示了当前任务在这两个状态下缺少距离区分，不证明奖励设计是失败的唯一原因。

checkpoint 在两个调试目标的 home 状态下给出确定性动作分别为 `(-0.9453, -0.5459)` 和 `(-0.8508, -0.4076)`。第一轴目标正负不同，动作方向都为负。本次检查仅覆盖这两个初始状态，不代表策略对所有目标均不敏感。checkpoint 的 `alpha` 从配置初始值 `0.01` 变为约 `0.000145`；它反映最终保存状态，现有日志没有训练过程中的温度、策略分布、损失或逐步状态，不能由此还原探索过程或断定其为根因。

已检查数据路径：归一化动作乘 `0.02` 后送入两轴环境，回放池保存归一化动作，超时回合按计划 reset 且保留 bootstrap；参数更新及确定性动作重载一致。日志不能证明所有算法或环境细节无误。短检查原始数据见 补充诊断（本机归档）。后续若做对照，先补逐步误差、动作、奖励、温度和损失记录，再控制单个因素进行验证；另行评估训练预算和运行方式。

### 训练日志与审计短检查

`panda_reach.py train` 的 `config.json` 记录奖励模式、阈值、日志频率和相关 Python 源码的 SHA-256。每次训练在 `episodes.jsonl` 记录初态、目标、回报、最小的两轴最大绝对位置误差、末步误差和速度、满足位置或速度阈值以及同时满足两者的步数。还记录归一化动作的均值、标准差和各轴 `|a| ≥ 0.95` 的比例，以及命令实际被裁剪的步数和比例。训练预算截止时的未完成回合标为 `partial`。

| 输出 | 字段与采样规则 |
| --- | --- |
| `steps.jsonl` | 观测、下一观测、仿真时间、归一化动作、请求与实际应用的增量、奖励和结束标志。`--step-log-every N` 按环境步采样。 |
| `updates.jsonl` | 两个 Critic、Actor 和温度损失、温度 `alpha`、平均 TD 目标、批内采样动作均值与标准差、熵估计和梯度 L2 范数。`--update-log-every N` 按优化更新次数采样。Critic 梯度在各自反向传播后、Actor 反向传播前采集。 |

两个采样参数默认均为 250，设为 0 可关闭相应文件。开启时额外记录首条和末条；梯度和损失检查在被采样的更新上执行。日志数据来自已有 transition 或更新张量，无需额外采样。短检查可沿用上文的 `$py`、`$scene`、`$project`，从仓库根目录使用两个新路径：

```powershell
$run = Join-Path "runs" ("panda-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
& $py "$project/panda_reach.py" train $scene "$run/logging-on" --steps 320 --seed 42 --step-log-every 25 --update-log-every 31
& $py "$project/panda_reach.py" train $scene "$run/logging-off" --steps 320 --seed 42 --step-log-every 0 --update-log-every 0
```

2026-11-12 的审计检查了动作单位、回放数据、终止掩码、SAC 更新顺序和温度更新。现有证据未定位到能解释 200 回合全部超时的代码错误。初期 256 个随机步只够约一个 250 步完整超时回合，探索覆盖和奖励设计仍是待验证假设。原 checkpoint 保存了网络和优化器等状态，精确续训还需要原回放池及环境状态。

同一环境、seed 42、两次各 320 步短检查均完成 64 次更新，开关采样日志后，回合记录、网络张量、优化器、温度和随机状态一致。开启时生成 14 条步日志和 4 条更新日志，循环耗时约 0.29 s。该测量仅用于短检查，完整运行的磁盘开销仍待测量。原 64 步单次更新与重载检查通过。固定边界检查验证了成功、快速经过目标、物理限位失败、超时前状态、bootstrap 掩码和两层动作限幅。位置基线在两例调试目标均用 18 步到达，64 次更新的 checkpoint 仅用于验证评估入口。详见 验证汇总（本机归档） 和 边界短检查（本机归档）。

原始产物在 `<本机路径>`，包括 `train/episodes.jsonl`、`train/summary.json`、`validation/sac.jsonl` 和 `validation/summary.json`。

### Panda SAC 的 HER 短检查

训练命令加 `--her` 启用 Panda 的 HER 回放。每批对一半的经验尝试重标注，候选来自同一回合中已经观测到的当前步末或更晚的实际关节位置。候选位置须在目标允许范围内、两轴速度绝对值均小于 0.05 rad/s，且该步没有物理失败。训练 seed 为 42 时，候选采样用独立 seed 44。找不到候选、当前状态已满足新目标或源经验为物理失败时，使用原经验。

重标注同时写入当前和下一观测的任务目标。沿用原来的位置与速度成功判据，重算奖励和 Critic 的终止掩码。重标注成功时奖励为 0 且停止 bootstrap，其余可重标注样本奖励为 −1 且继续 bootstrap。真实回合按环境的终止和超时字段切分。真实物理失败仍触发训练中止。训练日志分别记录实际回合结果和回放里的重标注次数、成功次数及跳过原因。评估使用上文的原始稀疏奖励、验证目标和成功条件。

**历史轨迹检查。** 原 5 万步运行仅保存了训练回合摘要和调试集的完整逐步轨迹。调试集 SAC 共 500 步，18 步满足低速阈值，全部来自 seed 102，实际角度集中在约 `[-0.288, -0.275] × [0.523, 0.534]` rad。按上述规则，均匀抽取两回合中的一个时间步，再在该步可用的未来候选中均匀选目标，每次尝试得到成功经验的估计概率约为 0.21%。这两条轨迹的候选都在 16 个训练目标的 0.01 rad 位置容差之外。同期随机基线有 23/500 步低速，分布更广。另一份 seed 42 的 320 步审计回合日志记有 14 步低速，但采样日志不足以还原完整位置分布。诊断只读取历史产物。见分析脚本（本机归档）和轨迹统计（本机归档）。

从仓库根目录用上文的 `$py`、`$scene` 和 `$project` 复现。输出目录需全新。

```powershell
$run = Join-Path "runs" ("panda-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
& $py "$project/panda_her_smoke.py"
& $py "$project/panda_reach.py" train $scene "$run/her-320" --steps 320 --seed 42 --her --step-log-every 1 --update-log-every 1
& $py "$project/panda_reach.py" evaluate $scene "$run/her-validation" --split validation --checkpoint "$run/her-320/sac_cpu.pt"
```

Agent 短检查覆盖高速经过、原超时样本和重标注成功样本的 bootstrap 掩码、物理失败、目标边界、回合边界和回放池容量。320 个环境步完成 64 次 SAC 更新。共尝试重标注 2,048 次，实际重标注 1,857 次，其中 41 次成功。计数按采样次数统计，经验可能重复抽到。checkpoint 重载前后参考动作一致，调试集位置基线 2/2 成功，短跑的 SAC + HER 为 0/2。关闭 HER 的 320 步复跑与此前审计的回合、步骤和更新日志逐字节相同，保存的网络、优化器、温度和随机状态也一致。原 64 步 CPU 检查通过。见训练摘要（本机归档）、调试集结果（本机归档）和复核记录（本机归档）。

### 2026-11-12 单 seed 50,000 步 SAC + HER 对照

Agent 在后台完成一次 50,000 步 CPU 单 seed 42 运行。场景哈希、模型、稀疏奖励、训练目标、warmup、batch、SAC 参数和训练步数与上文原 SAC 运行一致。训练保存每步轨迹和每 250 次更新的采样指标。复现命令使用上文的 `$py`、`$scene` 和 `$project`，输出目录须为新路径。

```powershell
$run = Join-Path "runs" ("panda-" + (Get-Date -Format "yyyyMMdd-HHmmssfff"))
& $py "$project/panda_reach.py" train $scene "$run/train" --steps 50000 --seed 42 --her --step-log-every 1 --update-log-every 250
if ($LASTEXITCODE -eq 0) {
    & $py "$project/panda_reach.py" evaluate $scene "$run/validation" --split validation --delay-steps 0 --checkpoint "$run/train/sac_cpu.pt"
}
```

采样与更新耗时 171.47 秒，共 49,744 次更新。原 SAC 训练的 200 个回合均超时。SAC + HER 训练有 930 个成功回合、87 个超时及 1 个预算结束的未完回合，16 个训练目标都至少成功过一次。新 checkpoint 的重载参考动作完全一致。调试集沿用 seed 101/102、home 初态、零延迟和确定性策略。评估成功要求两轴位置误差均小于 0.01 rad 且绝对速度均小于 0.05 rad/s。

| 调试目标 (rad) | 原 SAC | SAC + HER | SAC + HER 成功时最大轴误差 / 速度 |
| --- | --- | --- | --- |
| `(0.10, 0.10)` | 250 步超时 | 16 步成功 | 0.00897 rad / 0.04336 rad/s |
| `(-0.10, 0.10)` | 250 步超时 | 15 步成功 | 0.00517 rad / 0.04194 rad/s |

位置基线两例均为 18 步成功，两次运行的基线轨迹日志逐字节一致。HER 运行在前 10,000 步已抽到 13,062 次重标注成功经验，第 11,119 步出现首次真实目标成功。全程 83,886 次重标注成功抽样里，有 11,380 次的替代目标落在某个原训练目标的位置容差内。抽样次数可含重复经验。以上结果支持 HER 在这次运行中补足了稀疏成功信号，评估范围为单一训练 seed 与两个调试目标。

产物见 训练摘要（本机归档）、调试集结果（本机归档）和对照与回放重建（本机归档）。分析脚本（本机归档）从完整轨迹和两个回放随机种子重建采样，计数与训练摘要一致。

### 多 seed 配对与独立目标评估

训练 seed 固定为 43、137、2027。每个 seed 分别从头跑原稀疏奖励 SAC 和 SAC + HER，每次 50,000 环境步，模型、训练目标、SAC 参数与 seed 42 的对照一致。每个训练进程使用单线程 CPU。三个 seed 并行，同 seed 两个训练依次运行。用原 seed 42 的两个 checkpoint 补做独立目标评估作为参考。

启动训练前用独立的目标采样 seed 6111201 固定 24 个目标，四个符号象限各 6 个，两轴绝对值都在 0.04–0.16 rad 内。目标之间以及与训练集和已使用调试、测试目标之间，两轴最大差至少 0.021 rad。评估用例 seed 为 30101–30124，模型用确定性动作，从 home 出发，零延迟，每例最多 250 步。成功仍要求两轴位置误差均小于 0.01 rad，速度绝对值均小于 0.05 rad/s。完整坐标见 冻结协议（本机归档）。

每次训练保存完整逐步轨迹和每 250 次更新的指标。可将运行脚本（本机归档）和分析脚本（本机归档）复制到新输出目录，按下列命令复现。沿用上文的 `$py` 和 `$scene`，`$projectPath` 使用子项目的绝对路径。运行会执行六次 50,000 步 CPU 训练和两份参考 checkpoint 的评估。

```powershell
$oldSac = '<本机路径>'
$oldHer = '<本机路径>'
& $py "$run/run_experiment.py" freeze $projectPath $scene
& $py "$run/run_experiment.py" run $projectPath $scene "$oldSac/train/sac_cpu.pt" "$oldHer/train/sac_cpu.pt"
& $py "$run/analyze_results.py" $oldSac $oldHer
```

六次新训练均完成 50,000 环境步和 49,744 次更新，采样与更新耗时每次 163–181 秒。复核确认训练配置成对匹配，且每对前 256 个预热步的逐步日志完全相同。位置基线在固定新目标上为 24/24，各次轨迹日志一致。

| 训练 seed | 原 SAC 独立目标成功 | SAC + HER 独立目标成功 | 原 SAC 训练超时 / 已结束回合 | SAC + HER 训练超时 / 已结束回合 |
| --- | --- | --- | --- | --- |
| 43 | 0/24 | 14/24 | 200/200 | 113/521 |
| 137 | 0/24 | 24/24 | 200/200 | 137/558 |
| 2027 | 2/24 | 24/24 | 197/204 | 94/1,033 |
| 三组总计 | 2/72 | 62/72 | 597/604 | 344/2,112 |

参考 seed 42 的旧模型在同一独立目标集为原 SAC 0/24、SAC + HER 24/24。只计新训练的三个 seed，HER 的 344 个训练超时中 340 个在前 40,000 步结束。最后 10,000 步共 4/1,069 个超时，全部出自 seed 43，最后一次在第 41,097 步。三组 HER 各自最后 100 个已结束回合均成功。训练预算结束时的未完回合单独记录，原 SAC 有 1 个，SAC + HER 有 3 个。

新目标评估仍有 10 次 HER 超时，均来自 seed 43。逐步日志中，这 10 例各有 222–233 个低速步，最小的两轴最大绝对位置误差为 0.0105–0.0191 rad，始终高于 0.01 rad 位置阈值。这个 seed 在这几例的主要限制是停下后的定位精度。完整结果见 逐 seed 汇总与逐例分析（本机归档）和后台作业记录（本机归档）。

### Seed 43 低速停滞与位置控制收尾

将旧独立目标集的 10 条超时轨迹按 `goal - q = (goal - ctrl) + (ctrl - q)` 分解。末段位置误差最大的轴上，`ctrl` 离目标仍有 0.01054–0.02219 rad。每条轨迹最后 50 步的归一化动作绝对值均小于 `7e-7`，`ctrl` 峰峰值低于 `1.3e-7 rad`，没有命令裁剪、越限或仿真警告。第一轴主导的六例中 `q` 已基本等于 `ctrl`。第二轴主导的四例另有约 0.0072–0.0076 rad 的伺服偏差，方向抵消了一部分命令误差。日志表明策略在任务目标之前就停止调整位置命令。见十例逐项诊断（本机归档）。

对照沿用同一 seed 43 checkpoint、确定性动作、home 初态、零延迟、250 步上限及原成功条件。试验组连续 3 步同时满足两轴最大位置误差小于 0.03 rad、最大速度小于 0.05 rad/s、策略归一化动作绝对值均小于 0.01 时，转交现有位置控制器直至结束。后者每步发送 `clip(goal - ctrl, -0.02, 0.02)` rad。0.03 rad 是切换条件，成功仍要求两轴误差均小于 0.01 rad 且速度绝对值均小于 0.05 rad/s。旧 24 例用于回归，再用 seed 6111202 在评估前固定 24 个新目标，四个符号象限各 6 个，两轴绝对值在 0.04–0.16 rad 内。新目标与训练目标、之前用过的目标和彼此之间最大轴差至少 0.021 rad，用例 seed 为 30201–30224。见冻结协议（本机归档）。

| 目标集 | 原策略 | 加位置控制收尾 | 原超时样本最小最大轴误差中位数 |
| --- | --- | --- | --- |
| 原 24 例 | 14/24 | 24/24 | 0.01216 → 0.00449 rad |
| 新 24 例 | 9/24 | 24/24 | 0.01367 → 0.00405 rad |

原策略完整重现了历史逐步轨迹。所有对照轨迹在发生切换前逐步相同。原 10 个超时回合在加入收尾后均于 34–58 步成功，成功时最大轴误差均不超过 0.00671 rad，原已成功样本在两组目标中均保持成功。这里测量的是同一模型加控制器后的离线到达表现。将对照脚本（本机归档）复制到一个新输出目录，使用上文的 `$py`、`$scene` 和子项目绝对路径 `$projectPath` 复现。见逐例对照结果（本机归档）。

```powershell
& $py "$run/precision_compare.py" $projectPath $scene '<本机路径>'
```

**前期训练超时。** seed 43、137、2027 的 HER 训练在前 10,000 步共出现 117 个超时、3 个真实目标成功。为检查成功演示是否容易取得，用现有位置控制器跑了 16 个训练目标，全部成功，共 326 步。选两轴目标绝对值相等的 8 条轨迹，四个符号象限各两条，共 152 步，可与 104 个随机步合计 256 步作为下一次预热对照。演示轨迹单独计数，比较 Actor 自主采样前 10,000 步的超时比例与首次真实目标成功步数。此处已验证演示收集，学习效果待对照。演示日志（本机归档） · [离策略学习利用先验轨迹的研究](https://proceedings.mlr.press/v202/ball23a.html)

### Seed 43 演示预热对照

`--demo-warmup` 用位置控制器依次完成 8 个两轴目标绝对值相等的训练回合，累计 152 步，再随机采样 104 步。演示和随机预热均进入原 HER 回放，第 257 步开始 Actor 采样与 SAC 更新。未结束的随机回合按原训练逻辑接续。控制组是上文 seed 43、50,000 步、256 步随机预热的 SAC + HER。检查确认关闭演示时，新的 320 步短跑逐步复现旧训练，开启演示时第 1–152 步为 8 条完整成功演示。

前 10,000 步的自主回合统计纳入从重置起全程由 Actor 控制、且在第 10,000 步或之前结束的回合。演示、纯随机、跨预热边界的混合回合和窗口截止时的未结束回合单独列出。首次自主成功使用全局环境步数。最终评估用前一节的 24 个新目标，目标 seed 6111202、用例 seed 30201–30224，使用确定性 Actor、home 初态、零延迟和原成功阈值。两组评估均直接运行 Actor。见冻结协议（本机归档）。

复现时使用新的 `$run`，沿用上文的 `$py`、`$scene` 和子项目绝对路径 `$projectPath`。

```powershell
& $py "$projectPath/panda_reach.py" train $scene "$run/train" --steps 50000 --seed 43 --her --demo-warmup --step-log-every 1 --update-log-every 250
if ($LASTEXITCODE -eq 0) {
    & $py "<本机路径>" evaluate $projectPath $scene "$run/train/sac_cpu.pt" "$run/evaluation-demo"
}
```

演示组后台完成 50,000 步和 49,744 次更新，采样与更新耗时约 184 秒。复核确认模型、训练和 HER 参数、成功阈值与控制组一致。旧 checkpoint 的逐步评估日志也与上一节完全一致。

| Seed 43 的 SAC + HER | 前 10,000 步 Actor 自主超时 / 已结束回合 | 首次 Actor 自主成功的全局环境步 | 固定 24 例 Actor 成功 |
| --- | --- | --- | --- |
| 256 步随机预热 | 37/38（97.4%） | 9,862 | 9/24 |
| 152 步演示 + 104 步随机 | 38/38（100%） | 14,898 | 24/24 |

演示组 8 个演示回合均成功，单独计数。两组各有 1 个跨预热边界的超时回合，控制组另有 1 个纯随机超时回合。两组在第 10,000 步各有 1 个未结束的自主回合。这些回合均排除在上表的前期指标之外。本次演示预热的首次自主成功晚 5,036 步。最终评估中，24 例的最小最大轴误差中位数从 0.01119 降到 0.00655 rad。位置基线两组都是 24/24，评估轨迹一致。当前结论来自 seed 43 的单次配对。

产物见训练摘要（本机归档）、逐例结果与分组统计（本机归档）、统计脚本（本机归档）和后台作业记录（本机归档）。

## 奖励塑形与平滑探索对照及恢复实验

计划以 SAC + HER 为基线，分别对照奖励塑形、平滑探索及组合，并加入原 SAC 和位置控制基线。预设训练 seed 为 1103、1129、1181、1213、1277；验证目标生成 seed 为 2026111311，独立终测目标生成 seed 为 2026111312。详细协议、指标和预计成本见 [实验方案](factorial_panda_20261113/实验方案.md)。用户已请求在 2026-11-13 22:00（北京时间）一次性启动，每个 seed 的五个训练组以 CUDA 独立进程并发；是否启动和结果以定时任务和运行日志为准。此处仅为待执行方案，不代表任何效果已验证。

原定运行在 seed 1103 的奖励塑形组出现有限关节越限，已保留部分日志。应用户要求准备[恢复修订](factorial_panda_20261113/恢复修订_20261113.md)：所有训练组统一将可恢复越限作为真实失败终止、记录并重置，失败样本的训练基础奖励统一设为 −250（环境原奖励仍记录为 −1）；HER 不重标真实失败。非有限状态和仿真警告仍停止运行。重跑沿用以上训练 seed 和验证目标；由于旧终测有目标被短 smoke 接触，正式重跑前更换为目标生成 seed 2026111313、用例 ID 70101–70148，并排除与旧目标精确重复。修订发生于已见部分结果之后，不能称为完全盲态预注册，也不能与历史不同失败成本的运行直接合并。新输出与旧输出隔离，状态以新日志为准。

恢复实验已完成五 seed 的 25 个训练及各阶段验证与终测，独立完整性审计 PASS。10k–50k 五个验证节点的成功率梯形平均，对五个 seed 等权：原 SAC 1.61%、HER 52.14%、HER+塑形 61.98%、HER+平滑 34.79%、两者组合 31.93%。组合减 HER 的均值为 −20.21 个百分点，五个 seed 的差值均为负，未达到预设的组合提升门槛；单独塑形较 HER 平均 +9.84 个百分点，配对区间包含零，属于需进一步验证的迹象。终测每 seed 使用相同 48 个新目标，位置控制基线 48/48，塑形组每个 seed 均为 48/48；不能把它们当作 240 个不同目标，也不能据此声称策略优于基线。10k 时各组平均成功率均不足 3%，本实验未解决最初阶段探索效率问题。详细过程、各 seed、失败率及局限见验收报告（本机归档）、分析数据（本机归档）和完整性审计（本机归档）。以上为 Agent 按用户要求运行，未计为本人手动操作。

## 平滑探索日志诊断与后续候选

2026-11-14 根据恢复实验的训练日志做只读诊断：已完成 Actor 回合、每 250 步系统快照和每 1,000 步累计 HER 计数应分开解释；快照与 250 步超时上限同周期，不能视为无偏完整轨迹。结论和复核脚本见日志诊断（本机归档）。

下一步候选为固定 HER、塑形和失败成本，比较行为 EMA 当前动作系数 1.0、0.8、0.5；拟用训练 seed 1301、1423、1567、1693、1789，验证目标生成 seed 2026111401、终测目标生成 seed 2026111402。方案、非训练预演和未来启动命令见 [EMA 后续验证方案](factorial_panda_20261113/EMA后续验证方案_20261114.md)。该候选尚未正式冻结和执行，不能据此宣称改进。

## Panda 末端位置数值 IK

`panda_ik.py` 从 home 出发，在 `link0` 坐标系中求 Panda `hand` 原点的三维位置。求解使用 MuJoCo 位置 Jacobian 的阻尼最小二乘，关节角裁剪至关节和执行器范围交集，并用已有 FK 回代验证。参数为阻尼 0.02、单次更新每轴最多 0.15 rad、最多 150 次迭代和 0.1 mm 位置阈值。输出误差、限位、最小奇异值和退出原因。给定 `--goal X Y Z` 时，未收敛返回非零退出码。局部求解覆盖位置，姿态、碰撞和轨迹执行尚待验证。

无随机采样。三个可达位置分别由 home 和关节偏移 `(0.20,-0.18,0.10,0.15,0,0.10,0)`、`(-0.22,0.16,-0.12,-0.20,0.08,0,0)` rad 做 FK 生成，均从同一 home 求解。后两个样例各四次迭代，回代误差分别为 2.44 μm、1.55 μm。远目标 `(3,0,3)` m 达到迭代上限，`(0,0,-3)` m 触发关节限位后因误差无法继续降低而退出。2026-11-16 Agent 复验的固定条件、命令和输出见 [IK 固定样例与复现](IK固定样例与复现_20261116.md)。

## 在本机复现

从工作区根目录运行。已有模型缓存和虚拟环境时可直接运行最后一行。

```powershell
$cache = Join-Path $env:LOCALAPPDATA 'EmbodiedAI/PandaSmoke'
$model = Join-Path $cache 'menagerie-98a70c5'
$python = Join-Path $cache '.venv/Scripts/python.exe'
$rev = '98a70c569c4c1b6da467a20012f766ee1f8dc521'
New-Item -ItemType Directory -Force -Path $cache | Out-Null
python -m venv (Join-Path $cache '.venv')
& $python -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
git clone --filter=blob:none --sparse --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git $model
git -C $model sparse-checkout set franka_emika_panda
git -C $model checkout --detach $rev
& $python -I 15_RobotSystem/panda_smoke.py "$model/franka_emika_panda/scene.xml"
```

## W2 实测

Windows x64，Python 3.12.10，MuJoCo 3.17.0，NumPy 2.5.4。两次都从同一个 `home` 重置，使用固定输入，测试没有随机采样。每次前 100 步保持 `home` 控制目标，后 150 步将 `joint1` 的目标设为 0.100 rad。其余控制目标不变。

| 检查 | 实测 |
| --- | --- |
| 加载模型 | `nq=9`、`nv=9`、`nu=8`，物理步长 0.002 s |
| 关键帧重置 | 两次重置后 `qpos`、`qvel`、`ctrl`、仿真时间与 `home` 一致 |
| 推进 | 每次 250 步对应 0.500 s 仿真时间，状态有限，无 MuJoCo 警告 |
| 保持阶段 | 七轴相对 `home` 的最大位置偏移 0.00578 rad |
| 单轴阶跃 | `joint1` 结束位置 0.09537 rad，末帧绝对误差 0.00463 rad |
| 同进程重复 | 两条轨迹的 `qpos`、`qvel`、`qacc` 逐点最大绝对差均为 0 |

脚本输出逐轴名称、运行时地址和控制范围。七个臂轴的 `qpos`、`qvel` 地址和执行器下标本次均为 0–6，臂轴位置目标用 rad，臂轴速度用 rad/s。夹爪使用下标 7 的腱执行器，控制值范围 0–255，`home` 为 255。

以上为 Windows 上的 MuJoCo 独立冒烟。WSL 上的集成冒烟见下节。

## ROS2 端到端冒烟

2026-11-06 在 WSL 2 Ubuntu 26.04.1、ROS2 Lyrical、Python 3.14.4、MuJoCo 3.17.0、NumPy 2.5.4 上验证。C++ 发布器发送 `JointTrajectory`，Python 节点校验并执行，在 `/panda_smoke/joint_states` 和 `/clock` 发布数据，rosbag2 写入 MCAP。

### 复现

在 WSL 的 Bash 中运行，先准备 ROS2 Lyrical、`rosbag2`、`trajectory_msgs`、`colcon`、CMake、C++ 编译器和 Python venv。保留上一节的固定版本模型。以下命令从仓库根目录开始，模型目录按本机位置设置。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
cache="$PWD/runs"
scene="$PANDA_SCENE"
work=$(mktemp -d)
python3 -m venv --system-site-packages "$work/venv"
"$work/venv/bin/python" -m pip install 'mujoco==3.17.0' 'numpy==2.5.4'
cd "$work"
colcon build --base-paths "$project/ros2_smoke/trajectory_publisher" --packages-select panda_trajectory_smoke
source "$work/install/local_setup.bash"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=86
"$work/venv/bin/python" "$project/ros2_smoke/run.py" "$scene" "$cache/ros2-results/$(date +%Y%m%d-%H%M%S)"
```

本机已有可复用的 WSL venv，路径为 `/var/tmp/embodied-panda-ros2-lyrical/ros-venv`。WSL 下载 wheel 超时时，在 Windows 侧从 PyPI 下载 Linux x86_64 / CPython 3.14 的 wheel，再在 WSL 用 `pip install --no-index --find-links "$cache/ros2-wheelhouse"` 安装两个固定版本。加载 ROS 的 Python 节点依赖环境脚本设置的 `PYTHONPATH`。

### 固定输入与结果

下表记录固定目标为 0.1 rad 的历史基线，冒烟发布器源码当前使用 0.15 rad 做参数对照。测试不使用随机采样，每次从同一 `home` 重置，`joint1` 在 0.4 s 内线性升至目标，之后保持到 0.8 s。其余轴保持 `home` 目标，夹爪保持原命令。物理步长为 2 ms，每五步发布一次状态与时钟。稳态窗口为 0.61–0.80 s，预设每轴窗口内平均绝对误差不超过 0.05 rad。采样周期描述仿真时间，墙钟发布率另行测量。

| 检查 | 结果 |
| --- | --- |
| C++ 到 Python | 干净目录 `colcon build` 通过。合法轨迹接收并执行，未知关节、NaN、越界、重复时间点均被拒绝。 |
| 固定轨迹 | `joint1` RMSE 0.01519 rad、峰值误差 0.02418 rad、保持窗口平均绝对误差 0.00119 rad。七轴保持窗口均通过阈值。 |
| 录包与状态回放 | MCAP 包含 5 条命令、80 条关节状态与 80 条时钟消息。停止实时仿真后，状态和时钟逐条回放一致。 |
| 命令重执行 | 新建仿真节点并恢复同一 `home`，只播放命令。新生成的 80 个状态样本中，角度和角速度的最大逐样本差均为 0。 |

每次运行将 `summary.json`、`bag_info.txt`、各进程日志和 MCAP 存到新结果目录。录包使用默认记录时钟，状态误差按消息中的仿真时间对齐。此冒烟只覆盖一条单轴变化轨迹和输入校验，项目正式多轨迹评估、reset 服务、命令停发保护及墙钟频率统计见 [采样与评估说明](#2026-11-17-固定关节位置跟踪基线)。

## W3 实时关节状态节点

`ros2_ws/src/panda_sim` 提供独立的 `ament_python` 包。节点从 `home` 开始，每次定时回调推进 10 个 MuJoCo 物理步，然后按关节名发布七轴的位置和速度。使用固定模型的 2 ms 物理步长时，配置的发布周期为 20 ms。定时器使用单调时钟，状态消息和 `/clock` 共用步进后的 MuJoCo 时间。

2026-11-06 已在 WSL 2 Ubuntu 26.04.1 / ROS2 Lyrical 的干净目录验证构建与启动，CLI 订阅端收到一条含七轴位置和速度的实时 `JointState`。同一节点订阅 `trajectory`，按下一节的约定执行关节位置轨迹。

在 WSL 终端 A 构建并启动。按本机位置设置仓库路径、模型和已有 MuJoCo venv。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
scene="$PANDA_SCENE"
panda_python="$PANDA_PYTHON"
cd "$(mktemp -d)"
echo "构建目录 $PWD"
colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_sim panda_trajectory_cmd
source install/local_setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=89
ros2 run --prefix "$panda_python" panda_sim panda_sim_node --ros-args \
  -r __ns:=/panda -p "scene_path:=$scene"
```

在终端 B 检查实时消息。验收信号是 `/panda/panda_sim` 为发布者，消息含 `joint1` 到 `joint7` 及顺序匹配的位置和速度。

```bash
source /opt/ros/lyrical/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=89
ros2 topic info -v /panda/joint_states
ros2 topic echo /panda/joint_states --once --timeout 5
```

保持终端 A 运行。在终端 B 加载 A 打印的构建目录中的 `install/local_setup.bash`，再启动 C++ 发布器。下面的 `build_dir` 要替换为 A 打印的路径。

```bash
build_dir='/tmp/替换为终端A打印的目录名'
source "$build_dir/install/local_setup.bash"
ros2 run panda_trajectory_cmd publish_trajectory --ros-args -r __ns:=/panda
```

C++ 发布器等待发现两个订阅端点，向 `/panda/trajectory` 发送一次固定的两点轨迹，并在执行期间向 `/panda/trajectory_heartbeat` 发心跳。新启动的仿真节点应先记录 `Accepted JointTrajectory with 2 points`，再记录 `Trajectory complete; holding final targets`。此时读取 `/panda/joint_states`，可观察 `joint1` 接近 0.1 rad。

## W4 固定步长轨迹执行

- 轨迹必须包含七个不同的臂轴名，节点按名称将每个点重排到模型顺序。支持位置字段。点值须有限，并落在关节范围和执行器控制范围的交集内。
- `header.stamp` 须为零。第一个 `time_from_start` 须为零，之后严格递增。第一个点须与当前保持目标一致，绝对容差为 1 µrad。执行中收到新命令记录 `busy` 并拒绝。
- 接收时记录当前 MuJoCo 时间。每个物理步开始前按该时间差做分段线性插值，将结果写到臂轴的 `ctrl`，再推进物理。经过末点时保持最后一组位置目标。C++ 示例从 `home` 出发，重跑同一示例前可重启仿真节点。

轨迹执行功能检查使用固定输入，无随机采样。接入心跳前，已验证 C++ 示例在 0.4 s 内将 `joint1` 从 0 指向 0.1 rad，实测角进入目标 ±0.005 rad。关节名倒序的回程轨迹在 0.6 s 内回到初始目标，并用重复命令验证了 `busy`。错误关节名、非零时间戳、点维度错误、重复时间点、NaN、越界和首点不连续均通过拒绝检查。

### 仿真时钟重置与停发

- `/clock` 和 `JointState.header.stamp` 取同一个仿真采样时刻。每个 ROS 域使用一个 `/clock` 来源。用仿真时间的下游节点启用 `use_sim_time`，等到首个非零时钟再处理数据。
- `/panda/reset` 使用 `std_srvs/srv/Trigger`。执行中返回失败，空闲时恢复 `home` 的状态和控制量，并重建 volatile 轨迹订阅以清除其积压。下一帧时钟从新的仿真起点推进，录包与离线分析按回退分段。
- 单发送方约定下，C++ 示例每隔 50 ms 发一次空心跳。接收端用单调时间检查，活动轨迹在 250 ms 内未收到心跳时终止插值，保持最后已应用的 `ctrl`。迟到的心跳不会恢复已超时的轨迹。轨迹正常结束后保持末点。`/clock` 和心跳使用 best effort、队列深度 1，轨迹命令使用 reliable。

终端 A 运行仿真时，可在终端 B 观察时钟、调用 reset。执行完 C++ 示例后重置，可从 `home` 再次运行同一示例。

```bash
ros2 topic echo /clock --qos-reliability best_effort --once --timeout 5
ros2 service call /panda/reset std_srvs/srv/Trigger '{}'
```

要在空闲重置后复现无心跳超时，可单独发送一条两秒轨迹。终端 A 应记录 `Trajectory heartbeat timed out; holding last targets`，随后 `joint1` 保持在中间位置附近。

```bash
ros2 topic pub --once /panda/trajectory trajectory_msgs/msg/JointTrajectory \
  '{joint_names: [joint1, joint2, joint3, joint4, joint5, joint6, joint7], points: [{positions: [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]}, {positions: [0.5, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], time_from_start: {sec: 2}}]}'
```

WSL 功能检查验证了状态与时钟的时间戳一致、仿真时钟消费、忙时拒绝重置、空闲重置后时钟回退、运行中心跳停发后保持，以及 reset 后重新执行 C++ 示例。停发场景使用上面的两秒轨迹，按 50 ms 间隔发送五次心跳后停发，并检查迟到的心跳未恢复运动。原有冒烟和录包回放复跑通过。正式录包时把心跳和轨迹一起视为命令流，重执行时保留实时仿真发布的 `/clock`。

## W5 正式节点录包与回放冒烟

使用上述 `panda_sim` 和 `panda_trajectory_cmd` 正式节点。C++ 发一条固定轨迹，`joint1` 在 0.4 s 内从 0 变到 0.1 rad，其余轴保持 `home` 目标，无随机采样。录包保存 `/panda/trajectory`、`/panda/trajectory_heartbeat`、`/panda/joint_states` 和 `/clock`。录包以接收时间保留消息顺序与心跳间隔，轨迹对比用 `JointState` 自带的仿真时间。

1. 关闭实时节点，只从 MCAP 播放 `/panda/joint_states` 和 `/clock`。检查它们均来自播放器，关节名、位置、速度和消息时间戳逐条与包内数据一致。录制时 `/clock` 为 best effort，回放检查临时用 reliable 逐条比对。
2. 重新启动仿真节点，从 `home` 复位，再只播放轨迹和心跳。实时仿真是 `/clock` 和状态的唯一来源。两次都从各自接受命令的仿真时刻开始对齐，比较后续 20–800 ms 的 40 帧七轴位置和速度，每帧都核对对应的 `/clock` 时间戳。冒烟要求两次轨迹均正常完成，且 `joint1` 的末帧位置进入目标 ±0.005 rad。两个进程接收命令前保持 `home` 的时长可能不同，报告实测差异。

从仓库根目录在 WSL Bash 运行。沿用上文的固定版本模型和 ROS2 Lyrical venv，另需 `rosbag2_py` 和 MCAP 存储插件。为测试使用空闲的本地 ROS 域，避免其他节点也发布 `/clock`。

```bash
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
cache="$PWD/runs"
scene="$PANDA_SCENE"
panda_python="$PANDA_PYTHON"
cd "$(mktemp -d)"
colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_sim panda_trajectory_cmd
source install/local_setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=106
"$panda_python" "$project/ros2_smoke/run_formal.py" "$scene" \
  "$cache/ros2-results/$(date +%Y%m%d-%H%M%S)-formal-bag"
```

每次运行保存 `summary.json`、`bag_info.txt`、进程日志、QoS 配置和 MCAP。2026-11-07 的 `20261107-092518-formal-bag` 在 WSL 2 Ubuntu 26.04.1、ROS2 Lyrical、MuJoCo 3.17.0、NumPy 2.5.4 上通过。录得 1 条轨迹、12 条心跳、94 条状态与 90 条时钟，状态与时钟各自完整回放。重执行的 40 帧中，七轴位置和速度的最大绝对差分别为 `8.08e-9 rad`、`8.78e-8 rad/s`，两次 `joint1` 末帧均约为 `0.099622 rad`。整包的状态和时钟计数可以不同，比较窗口内逐帧检查二者的时间戳。rosbag2 对空心跳类型可能提示缺少内嵌消息定义，脚本使用本机 `std_msgs/msg/Empty` 验证解码和回放。

录包脚本将 C++ 发布器的 `required_subscribers` 设为 2，等仿真节点和录包器分别匹配轨迹及心跳话题后才发送。直接运行发布器和 launch 时默认只要求 1 个订阅者。

## 正式启动与参数

沿用上一节的 WSL 环境，在干净目录构建两个包并加载 overlay。使用空闲 ROS 域，每个域只运行一个发布 `/clock` 的仿真节点。

```bash
ros2 launch panda_sim panda_demo.launch.py "scene_path:=$scene" "sim_python:=$panda_python"
```

launch 默认在 `/panda` 启动仿真和 C++ 发布器。发布器完成一次发送后退出，仿真继续保持最终目标并发布状态，按 Ctrl+C 结束。可加 `send_demo:=false` 只启动仿真。启动时可设置 `steps_per_message`（默认 10）和 `heartbeat_timeout_s`（默认 0.25 s）；两个值必须为正数，心跳期限须有限。模型步长为 2 ms，默认每个定时周期推进十步，名义状态周期为 20 ms。

## 三组固定轨迹评估

使用同一固定版本模型和默认参数。每组运行 3 次，共 9 次，无随机采样。每次启动新进程，通过 `/panda/reset` 恢复 `home` 并观察时间回退，等待仿真从 reset 后推进至少 1 s，再由 C++ 发出固定命令。用独立 MCAP 记录轨迹、心跳、状态和时钟。所有轨迹点的七轴值见 [发布器](ros2_ws/src/panda_trajectory_cmd/publish_trajectory.cpp) 和运行后的 `summary.json`。

| 输入 | 运动时长 | 覆盖 |
| --- | ---: | --- |
| `single_axis` | 0.4 s | `joint1` 从 0 到 0.1 rad，其余保持 home |
| `multi_axis` | 0.8 s | 七轴同时由 home 到一个终点 |
| `multi_waypoint` | 1.2 s | 七轴由 home 经两个中间点到终点，部分关节中途反向 |

以接收命令日志中的仿真时刻为零点，在其后每 20 ms 核对状态与 `/clock`。用包里的位置点分段插值得到参考，误差为实测减参考。运动段报告每轴 RMSE 和最大绝对误差。结束后继续记录 400 ms，以最后 200 ms 的最大绝对误差作为稳态验收，阈值预先固定为每轴 0.05 rad。重复差异取同一相对时间、同一关节在三次运行中的最大两两差值。频率和 P50/P95/P99 间隔从 MCAP 接收墙钟计算，反映录包端收到状态的情况。

```bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=113
out="$cache/ros2-results/$(date +%Y%m%d-%H%M%S)-three-trajectories"
"$panda_python" "$project/ros2_smoke/evaluate.py" run "$scene" "$out"
# 从既有 MCAP 和接受日志重新生成数据表
"$panda_python" "$project/ros2_smoke/evaluate.py" analyze "$out"
# 使用已安装 Matplotlib 的 Python 绘图
python "$project/ros2_smoke/plot_tracking.py" "$out"
```

输出包括 `configuration.json`、`summary.json`、63 行七轴指标的 `joint_metrics.csv`，以及各次运行的 MCAP、日志和 `tracking.csv`。绘图脚本为每组生成一张 `tracking.png`。

2026-11-07 在本机 WSL 2、ROS2 Lyrical、MuJoCo 3.17.0、NumPy 2.5.4 上完成九次。最大稳态绝对误差为 0.01137 rad，运动段最差单轴 RMSE 为 0.04432 rad，最大瞬时误差为 0.07341 rad。状态接收频率为 49.98–50.04 Hz，各次 P99 接收间隔最高为 21.19 ms。三组重复运行的最大位置差为 `8.07e-9 rad`。本轮实测文件在 `<本机路径>`。
### 面试演示用的七轴多航点回放

2026-11-22 复跑上述 multi_waypoint 一次。七轴均改变目标角，关节从 home 经两个中间点到终点，轨迹时长 1.2 s。另记录 0.4 s 以检查跟踪后的稳定性。MCAP 的七轴状态用于渲染，图中第一轴的轨迹参考按消息中的航点和时间插值。此次只运行一次，最大稳态绝对误差 0.01137 rad，低于协议的 0.05 rad。 [MCAP 与采样](results/20261122_multi_waypoint/) · [单次摘要](results/20261122_multi_waypoint/summary.json)

## 2026-11-14 动作延迟设计与训练执行接口检查

[延迟对照实验设计](动作延迟对照实验设计.md) 规定 0、20、40 ms 动作 FIFO、已有策略及队列感知的传统控制对照、独立目标集、保持成功、超调、安全和成本指标。正式实验的持续评估、强基线、目标集生成器及 runner 尚未完成，不存在可直接运行的完整正式对照命令或可信耗时；当前仅列出旧评估器的诊断命令。设计不预设 RL 优于传统控制，也不代表真实硬件延迟。

[训练执行接口一致性设计](训练执行接口一致性设计.md) 记录观测顺序、动作缩放、物理步时序、重置和后续 ROS 验收。当前原两轴环境默认 `hold`，周期开始直接保持目标；ROS 节点在物理步前按轨迹插值。环境已增加可选 `control_mode='linear'`，旧训练默认不变，训练 CLI 尚未接入新选项。

接口短检查的完整协议：固定 Menagerie `98a70c569c4c1b6da467a20012f766ee1f8dc521` 的 `scene.xml`，MuJoCo 3.17.0、NumPy 2.5.4、`home` 初态、目标 `(0.1, 0.1) rad`；直接输入环境增量 `(0.02, -0.01) rad`，2 ms 物理步共 10 步，ROS 轨迹时长 20 ms，**无随机采样、无 seed**。脚本抽取 ROS 生产源码的物理循环，分别比较默认阶跃模式、固定目标循环、线性模式和 ROS 插值的每物理步 ctrl、q、dq。默认阶跃与 ROS 插值的最大差分别为 `0.02 rad`、约 `0.002254 rad` 和 `0.270103 rad/s`；线性模式与抽取循环的最大差分别约 `1.73e-18 rad`、`2.17e-19 rad` 和 `1.39e-17 rad/s`。仅是单段固定样例中的数值结果。

脚本与复现命令见 [接口设计](训练执行接口一致性设计.md#已运行检查和复现)；摘要（本机归档） 和 逐步数据（本机归档） 包含结果及源码 hash。模型 `scene.xml` 的 hash 不覆盖其引用资源，完整模型依赖固定提交。

该检查未启动 ROS 节点，没有覆盖通信、接收、心跳、reset 或连续动作。现有 ROS 轨迹在第 11 个物理步才处理结束，连续 20 ms 下发可能遭遇 `busy`；不能将单段数值一致称为策略闭环一致。本人复现尚待记录。

## 2026-11-17 固定关节位置跟踪基线

`panda_fixed_position_baseline.py` 从同一固定版本模型的 `home` 复位，使用模型自带的位置执行器，七个臂关节各自发送绝对位置目标，不使用策略、随机数或 seed。`joint1`、`joint2` 目标依次从 `(0, 0)` 在 0.8 s 线性到 `(0.1, -0.1)` rad，保持至 1.2 s，线性至 2.0 s 的 `(-0.1, 0.1)` rad，保持至 2.4 s；其余五轴保持 home 目标。插值每个 2 ms 物理步施加一次，每 10 步记录一次（20 ms）；并非每 20 ms 阶跃更新。夹爪控制维持 home。关节和执行器范围按模型交集在运行前校验，关节名与执行器一一对应也会检查。采样点另检查关节位置是否超出模型限位及 0.001 rad 容差；该采样检查不保证采样点之间无越界。

从仓库根目录的 PowerShell 运行，输出路径必须不存在：

```powershell
$py = (Get-Command python).Source
$scene = Join-Path $env:LOCALAPPDATA 'EmbodiedAI\PandaSmoke\menagerie-98a70c5\franka_emika_panda\scene.xml'
$project = '15_RobotSystem'
$out = Join-Path 'runs' ('panda-position-baseline-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
& $py "$project/panda_fixed_position_baseline.py" $scene $out
```

结果目录有 `tracking.csv`、`summary.json` 和无需额外绘图库即可在浏览器打开的 `tracking.svg`。CSV 对每个采样点保留七轴目标、位置和速度。`sample_time_s` 是物理步结束状态时刻，`control_time_s` 是该步开始施加目标的时刻，相差 2 ms；图中横坐标取状态时刻，比较的是该状态与该步已施加的目标。误差按 `actual - applied_target`，逐轴 RMSE、最大绝对误差和末帧绝对误差分别在摘要中记录。

2026-11-17 Agent 使用 MuJoCo 3.17.0、NumPy 2.5.4 和页首固定 Menagerie 模型运行，120 个采样点均有限、无 MuJoCo 警告，采样关节位置未超限。`joint1`、`joint2` 的 RMSE 分别约 `0.01488`、`0.01428 rad`，峰值绝对误差分别约 `0.02458`、`0.01846 rad`；结果见 摘要（本机归档）、[数据](results/20261117_fixed_tracking/tracking.csv) 和 曲线（本机归档）。这是固定轨迹的一次离线仿真检查，不是 IK 末端任务、ROS 闭环通信或 SAC 同协议对照，不代表更广目标上的表现。本人确认完成同协议复跑，实际操作日期待核。原始输出见 [本人轨迹 CSV](results/20261117_fixed_tracking/tracking.csv) 和 本人摘要（本机归档）。本人已于 11 月 22 日独立解释第一轴跟踪误差。见 [跟踪误差说明](经验总结.md#七轴位置控制)。

## 2026-11-19 ROS 2 策略闭环

策略和仿真各自运行在独立的 ROS 2 进程中。`reset_episode` 从 home 返回初态和回合标识。策略根据上一步返回的两轴位置、速度、任务目标与执行器命令推理，再通过 `step_episode` 发送两轴位置增量。仿真服务复用训练环境的 `hold` 执行方式，每个接受的请求最多推进 10 个 2 ms 物理步，并返回下一状态、实际施加的增量和结束标记。仿真同时发布带仿真时间的 `/panda_policy/joint_states`，包含前两个关节。服务用回合标识和步编号拒绝重复、旧回合以及终止后的动作。客户端在服务超时后退出。

```mermaid
flowchart LR
    actor["策略进程<br/>SAC＋HER 推理"]
    sim["仿真进程<br/>MuJoCo 位置执行器"]
    actor -->|"ROS 2 请求：回合、步号、位置增量"| sim
    sim -->|"ROS 2 响应：下一状态、奖励、结束标记"| actor
```

### 验收结果

使用 WSL 2 Ubuntu、ROS 2 Lyrical、MuJoCo 3.17.0、NumPy 2.5.4 和训练 seed 43 的 SAC＋HER checkpoint。[冻结目标协议](results/20261119_panda_policy/protocol.json)使用目标生成 seed 6111201、用例 30101–30124、home 初态、确定性推理、零延迟和 250 步上限。每轴成功阈值是位置误差严格小于 0.01 rad，速度绝对值严格小于 0.05 rad/s。导出的 NumPy actor 在 3021 个原评估状态上与 PyTorch 动作的最大绝对差为 5.36e-7。

ROS 2 路径和同一导出 actor 直接调用 MuJoCo 的路径在 24 个目标上均为 14 次成功、10 次超时，完整 3021 条逐步记录逐字节相同。24 个用例的步数和结束原因也与原 PyTorch 评估一致。30104 在第 16 步成功。30124 在第 250 步超时，第一轴位置误差约 0.01911 rad。通过重复步、旧回合请求和回合结束后请求的拒绝检查。另行复跑原 C++ `JointTrajectory` 录包、状态回放及命令重执行，七轴轨迹比较通过。汇总、哈希和失败末步见 [项目验收数据](results/20261119_panda_policy/evidence.json)，本机完整文件见 ROS 2 逐步验收（本机归档）和 固定轨迹录包验收（本机归档）。

闭环验收使用同步、按请求步进的接口。墙钟节拍下的策略执行、动作延迟和真机接入需各自验收。

### 复现闭环

从仓库根目录用 PowerShell 导出已训练 actor。以下命令沿用本机已有 CPU 环境、固定模型和历史评估产物。请保留打印的输出目录。

```powershell
$work = "runs/panda-policy-export"
$py = Join-Path $work '.venv-panda-sac-cpu/Scripts/python.exe'
$scene = Join-Path $env:LOCALAPPDATA 'EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
$reference = Join-Path $work 'panda-sac-her-paired-20261112-221112/seed-43/sac_her'
$out = Join-Path $work ('panda-policy-export-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$cases = @(30101..30124 | ForEach-Object { '--case'; "$_" })
& $py '15_RobotSystem/panda_policy_actor.py' $scene (Join-Path $reference 'train/sac_cpu.pt') (Join-Path $reference 'heldout/sac.jsonl') $out @cases
if ($LASTEXITCODE -eq 0) { Write-Output $out }
```

在 WSL Bash 中构建接口并运行两个独立进程。将 `actor_dir` 换成上一步输出的 WSL 路径。运行全套目标只进行推理。

```bash
(
set -eo pipefail
source /opt/ros/lyrical/setup.bash
project="$PWD/15_RobotSystem"
scene="$PANDA_SCENE"
ros_python=/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python
actor_dir="$project/results/20261119_panda_policy"
output="$PWD/runs/ros2-policy-$(date +%Y%m%d-%H%M%S)"
cd "$(mktemp -d)"
colcon build --base-paths "$project/ros2_ws/src" --packages-select panda_policy_interfaces
source install/local_setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST ROS_DOMAIN_ID=120
"$ros_python" "$project/panda_ros2_policy.py" serve "$scene" &
sim_pid=$!
trap 'kill -TERM "$sim_pid" 2>/dev/null || true; wait "$sim_pid" 2>/dev/null || true' EXIT
cases=()
for case_id in {30101..30124}; do cases+=(--case "$case_id"); done
"$ros_python" "$project/panda_ros2_policy.py" evaluate "$scene" "$actor_dir/actor.npz" \
  "$project/results/20261119_panda_policy/protocol.json" "$output" "${cases[@]}"
)
```

输出含 `config.json`、`summary.json` 和每例各一份 `-ros2.jsonl`、`-offline.jsonl`。两条路径逐条检查动作、观测、时间、奖励、结束标记与告警。改用其他 checkpoint 时，应先生成同模型、同协议的 PyTorch 参考日志，用相应 checkpoint 和日志重新导出 actor。
