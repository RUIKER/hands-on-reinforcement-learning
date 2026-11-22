# 本机仿真与后训练预检

2026-11-09。环境位于 WSL `~/eip-robot-preflight`。`verl-env` 对应官方依赖与 GPU 检查，`sim-env` 对应官方 LIBERO 单任务 CPU 渲染检查。Ubuntu 为 26.04.1，官方 Quick Start 用 Ubuntu 22.04。使用隔离的 Python 3.10.22，OSMesa 与 FFmpeg 共享库解包到项目目录。WSL 看见一张 RTX 5060 Ti，总显存 16311 MiB，检查时空闲 13931 MiB。

| 检查 | 证据 |
| --- | --- |
| WSL CUDA 驱动 | `cuda_driver_smoke.py` 在 GPU 上执行 PTX 内核并读回 42，计算能力 12.0。 |
| CPU 离屏渲染 | 现有 Panda 场景在 OSMesa 下输出 64×64 RGB，像素范围 13–255。 |
| 官方源码 | `verl-vla` 固定在 `c1de826c7c9256e4d9b6900091e4b0e243d16f5e`。`REACT` 固定在 `eaad1d1b80f31b08c461dd067272e62bd9addb69`。 |
| LIBERO 版本和导入 | 隔离 CPU 环境中，官方检查脚本的导入和 `check_versions()` 已通过。LIBERO 0.1.1、robosuite 1.4.0、MuJoCo 3.3.1、Gymnasium 1.2.2。 |
| LIBERO 任务步进 | 首次缺少场景资产。重试官方安装脚本后，固定提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01` 的归档通过 SHA-256 校验。官方 `check_libero.py` 随后通过，覆盖 LIBERO Spatial 任务 0 的 reset、固定初态、一次动作及双相机 OSMesa 渲染。 |
| PyTorch CUDA | 重试成功，`2.7.1+cu128` 和 `torchvision 0.22.1+cu128` 已装，CUDA 12.8 矩阵乘通过。 |
| ACT GPU 短检查 | LeRobot 0.4.4 使用合成输入生成形状 `[2, 4, 7]` 的动作块，一次参数更新和 checkpoint 重载通过。 |
| REACT 依赖 | 完整 CUDA 锁定环境的大包下载未完成。Python 3.11 CPU 测试环境沿用锁定版本，使用 PyTorch 2.7.1 CPU wheel 和官方 Transformers 补丁，官方小模型测试 13 项通过。 |

`sim-env` 只安装 OSMesa 单步检查所需的依赖。完整安装触发 `egl-probe` 的 CMake 兼容报错，Ubuntu 软件源下载 EGL 开发包超时。TorchCodec 0.4.0 的官方说明支持 FFmpeg 4–7，本机解包的 FFmpeg 8 动态库还需换成兼容版本后再做录制依赖检查。[TorchCodec 0.4.0](https://github.com/meta-pytorch/torchcodec/tree/v0.4.0)

复验可直接运行下列单步脚本。重建环境时，先用官方资产脚本下载并校验归档。

```bash
root="$HOME/eip-robot-preflight"
project="$PWD/15_RobotSystem"
cd "$root/verl-vla" || exit 1
export LIBERO_CONFIG_PATH="$root/libero-cpu-config"
export LD_LIBRARY_PATH="$root/osmesa/usr/lib/x86_64-linux-gnu"
timeout 45 "$root/sim-env/bin/python" -u scripts/install_checks/check_libero.py
```

同一 LIBERO 任务 0 另做了一轮 CPU 短测。沿用官方检查的 seed 0、初态索引 0 和两个 64×64 RGB 相机，用零动作预热 1 步后计时 10 步。步进与双相机渲染合计中位 156.64 ms，范围 151.19–160.95 ms。最后一帧的主相机像素范围为 7–226，腕部相机为 4–232。计时覆盖 `env.step()`，模型推理和加载另测。

## ACT GPU 单批检查

在官方 LeRobot 固定依赖与 `lerobot==0.4.4` 上补装了 `einops==0.8.2`。短检查用 seed 0、批大小 2、两张合成 64×64 图像、8 维状态和 7 维合成动作标签。沿用 ACT 的 VAE 路径，缩小网络与动作块，不下载预训练权重。模型在 RTX 5060 Ti 上生成动作块，完成一次 AdamW 更新，重载后输出完全一致。前向和更新期间 PyTorch 峰值已分配显存 237.01 MiB，只计框架已分配张量。本检查与 LIBERO 步进独立，真实演示数据、视觉预处理、完整训练依赖和策略闭环仍需验证。`uv pip check` 发现配方的 Rerun 版本与 LeRobot 包元数据有冲突，CMake、W&B 和 termcolor 也仍缺。见 [检查代码](act_cuda_smoke.py) 与 [结果](act-cuda-result.json)。

```bash
timeout 40 "$HOME/eip-robot-preflight/verl-env/bin/python" -u \
  "$project"/wsl_preflight/act_cuda_smoke.py
```

## 配方与本机范围

`verl-vla/examples/rl/sac/act/act_sac.yaml` 启用 `td3.enabled: true` 和 `bc_alpha: 0.5`。`vvla-train-sac` 是训练入口，ACT 示例的更新按配置属于 TD3+BC。官方示例为训练和 LIBERO 并行环境各分一张 GPU。本机适配先拆分渲染、模型推理与更新，单 GPU/CPU 渲染组合需单独验证。[verl-vla Quick Start](https://verl-vla.readthedocs.io/en/stable/getting-started/index.html)

REACT 使用 π₀.₅ 的 PyTorch 版本，改变动作头的时间条件和训练损失的采样方式，并在推理中保留滚动缓冲区。论文在 24 GB 的 RTX 4090 上测量推理，仓库提供多 GPU 训练命令，以及无需模型权重的小模型 CPU 测试入口 `pytest src/openpi/react`。本机完整模型运行以兼容 checkpoint、实测峰值显存和测速为准。[REACT 论文](https://arxiv.org/abs/2610.12007) · [官方代码](https://github.com/houlong-xiong/REACT)

## 小规模实验设计

1. 优先验证调度改动。使用同一冻结策略，比较固定执行块与受观测新鲜度触发的短块重规划。固定初态、扰动时刻和控制频率。记录成功情况、观测年龄、扰动到动作响应的时间和动作跳变。若采用 ACT，此阶段称 REACT 启发的执行调度，滚动去噪需采用兼容的 flow 模型。
2. 若有兼容 REACT checkpoint，先做串行滚动推理，再考虑双解耦。每回合清空缓冲区，日志记录计划生成时刻、观测时间、实际执行动作和重置边界。完整 REACT 依赖动作头的逐位置时间条件及与滚动推理匹配的训练。
3. 若只增加一个学习模块，冻结基策略，让小型 SAC 在动作块边界输出有界残差。残差的坐标、单位和限幅与环境动作一致。回放记录基动作块、实际动作、剩余队列、观测年龄、持续步数和结束状态；按块累加奖励，并按实际执行步数折扣。对照冻结基策略和零残差，单独归因调度和学习的影响。
4. 固定离线示范与单批 rollout 时，先保留官方 TD3+BC 作为可追溯对照。可以在线采样且需要学习有界残差时，先保留 SAC。两种数据条件分开记录。正式运行前由短检查测出吞吐和显存，按配置给出命令、预计耗时和新建输出目录，再由用户决定启动。[残差强化学习](https://arxiv.org/abs/1812.03201) · [TD3+BC](https://arxiv.org/abs/2106.06860)

## Panda 调度对照实测

Agent 在 Windows CPU 上运行 MuJoCo 3.17.0、NumPy 2.5.4。复用现有 `PandaTwoJointEnv` 和 Menagerie 模型提交 `98a70c569c4c1b6da467a20012f766ee1f8dc521`。每组从同一 `home` 开始，无随机采样。每步 20 ms，固定规划长度 8 步。两组共用同一个规则规划器，每步将目标与当前命令的差限幅到 ±0.02 rad。固定组执行完整动作块，事件组在观测到目标切换时提前生成新块。两组都在目标实际切换后第 2 个 tick 收到新目标。

实际目标从 `(0.1, 0.1)` 切换至 `(-0.08, -0.05)` rad，切换在第 6、9 或 13 个 tick 开始处。tick 从 0 计数。响应延迟从实际切换开始，量到首次执行朝新目标方向的命令增量。两个关节的误差和速度由环境原有成功判据检查。动作跳变是相邻两步实际命令增量之差在各轴的最大绝对值，计入初始零增量。统计结果见 [摘要](panda-schedule-run/summary.json) 和 [逐步轨迹](panda-schedule-run/traces.json)。

| 目标切换 tick | 响应延迟 固定 / 事件 | 到达步数 固定 / 事件 | 计划次数 固定 / 事件 |
| --- | --- | --- | --- |
| 无 | — | 18 / 18 | 3 / 3 |
| 6 | 40 / 40 ms | 30 / 30 | 4 / 4 |
| 9 | 140 / 40 ms | 39 / 34 | 5 / 5 |
| 13 | 60 / 40 ms | 39 / 38 | 5 / 5 |

两种调度在所有设定中都成功，峰值命令增量跳变均为 0.02 rad。复跑两份输出逐字节相同。切换恰好在块边界可见时，两个调度一致。其余两组省掉的仿真时间对应等待块结束的时间。该对照使用关节目标信号与规则策略，结果仅覆盖人为设定的观测延迟与调度。模型实际推理时延、感知误差、真实运动平顺性和 REACT 的滚动去噪尚未评估。

从 EIP 目录运行，在 `--output` 指定新目录。

```powershell
$project = '15_RobotSystem'
$scene = $env:PANDA_SCENE
$env:PYTHONDONTWRITEBYTECODE = '1'
python 15_RobotSystem/wsl_preflight/panda_chunk_schedule.py --project $project --scene $scene --output runs/panda-schedule-new
```

## LIBERO 示范与 REACT 反馈实测

- LIBERO Spatial 任务 0，固定 HDF5 修订 `e434c228d52f21e37c6f45046a6476a8fa362b8d`，文件 SHA-256 `ff6f26121653c77280eb40a38773a74141c11a8509f3466058cb56dd2cc60ead`，本机 `~/eip-robot-preflight/demo/task0.hdf5`。`libero_demo_smoke.py` 以 seed 0 和 `demo_0` 首帧状态复位，逐条回放 98 个原始动作并检测到成功。首两帧的双相机图像、8 维状态和 4×7 动作标签已导出到 `demo/act-demo0.npz`。`act_cuda_smoke.py --batch` 用这两个样本完成一次 GPU 更新及重载。见 [示范回放](libero-demo-result.json) 和 [ACT 更新](act-demo0-result.json)。[固定数据源](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/blob/e434c228d52f21e37c6f45046a6476a8fa362b8d/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5)
- `react_feedback_smoke.py` 加载固定提交的官方 `ReactPolicy` 和 `RollingBuffer`。动作专家使用测试替身，只读取 LIBERO 末端位置，未加载 π₀.₅。相同 seed 和初态下，首块动作相同。执行后末端移动约 20.0 mm，把更新观测与初始旧观测分别传给下一次 `infer()`，第二块动作元素的最大绝对差为 0.00692。测试采用同步模式和纯噪声 warm-up，避免把 LIBERO 的 8 维状态直接当作 7 维动作。此结果只验证执行反馈和缓冲接口。见 [反馈结果](react-feedback-result.json)。

## REACT 官方测试与 π₀.₅ 配对准备

`pytest -q src/openpi/react` 在 WSL `~/eip-robot-preflight/react-cpu-tests` 报告 13 项通过，用时 62.13 秒。单独加载锁定版本 JAX 0.5.3 的 CUDA 插件并复用已有 CUDA 12.8 动态库，RTX 5060 Ti 上的 16×16 矩阵乘通过。见 [官方测试日志](react-official-test.log)。

`pi05_libero_feedback.py` 已验证 LIBERO Spatial 任务 0 能输出双 256×256 相机、8 维状态与指令。完整配对将执行官方 `pi05_libero` 首块的前 5 个动作，再用同一采样噪声分别传入执行前与执行后的观测。两条下一块预测保存为 NPZ，并报告首动作、前五个动作和全块的最大绝对差。

官方 checkpoint 共约 12.44 GB。最后一个分片通过 GCS JSON 媒体入口恢复下载，WSL `~/eip-robot-preflight/pi05-libero` 的 16 个文件大小均匹配官方清单。标准 π₀.₅ LIBERO 权重在 RTX 5060 Ti 上完成配对短测。执行前 5 个动作后，末端移动约 25.76 mm。使用同一采样噪声时，两条下一动作块均为 10×7，最大绝对差 0.75548，出现在第 1 个动作的第 1 维。该维度从旧观测的 0.12921 变为新观测的 0.88468。此次实测覆盖一个初态和一次反馈，REACT 官方滚动推理需要适配权重。见 [结果](pi05-feedback-run/summary.json) 和 [原始动作块](pi05-feedback-run/next_chunks.npz)。[openpi 官方仓库](https://github.com/Physical-Intelligence/openpi) · [REACT 官方仓库](https://github.com/houlong-xiong/REACT)

## WSL 配置迁移

以上上游仓库以固定提交外部获取。WSL 中的 LIBERO 配置使用安装目录下的 benchmark、BDDL、初态、数据和资产路径。迁移后可用当前虚拟环境的导入位置生成同结构的配置。

~~~bash
"$HOME/eip-robot-preflight/sim-env/bin/python" 15_RobotSystem/wsl_preflight/write_libero_config.py "$HOME/eip-robot-preflight/libero-config"
export LIBERO_CONFIG_PATH="$HOME/eip-robot-preflight/libero-config"
~~~

[REACT CPU 预检锁定表](react-cpu-lock.txt)来自当时 WSL 的依赖导出，使用时与上方固定上游提交及其补丁说明配套。模型权重和示范数据通过各自的发布方获取。
