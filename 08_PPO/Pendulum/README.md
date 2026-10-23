# PPO on Pendulum

在 `Pendulum-v1` 上使用连续动作 PPO。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

策略网络输出高斯分布的均值和标准差，动作采样后裁剪到环境边界。PPO 的 log probability 会沿动作维度求和，reward 会在 critic 更新时进行缩放。
