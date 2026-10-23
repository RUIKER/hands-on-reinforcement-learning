# TRPO on Pendulum

这是 TRPO 的连续动作版本，使用 `Pendulum-v1` 环境和高斯策略。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

## 和 CartPole 版本的区别

- 策略网络输出高斯分布的 `mean` 和 `std`。
- 动作从 `Normal` 分布采样，并裁剪到环境动作边界。
- log probability 对动作维度求和。
- KL 散度对动作维度求和。
- 使用 reward scaling 改善 critic 的数值范围。

TRPO 的 GAE、共轭梯度、Hessian-vector product 和线搜索步骤与离散版本保持一致。
