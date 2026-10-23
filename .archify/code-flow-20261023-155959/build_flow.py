#!/usr/bin/env python3
"""Build an interactive per-file code-flow page from the extracted block atlas."""
from __future__ import annotations

import json
import os
import re

ROOT = "/Users/yanrui/Downloads/硕士/gpulab/Reinforcement Learning/code"
ARCHIVE = os.path.join(ROOT, ".archify")
if not os.path.isdir(ARCHIVE):
    ARCHIVE = os.path.join(ROOT, "archify")
ATLAS = os.path.join(ARCHIVE, "code-atlas-20261023-153518")
BASE = os.path.join(ARCHIVE, "code-flow-20261023-155959")
OUT = os.path.join(BASE, "code-flow.html")
ROOT_OUT = os.path.join(ROOT, "code-flow.html")

META = [
    ("01", "01_Cliff Walking", "价值型", "表格型 Q-learning / SARSA"),
    ("02", "02_DQN", "价值型", "最小 DQN 练习"),
    ("03", "03_Double DQN、Dueling DQN", "价值型", "Double / Dueling 统一 pipeline"),
    ("04", "04_策略梯度算法_Reinforce", "策略梯度", "REINFORCE"),
    ("05", "05_Mountain Car", "策略梯度", "线性 Actor-Critic"),
    ("06", "06_Mountain Car PyTorch", "策略梯度", "MLP Actor-Critic"),
    ("07", "07_TRPO", "策略梯度", "信赖域策略优化"),
    ("08", "08_PPO", "策略梯度", "裁剪代理目标"),
    ("09", "09_DDPG", "连续控制", "确定性 actor-critic"),
    ("10", "10_TD3", "连续控制", "双 critic + 延迟更新"),
    ("11", "11_SAC", "连续控制", "最大熵 + 自动温度"),
]

PSEUDO: dict[str, str] = {
    "QTable.take_action": """if rand() < epsilon:
    return random_action()
return argmax_a Q[s, a]""",
    "SarsaTable.take_action": """if rand() < epsilon:
    return random_action()
return argmax_a Q[s, a]""",
    "QTable.max_q": """return max_a Q[s, a]""",
    "QTable.update": """# Q-learning: 目标动作由 max 选出
s, a, r, s2, done = transition
if done:
    target = r
else:
    target = r + gamma * max_a2 Q[s2, a2]
Q[s, a] += lr * (target - Q[s, a])""",
    "SarsaTable.update": """# SARSA: 目标动作是实际执行的 a2
s, a, r, s2, a2, done = transition
if done:
    target = r
else:
    target = r + gamma * Q[s2, a2]
Q[s, a] += lr * (target - Q[s, a])""",
    "DQN.update": """batch = replay.sample(B)
q = Q_online(s).gather(1, a)
if dqn_type == "DoubleDQN":
    a2 = argmax_a Q_online(s')
    q2 = Q_target(s').gather(1, a2)
else:
    q2 = max_a Q_target(s')
target = r + gamma * q2 * (1 - done)
loss = MSE(q, target)
update Q_online
every C steps: Q_target <- Q_online""",
    "REINFORCE.take_action": """logits = pi(s)
dist = Categorical(softmax(logits))
a = dist.sample()
return a.item()""",
    "REINFORCE.update": """G = 0
loss = 0
for t = T-1 ... 0:
    G = r[t] + gamma * G
    logp = log pi(a[t] | s[t])
    loss += -logp * G
backprop(loss)
optimizer.step()""",
    "ValueEstimator.update": """v = w . x
error = target - v
w += lr * error * x
return error""",
    "GaussianPolicy.forward": """return self.mean(x), self.standard_deviation(x)""",
    "GaussianPolicy.mean": """return w . x""",
    "GaussianPolicy.log_standard_deviation": """return v . x""",
    "GaussianPolicy.standard_deviation": """return exp(v . x)""",
    "GaussianPolicy.update": """mu = w . x
sigma = softplus(v . x)
d_mu = (a - mu) / sigma^2 * x
d_logs = ((a - mu)^2 / sigma^2 - 1) * x
w += clip(lr * A * d_mu, -0.05, 0.05)
v += clip(lr * (A * d_logs + beta * x), -0.05, 0.05)""",
    "DDPG.take_action": """a = actor(s)
if explore:
    a = a + noise.sample()
return clip(a, action_low, action_high)""",
    "DDPG.reset_noise": """noise.reset()""",
    "DDPG.soft_update": """for target_param, online_param:
    target_param = (1 - tau) * target_param
                 + tau * online_param""",
    "DDPG.update": """a2 = mu_target(s')
y = r + gamma * (1 - done) * Q_target(s', a2)
Lq = MSE(Q(s, a), y)
update Q
Lpi = -mean(Q(s, mu(s)))
update mu
soft_update(mu, mu_target)
soft_update(Q, Q_target)""",
    "TD3.calc_target": """a2 = mu_target(s')
eps = randn_like(a2) * target_noise * scale
eps = clip(eps, -noise_clip * scale, noise_clip * scale)
a2 = clip(a2 + eps, low, high)
y = r + gamma * (1 - done)
    * min(Q1_target(s', a2), Q2_target(s', a2))
return y""",
    "TD3.update": """y = calc_target(r, s', done)
Lq = MSE(Q1(s, a), y) + MSE(Q2(s, a), y)
update Q1, Q2
update_count += 1
if update_count % policy_delay == 0:
    Lpi = -mean(Q1(s, mu(s)))
    update mu
    soft_update(mu, mu_target)
    soft_update(Q1, Q1_target)
    soft_update(Q2, Q2_target)""",
    "TD3.take_action": """a = actor(s)
if explore:
    a = a + noise.sample()
return clip(a, action_low, action_high)""",
    "TD3.reset_noise": """noise.reset()""",
    "TD3.soft_update": """for target_param, online_param:
    target_param = (1 - tau) * target_param
                 + tau * online_param""",
    "SAC.calc_target": """a2, logp2 = mu_target(s')
q2 = min(Q1_target(s', a2), Q2_target(s', a2))
y = r + gamma * (1 - done) * (q2 - alpha * logp2)
return y""",
    "SACContinuous.calc_target": """a2, logp2 = mu_target(s')
q2 = min(Q1_target(s', a2), Q2_target(s', a2))
y = r + gamma * (1 - done) * (q2 - alpha * logp2)
return y""",
    "SAC.update": """y = calc_target(r, s', done)
Lq1 = MSE(Q1(s, a), y); update Q1
Lq2 = MSE(Q2(s, a), y); update Q2
a2, logp = mu(s)                  # reparameterized
Lpi = mean(alpha * logp - min(Q1(s, a2), Q2(s, a2)))
update mu
Lalpha = -mean(log_alpha * (logp + target_entropy))
update log_alpha
soft_update(Q1, Q1_target); soft_update(Q2, Q2_target)""",
    "SACContinuous.update": """y = calc_target(r, s', done)
Lq1 = MSE(Q1(s, a), y); update Q1
Lq2 = MSE(Q2(s, a), y); update Q2
a2, logp = mu(s)                  # reparameterized
Lpi = mean(alpha * logp - min(Q1(s, a2), Q2(s, a2)))
update mu
Lalpha = -mean(log_alpha * (logp + target_entropy))
update log_alpha
soft_update(Q1, Q1_target); soft_update(Q2, Q2_target)""",
    "SAC.soft_update": """for target_param, online_param:
    target_param = (1 - tau) * target_param
                 + tau * online_param""",
    "TRPO._log_probs": """dist = self._distribution(s)
return dist.log_prob(a).sum(dim=-1, keepdim=True)""",
    "TRPOContinuous._log_probs": """dist = self._distribution(s)
return dist.log_prob(a).sum(dim=-1, keepdim=True)""",
    "TRPO._distribution": """mu = self.actor(s)
sigma = self.actor.log_std.exp()
return Normal(mu, sigma)""",
    "TRPOContinuous._distribution": """mu = self.actor(s)
sigma = self.actor.log_std.exp()
return Normal(mu, sigma)""",
    "PPO._distribution": """mu = self.actor(s)
sigma = self.actor.log_std.exp()
return Normal(mu, sigma)""",
    "PPOContinuous._distribution": """mu = self.actor(s)
sigma = self.actor.log_std.exp()
return Normal(mu, sigma)""",
    "TRPO.compute_surrogate_obj": """ratio = exp(logpi(a|s) - logpi_old(a|s))
return mean(ratio * advantage)""",
    "TRPOContinuous.compute_surrogate_obj": """ratio = exp(logpi(a|s) - logpi_old(a|s))
return mean(ratio * advantage)""",
    "TRPO.conjugate_gradient": """x = 0; r = g; p = g
repeat at most 10:
    Hp = H(p) + damping * p
    alpha = (r . r) / (p . Hp)
    x += alpha * p
    r_new = r - alpha * Hp
    if r_new . r_new < 1e-10: break
    beta = (r_new . r_new) / (r . r)
    p = r_new + beta * p
    r = r_new
return x""",
    "TRPOContinuous.conjugate_gradient": """x = 0; r = g; p = g
repeat at most 10:
    Hp = H(p) + damping * p
    alpha = (r . r) / (p . Hp)
    x += alpha * p
    r_new = r - alpha * Hp
    if r_new . r_new < 1e-10: break
    beta = (r_new . r_new) / (r . r)
    p = r_new + beta * p
    r = r_new
return x""",
    "TRPO.hessian_matrix_vector_product": """kl = mean(KL(old_dist || new_dist))
g = grad(kl, actor_params, create_graph=True)
return grad(g . vector, actor_params) + damping * vector""",
    "TRPOContinuous.hessian_matrix_vector_product": """kl = mean(KL(old_dist || new_dist))
g = grad(kl, actor_params, create_graph=True)
return grad(g . vector, actor_params) + damping * vector""",
    "TRPO.line_search": """old_obj = surrogate(old_params)
for i = 0 ... 9:
    candidate = old_params + (0.5^i) * step
    if surrogate(candidate) > old_obj
       and KL(old || candidate) < delta:
        return candidate
return old_params""",
    "TRPOContinuous.line_search": """old_obj = surrogate(old_params)
for i = 0 ... 9:
    candidate = old_params + (0.5^i) * step
    if surrogate(candidate) > old_obj
       and KL(old || candidate) < delta:
        return candidate
return old_params""",
    "TRPO.policy_learn": """g = grad(surrogate)
step = conjugate_gradient(H, g)
curvature = step . H(step)
step *= sqrt(2 * delta / max(curvature, 1e-8))
new_params = line_search(step)
copy new_params into actor""",
    "TRPOContinuous.policy_learn": """g = grad(surrogate)
step = conjugate_gradient(H, g)
curvature = step . H(step)
step *= sqrt(2 * delta / max(curvature, 1e-8))
new_params = line_search(step)
copy new_params into actor""",
    "TRPO.update": """scaled_r = (r + shift) / scale
td = scaled_r + gamma * V(s') * (1 - done)
A = GAE(td, gamma, lambda)
old_logp = logpi_old(a | s).detach()
old_dist = Normal(mu_old(s), sigma_old(s))
update critic: MSE(V(s), td)
policy_learn(s, a, old_dist, old_logp, A)""",
    "TRPOContinuous.update": """scaled_r = (r + shift) / scale
td = scaled_r + gamma * V(s') * (1 - done)
A = GAE(td, gamma, lambda)
old_logp = logpi_old(a | s).detach()
old_dist = Normal(mu_old(s), sigma_old(s))
update critic: MSE(V(s), td)
policy_learn(s, a, old_dist, old_logp, A)""",
    "PPO.update": """scaled_r = (r + shift) / scale
td = scaled_r + gamma * V(s') * (1 - done)
A = GAE(td, gamma, lambda)
old_logp = logpi_old(a | s).detach()
for epoch in 1 ... epochs:
    ratio = exp(logpi(a|s) - old_logp)
    clipped = clip(ratio, 1 - eps, 1 + eps)
    Lpi = -mean(min(ratio * A, clipped * A))
    Lv = MSE(V(s), td)
    update actor and critic""",
    "PPOContinuous.update": """scaled_r = (r + shift) / scale
td = scaled_r + gamma * V(s') * (1 - done)
A = GAE(td, gamma, lambda)
old_logp = logpi_old(a | s).detach()
for epoch in 1 ... epochs:
    ratio = exp(logpi(a|s) - old_logp)
    clipped = clip(ratio, 1 - eps, 1 + eps)
    Lpi = -mean(min(ratio * A, clipped * A))
    Lv = MSE(V(s), td)
    update actor and critic""",
    "PPO._log_probs": """dist = Normal(mu(s), sigma(s))
return dist.log_prob(a).sum(dim=-1, keepdim=True)""",
    "PPOContinuous._log_probs": """dist = Normal(mu(s), sigma(s))
return dist.log_prob(a).sum(dim=-1, keepdim=True)""",
    "ReplayBuffer.add": """buffer.append((s, a, r, s2, done))
if len(buffer) > capacity:
    buffer.popleft()""",
    "ReplayBuffer.sample": """batch = random.sample(buffer, batch_size)
return zip(*batch)   # states, actions, rewards, next_states, dones""",
    "GaussianNoise.sample": """return sigma * randn(action_dim).astype(float32)""",
    "NormalNoise.sample": """return sigma * randn(action_dim)""",
    "OUNoise.sample": """x = x + theta * (mu - x) + sigma * randn(action_dim)
return x""",
    "smooth_target_action": """eps = randn_like(a) * target_noise * action_scale
eps = clip(eps, -noise_clip * action_scale,
                noise_clip * action_scale)
return clip(a + eps, action_low, action_high)""",
    "ValueNet.forward": """h = relu(fc1(s))
h = relu(fc2(h))
return fc3(h)   # V(s)""",
    "ValueEstimator.predict": """return w . x""",
    "Actor.distribution": """return Categorical(softmax(fc(s)))""",
    "Actor.sample": """return self.distribution(s).sample()""",
    "normalize_state": """return (state - low) / (high - low)""",
    "continuous_to_discrete": """idx = floor((x - low) / (high - low) * n)
return clip(idx, 0, n - 1)""",
    "RBFStateFeaturizer.transform": """for each center c:
    feature[i] = exp(-||s - c||^2 / (2 * sigma^2))
return features""",
    "moving_average": """return [mean(x[max(0, i-w+1):i+1]) for i in range(len(x))]""",
    "set_seed": """random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)""",
    "make_environment": """env = gym.make(env_name)
env.reset(seed=seed)
return env""",
    "Actor.forward": """h = relu(W1 s + b1)
h = relu(W2 h + b2)
u = tanh(W3 h + b3)
return action_bias + action_scale * u""",
    "Critic.forward": """x = concat(s, a)
h = relu(W1 x + b1)
h = relu(W2 h + b2)
return W3 h + b3""",
    "Qnet.forward": """h = relu(fc1(s))
h = relu(fc2(h))
return fc3(h)   # 每个离散动作一个 Q 值""",
    "DuelingQnet.forward": """h = relu(fc1(s)); h = relu(fc2(h))
V = value_head(h)
A = advantage_head(h)
return V + A - mean(A, dim=1, keepdim=True)""",
    "PolicyNet.forward": """h = relu(fc1(s)); h = relu(fc2(h))
return softmax(fc3(h))   # 离散动作概率""",
    "PolicyNetContinuous.forward": """h = relu(fc1(s)); h = relu(fc2(h))
mu = tanh(mu_head(h))
sigma = softplus(sigma_head(h)) + 1e-5
return mu, sigma""",
    "QValueNet.forward": """h = relu(fc1(s)); h = relu(fc2(h))
return fc3(h)   # V(s)""",
    "QValueNetContinuous.forward": """x = concat(s, a)
h = relu(fc1(x)); h = relu(fc2(h))
return fc3(h)   # Q(s, a)""",
    "GaussianPolicy.sample": """mu = w . x
sigma = softplus(v . x)
a = N(mu, sigma).sample()
return clip(a, action_low, action_high)""",
    "DQN.take_action": """if rand() < epsilon:
    return random_action()
return argmax_a Q(s, a)""",
    "DQN.max_q_value": """return max_a Q(s, a)""",
    "SAC.take_action": """with no_grad:
    a, _ = mu(s)
return a""",
    "SACContinuous.take_action": """with no_grad:
    a, _ = mu(s)
return a""",
    "PPO.take_action": """dist = Categorical(pi(s))
a = dist.sample()
return a.item()""",
    "PPOContinuous.take_action": """mu, sigma = policy(s)
a = Normal(mu, sigma).sample()
return a""",
    "TRPO.take_action": """dist = policy_dist(s)
a = dist.sample()
return clip(a, low, high)""",
    "TRPOContinuous.take_action": """dist = policy_dist(s)
a = dist.sample()
return clip(a, low, high)""",
    "train_off_policy_agent": """for episode in 1 ... num_episodes:
    s = env.reset()
    done = False
    while not done:
        a = agent.take_action(s, explore=True)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        buffer.add(s, a, r, s2, terminated)
        if buffer.size >= minimal_size:
            agent.update(buffer.sample(batch_size))
        s = s2""",
    "train_on_policy_agent": """for episode in 1 ... num_episodes:
    rollout = []
    s = env.reset()
    done = False
    while not done:
        a = agent.take_action(s)
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        rollout.append((s, a, r, s2, terminated))
        s = s2
    agent.update(rollout)""",
    "evaluate_policy": """for i in 1 ... n_episodes:
    s, _ = env.reset(seed=seed + i)
    done = False
    total = 0
    while not done:
        a = agent.take_action(s, explore=False)
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        total += r
    returns.append(total)
return mean(returns), std(returns)""",
    "compute_advantage": """adv = 0
for t = T-1 ... 0:
    adv = td_delta[t] + gamma * lambda * adv
    advantage[t] = adv
return advantage""",
    "n_step_targets": """for t in range(T):
    G = sum_{k=0}^{n-1} gamma^k * r[t+k]
    if not done[t+n-1]:
        G += gamma^n * V(s[t+n])
    targets[t] = G""",
}


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def parse_line(value) -> tuple[int, int]:
    text = str(value or "1")
    if "-" in text:
        a, b = text.split("-", 1)
        try:
            return max(1, int(a)), max(1, int(b))
        except ValueError:
            return 1, 1
    try:
        n = max(1, int(text))
        return n, n
    except ValueError:
        return 1, 1


def kind_class(kind: str) -> str:
    return {
        "导入与常量": "const",
        "类": "cls",
        "方法": "method",
        "函数": "func",
        "入口": "entry",
    }.get(kind, "func")


STAGE_RULES = {
    "config.py": [],
    "environment.py": ["reset", "step", "render", "close"],
    "networks.py": ["__init__", "forward", "sample", "distribution"],
    "agents.py": ["__init__", "take_action", "update", "calc_target",
                  "policy_learn", "line_search", "conjugate_gradient",
                  "soft_update"],
    "trainer.py": ["train_off_policy_agent", "train_on_policy_agent",
                   "evaluate_policy", "train"],
    "rl_utils.py": ["__init__", "add", "sample", "moving_average"],
    "noise.py": ["sample", "smooth_target_action", "reset"],
    "main.py": ["main"],
    "visualization.py": ["plot_episode_stats", "plot_returns",
                         "plot_training_results"],
    "features.py": ["__init__", "transform", "feature_count"],
    "estimators.py": ["__init__", "predict", "update"],
    "code.py": ["main"],
    "practice.py": ["__init__", "take_action", "update", "calc_target",
                    "policy_learn", "soft_update"],
}


def pick_stages(blocks: list[dict]) -> list[int]:
    base = os.path.basename(blocks[0]["_path"]) if blocks else ""
    rules = STAGE_RULES.get(base, [])
    chosen: list[int] = []
    for rule in rules:
        for i, b in enumerate(blocks):
            if b["kind"] in {"类", "导入与常量", "入口"}:
                continue
            if b["name"].split(".")[-1] == rule:
                if i not in chosen:
                    chosen.append(i)
                break
    if not chosen:
        if base == "config.py":
            for i, b in enumerate(blocks):
                if b["kind"] == "导入与常量":
                    chosen.append(i)
                    break
        else:
            for i, b in enumerate(blocks):
                if b["kind"] in {"函数", "方法"}:
                    chosen.append(i)
                    if len(chosen) >= 4:
                        break
    for i, b in enumerate(blocks):
        if b["kind"] == "入口" and i not in chosen:
            chosen.append(i)
            break
    return chosen[:6]


def build() -> dict:
    roots = []
    search = []
    totals = {"files": 0, "blocks": 0, "classes": 0, "methods": 0, "functions": 0}
    for rid, rdir, branch, short in META:
        path = os.path.join(ATLAS, "data", f"{rid}.json")
        payload = json.load(open(path, encoding="utf-8"))
        files = []
        for f in payload.get("files", []):
            rel = f["path"].replace("\\", "/")
            full = os.path.join(ROOT, rel)
            missing_src = not os.path.exists(full)
            source_lines = (
                [] if missing_src
                else open(full, encoding="utf-8").read().splitlines()
            )
            blocks = []
            for b in f.get("blocks", []):
                start, end = parse_line(b.get("line"))
                end = min(end, len(source_lines))
                src = source_lines[start - 1:end] if start <= end else []
                name = b.get("name", "")
                blocks.append({
                    "kind": b.get("kind", ""),
                    "kc": kind_class(b.get("kind", "")),
                    "name": name,
                    "line": str(b.get("line", "")),
                    "logic": clean(b.get("logic", "")),
                    "srcStart": start,
                    "src": src,
                    "pseudo": PSEUDO.get(name, "").strip(),
                })
                totals["blocks"] += 1
                if b.get("kind") == "类":
                    totals["classes"] += 1
                elif b.get("kind") == "方法":
                    totals["methods"] += 1
                elif b.get("kind") == "函数":
                    totals["functions"] += 1
            # attach path for stage selection before dropping the helper key
            for b in blocks:
                b["_path"] = rel
            stages = pick_stages(blocks)
            for b in blocks:
                b.pop("_path", None)

            class_groups = []
            consumed_methods = set()
            for ci, b in enumerate(blocks):
                if b["kind"] != "类":
                    continue
                mids = []
                prefix = b["name"] + "."
                for mi, m in enumerate(blocks):
                    if m["kind"] == "方法" and m["name"].startswith(prefix):
                        mids.append(mi)
                        consumed_methods.add(mi)
                class_groups.append({"classIdx": ci, "methodIdxs": mids})
            func_idxs = [i for i, b in enumerate(blocks) if b["kind"] == "函数"]
            entry_idxs = [i for i, b in enumerate(blocks) if b["kind"] == "入口"]
            orphan_methods = [
                i for i, b in enumerate(blocks)
                if b["kind"] == "方法" and i not in consumed_methods
            ]
            const_idx = next(
                (i for i, b in enumerate(blocks) if b["kind"] == "导入与常量"), None
            )
            files.append({
                "path": rel,
                "rootId": rid,
                "rootDir": rdir,
                "name": os.path.basename(rel),
                "dir": os.path.dirname(rel),
                "purpose": clean(f.get("purpose", "")),
                "lines": f.get("lines", len(source_lines)),
                "missingSrc": missing_src,
                "blocks": blocks,
                "stages": stages,
                "classGroups": class_groups,
                "funcIdxs": func_idxs + orphan_methods,
                "entryIdxs": entry_idxs,
                "constIdx": const_idx,
            })
            totals["files"] += 1
            for i, b in enumerate(blocks):
                search.append({
                    "path": rel,
                    "root": rdir,
                    "idx": i,
                    "kind": b["kind"],
                    "name": b["name"],
                    "logic": b["logic"],
                })
        roots.append({
            "id": rid,
            "dir": rdir,
            "branch": branch,
            "short": short,
            "summary": clean(payload.get("summary", "")),
            "files": files,
        })
    return {
        "title": "强化学习代码流：文件、函数与伪代码",
        "stats": totals,
        "roots": roots,
        "search": search,
    }


def main() -> None:
    data = build()
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = TEMPLATE.replace("__DATA__", payload)
    for target in (OUT, ROOT_OUT):
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(html)
        print("wrote", target)
    print("bytes", os.path.getsize(OUT))
    print("stats", data["stats"])


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>强化学习代码流：文件、函数与伪代码</title>
<style>
:root{
  --bg:#070b11; --panel:#0d141e; --panel2:#111b27; --panel3:#152131;
  --line:#1e2c3b; --line2:#2a3b4e; --text:#e6edf3; --muted:#8b9bad; --dim:#5d6b7a;
  --cyan:#22d3ee; --violet:#a78bfa; --amber:#fbbf24; --emerald:#34d399;
  --rose:#fb7185; --slate:#94a3b8;
  --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif;
  color-scheme:dark;
}
*{box-sizing:border-box}
html,body{height:100%}
body{
  margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);
  -webkit-font-smoothing:antialiased;
}
body::before{
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:
    linear-gradient(rgba(34,211,238,.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba(34,211,238,.035) 1px,transparent 1px),
    radial-gradient(900px 500px at 15% -10%,rgba(34,211,238,.10),transparent 60%),
    radial-gradient(800px 450px at 90% 0%,rgba(167,139,250,.08),transparent 60%);
  background-size:32px 32px,32px 32px,100% 100%,100% 100%;
}
button{font:inherit;color:inherit}
button:focus-visible,input:focus-visible,select:focus-visible{outline:2px solid var(--cyan);outline-offset:2px}
.mono{font-family:var(--mono)}
.topbar{
  position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:22px;
  min-height:68px;padding:10px 24px;background:rgba(7,11,17,.88);
  border-bottom:1px solid var(--line);backdrop-filter:blur(14px);
}
.brand{display:flex;align-items:center;gap:12px;min-width:300px}
.brand-mark{
  width:40px;height:40px;border:1px solid rgba(34,211,238,.55);border-radius:10px;
  display:grid;place-items:center;font-family:var(--mono);font-weight:700;
  color:var(--cyan);background:rgba(34,211,238,.08);box-shadow:0 0 24px rgba(34,211,238,.16);
}
.brand h1{font-size:16px;margin:0;letter-spacing:.02em}
.brand p{margin:2px 0 0;color:var(--muted);font-size:12px}
.search-wrap{position:relative;flex:1;max-width:620px}
.search-wrap input{
  width:100%;height:40px;border-radius:10px;border:1px solid var(--line2);
  background:rgba(13,20,30,.85);color:var(--text);padding:0 14px 0 38px;
  font-size:13px;
}
.search-wrap::before{
  content:"⌕";position:absolute;left:13px;top:50%;transform:translateY(-55%);
  color:var(--dim);font-size:18px;pointer-events:none;
}
.search-results{
  position:absolute;top:46px;left:0;right:0;max-height:52vh;overflow:auto;
  background:#0c131d;border:1px solid var(--line2);border-radius:12px;
  box-shadow:0 22px 60px rgba(0,0,0,.55);padding:6px;z-index:45;
}
.result{
  display:block;width:100%;text-align:left;background:none;border:0;
  border-radius:8px;padding:9px 10px;cursor:pointer;
}
.result:hover{background:rgba(34,211,238,.08)}
.result-path{display:block;color:var(--dim);font-family:var(--mono);font-size:10.5px}
.result-name{display:block;font-size:13px;margin-top:2px}
.result-kind{color:var(--cyan);font-size:10px;margin-right:6px}
.search-empty{padding:12px;color:var(--muted);font-size:12px}
.mobile-select{display:none}
.shell{position:relative;z-index:1;display:grid;grid-template-columns:292px minmax(0,1fr);height:calc(100vh - 68px)}
.sidebar{
  overflow:auto;border-right:1px solid var(--line);padding:14px 10px 50px;
  background:rgba(10,16,24,.58);
}
.sidebar-head{
  display:flex;justify-content:space-between;color:var(--dim);
  font-family:var(--mono);font-size:10.5px;letter-spacing:.12em;text-transform:uppercase;
  padding:4px 10px 10px;
}
.root-group{margin:2px 0}
.root-head{
  width:100%;display:flex;align-items:center;gap:8px;background:none;border:0;
  padding:9px 8px;border-radius:8px;cursor:pointer;text-align:left;color:var(--muted);
}
.root-head:hover{background:rgba(255,255,255,.035)}
.root-head.active{background:rgba(34,211,238,.075);color:var(--text)}
.root-id{font-family:var(--mono);font-size:10.5px;color:var(--cyan);width:20px}
.root-name{flex:1;font-size:12px;line-height:1.35}
.root-count{font-family:var(--mono);font-size:10px;color:var(--dim)}
.root-files{display:none;padding:2px 0 6px 26px}
.root-group.open .root-files{display:block}
.file-item{
  width:100%;display:flex;justify-content:space-between;gap:8px;align-items:center;
  background:none;border:1px solid transparent;border-radius:7px;padding:6px 8px;
  cursor:pointer;text-align:left;color:var(--muted);
}
.file-item:hover{background:rgba(255,255,255,.035);color:var(--text)}
.file-item.active{background:rgba(34,211,238,.09);border-color:rgba(34,211,238,.22);color:var(--text)}
.file-item span:first-child{font-family:var(--mono);font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.file-blocks{font-family:var(--mono);font-size:9.5px;color:var(--dim)}
.main{overflow:auto;padding:28px 34px 90px;min-width:0}
.crumbs{display:flex;gap:8px;align-items:center;color:var(--dim);font-family:var(--mono);font-size:11px;margin-bottom:12px}
.crumbs b{color:var(--muted);font-weight:500}
.file-title{margin:0;font-size:clamp(22px,3vw,32px);letter-spacing:-.01em}
.file-sub{margin:8px 0 0;color:var(--muted);font-size:13.5px;line-height:1.7;max-width:900px}
.meta-row{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0 22px}
.pill{
  display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line2);
  background:rgba(17,27,39,.7);border-radius:999px;padding:5px 10px;
  font-family:var(--mono);font-size:10.5px;color:var(--muted);
}
.pill b{color:var(--text);font-weight:600}
.section-label{
  display:flex;align-items:center;gap:10px;margin:26px 0 12px;
  font-family:var(--mono);font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);
}
.section-label::after{content:"";height:1px;flex:1;background:linear-gradient(90deg,var(--line2),transparent)}
.pipeline{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:2px 2px 12px}
.pipeline.single .stage{flex:1 1 100%;max-width:620px}
.stage{
  flex:0 0 172px;min-height:104px;text-align:left;cursor:pointer;
  border:1px solid var(--line2);border-radius:12px;padding:12px;
  background:linear-gradient(160deg,rgba(21,33,49,.95),rgba(13,20,30,.95));
  position:relative;transition:border-color .15s,transform .15s,box-shadow .15s;
}
.stage:hover{border-color:rgba(34,211,238,.55);transform:translateY(-2px);box-shadow:0 10px 26px rgba(0,0,0,.35)}
.stage-num{font-family:var(--mono);font-size:9.5px;color:var(--cyan);letter-spacing:.12em}
.stage-name{display:block;font-family:var(--mono);font-size:12px;margin:6px 0 4px;color:var(--text);line-height:1.35}
.stage-sub{display:block;color:var(--muted);font-size:10.5px;line-height:1.45;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.stage .key-dot{position:absolute;top:11px;right:11px;width:6px;height:6px;border-radius:50%;background:var(--amber);box-shadow:0 0 8px rgba(251,191,36,.7)}
.flow-arrow{align-self:center;flex:0 0 16px;text-align:center;color:var(--cyan);font-family:var(--mono);font-size:15px;opacity:.8}
.class-group{
  margin:16px 0 24px;border:1px solid var(--line);border-radius:15px;
  background:rgba(13,20,30,.62);overflow:hidden;
}
.class-group > .block-card{border:0;border-bottom:1px solid var(--line);border-radius:0;background:linear-gradient(180deg,rgba(167,139,250,.07),transparent)}
.methods{position:relative;padding:12px 14px 14px 34px;display:flex;flex-direction:column;gap:8px}
.methods::before{
  content:"";position:absolute;left:19px;top:2px;bottom:18px;width:1px;
  background:linear-gradient(180deg,rgba(167,139,250,.5),rgba(34,211,238,.18));
}
.block-card{
  position:relative;display:block;width:100%;text-align:left;cursor:pointer;
  border:1px solid var(--line);border-radius:11px;padding:12px 14px;
  background:rgba(17,27,39,.62);color:var(--text);
  transition:border-color .15s,background .15s,transform .15s;
}
.block-card:hover{border-color:rgba(34,211,238,.45);background:rgba(21,33,49,.85);transform:translateX(2px)}
.block-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.badge{
  display:inline-flex;align-items:center;border-radius:5px;padding:2px 6px;
  font-family:var(--mono);font-size:9.5px;letter-spacing:.06em;border:1px solid transparent;
}
.badge.cls{color:var(--violet);background:rgba(167,139,250,.10);border-color:rgba(167,139,250,.28)}
.badge.method{color:var(--cyan);background:rgba(34,211,238,.08);border-color:rgba(34,211,238,.24)}
.badge.func{color:var(--amber);background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.24)}
.badge.entry{color:var(--emerald);background:rgba(52,211,153,.08);border-color:rgba(52,211,153,.24)}
.badge.const{color:var(--slate);background:rgba(148,163,184,.08);border-color:rgba(148,163,184,.22)}
.block-name{font-family:var(--mono);font-size:12.5px;color:var(--text)}
.block-line{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--dim)}
.block-logic{margin:8px 0 0;color:var(--muted);font-size:12.5px;line-height:1.65}
.key-flag{
  display:inline-flex;margin-top:9px;border:1px solid rgba(251,191,36,.3);
  background:rgba(251,191,36,.07);color:var(--amber);border-radius:5px;
  padding:2px 7px;font-family:var(--mono);font-size:9.5px;
}
.func-flow{display:flex;flex-direction:column;gap:6px}
.down-arrow{text-align:center;color:var(--cyan);opacity:.55;font-family:var(--mono);font-size:13px;line-height:1}
.const-chip{
  display:inline-flex;align-items:center;gap:8px;cursor:pointer;margin:6px 0 8px;
  border:1px dashed var(--line2);border-radius:8px;background:rgba(148,163,184,.05);
  color:var(--muted);padding:7px 10px;font-family:var(--mono);font-size:10.5px;
}
.const-chip:hover{border-color:var(--slate);color:var(--text)}
.entry-wrap{margin-top:14px}
.src-warn{
  margin:0 0 8px;padding:9px 12px;border:1px solid rgba(251,191,36,.26);
  border-radius:9px;background:rgba(251,191,36,.055);color:#e8c879;font-size:11.5px;line-height:1.6;
}
.drawer-backdrop{position:fixed;inset:0;z-index:50;background:rgba(0,0,0,.58);backdrop-filter:blur(2px)}
.drawer{
  position:fixed;top:0;right:0;z-index:60;height:100%;width:min(590px,94vw);
  background:#0a111a;border-left:1px solid var(--line2);
  transform:translateX(102%);transition:transform .22s ease;
  display:flex;flex-direction:column;box-shadow:-30px 0 80px rgba(0,0,0,.5);
}
.drawer.open{transform:none}
.drawer-head{display:flex;gap:12px;align-items:flex-start;padding:18px 20px 14px;border-bottom:1px solid var(--line)}
.drawer-head h2{margin:6px 0 0;font-family:var(--mono);font-size:17px;line-height:1.35}
.drawer-head p{margin:5px 0 0;color:var(--dim);font-family:var(--mono);font-size:10.5px}
.icon-btn{
  margin-left:auto;width:32px;height:32px;border-radius:8px;border:1px solid var(--line2);
  background:rgba(255,255,255,.03);cursor:pointer;color:var(--muted);font-size:18px;line-height:1;
}
.icon-btn:hover{color:var(--text);border-color:var(--cyan)}
.drawer-body{padding:16px 20px 30px;overflow:auto}
.annotation{margin:0 0 14px;color:var(--muted);font-size:12.5px;line-height:1.75}
.tabs{display:flex;gap:6px;margin:4px 0 10px;border-bottom:1px solid var(--line);padding-bottom:8px}
.tab{
  border:1px solid transparent;background:none;border-radius:7px;padding:6px 10px;
  cursor:pointer;color:var(--dim);font-family:var(--mono);font-size:10.5px;
}
.tab:hover{color:var(--text)}
.tab.active{color:var(--cyan);background:rgba(34,211,238,.08);border-color:rgba(34,211,238,.25)}
pre.code{
  margin:0;white-space:pre;overflow:auto;tab-size:2;
  font-family:var(--mono);font-size:12px;line-height:1.7;
  background:#060a0f;border:1px solid var(--line);border-radius:10px;padding:14px;
  max-height:56vh;color:#cfe3ee;
}
pre.code.source{font-size:11.5px;color:#b8c9d6}
[hidden]{display:none!important}
@media(max-width:980px){
  .shell{grid-template-columns:1fr;height:auto}
  .sidebar{display:none}
  .mobile-select{
    display:block;width:100%;height:38px;border-radius:9px;border:1px solid var(--line2);
    background:#0d141e;color:var(--text);padding:0 10px;font-size:12px;
  }
  .topbar{flex-wrap:wrap;min-height:0;padding:12px 16px;gap:10px}
  .brand{min-width:0;flex:1}
  .search-wrap{order:3;flex:1 1 100%;max-width:none}
  .main{padding:20px 16px 70px}
  .pipeline{flex-direction:column;overflow:visible}
  .flow-arrow{transform:rotate(90deg);align-self:flex-start;margin-left:18px}
  .stage{flex:1 1 auto;width:100%}
}
@media(prefers-reduced-motion:reduce){
  *{transition:none!important;scroll-behavior:auto!important}
}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="brand-mark">RL</span>
    <div>
      <h1>强化学习代码流</h1>
      <p id="brand-sub">按文件看主流程、逻辑块与伪代码</p>
    </div>
  </div>
  <div class="search-wrap">
    <input id="search" type="search" placeholder="搜索文件、函数或逻辑…  按 / 聚焦" autocomplete="off" aria-label="搜索代码块">
    <div id="search-results" class="search-results" hidden></div>
  </div>
  <select id="mobile-file" class="mobile-select" aria-label="选择文件"></select>
</header>
<div class="shell">
  <aside class="sidebar">
    <div class="sidebar-head"><span>算法目录</span><span id="file-count"></span></div>
    <nav id="nav"></nav>
  </aside>
  <main id="main" class="main"></main>
</div>
<div id="drawer-backdrop" class="drawer-backdrop" hidden></div>
<aside id="drawer" class="drawer" aria-hidden="true" aria-label="代码块详情">
  <header class="drawer-head">
    <div>
      <span id="drawer-kind" class="badge"></span>
      <h2 id="drawer-name"></h2>
      <p id="drawer-path"></p>
    </div>
    <button id="drawer-close" class="icon-btn" aria-label="关闭详情">×</button>
  </header>
  <div class="drawer-body">
    <p id="drawer-logic" class="annotation"></p>
    <div class="tabs" role="tablist">
      <button class="tab" data-tab="pseudo" role="tab">伪代码</button>
      <button class="tab" data-tab="source" role="tab">源码</button>
    </div>
    <section id="tab-pseudo"><pre id="pseudo-code" class="code"></pre></section>
    <section id="tab-source"><pre id="source-code" class="code source"></pre></section>
  </div>
</aside>
<script>
const DATA = __DATA__;
const $ = (sel, el=document) => el.querySelector(sel);
const state = {path:null, block:null, tab:"pseudo"};
const fileMap = new Map();
DATA.roots.forEach(root => root.files.forEach(file => fileMap.set(file.path, file)));

function esc(value){
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"
  }[ch]));
}
function initials(root){
  return root.files.length;
}
function updateHash(path, idx){
  const hash = idx == null ? encodeURIComponent(path) : encodeURIComponent(path) + "::" + idx;
  history.replaceState(null, "", "#" + hash);
}
function renderNav(){
  $("#nav").innerHTML = DATA.roots.map(root => `
    <div class="root-group" data-root="${root.id}">
      <button class="root-head" data-root-toggle="${root.id}" aria-expanded="false">
        <span class="root-id">${esc(root.id)}</span>
        <span class="root-name">${esc(root.dir.replace(/^\d+_/, ""))}</span>
        <span class="root-count">${initials(root)}</span>
      </button>
      <div class="root-files">
        ${root.files.map(file => `
          <button class="file-item" data-path="${esc(file.path)}">
            <span>${esc(file.name)}</span>
            <span class="file-blocks">${file.blocks.length}</span>
          </button>`).join("")}
      </div>
    </div>`).join("");
  $("#file-count").textContent = DATA.stats.files + " files";
  $("#brand-sub").textContent = `${DATA.stats.files} 个文件 · ${DATA.stats.blocks} 个逻辑块 · 点击块看伪代码`;
  $("#mobile-file").innerHTML = DATA.roots.map(root => `
    <optgroup label="${esc(root.dir)}">
      ${root.files.map(file => `<option value="${esc(file.path)}">${esc(file.name)}</option>`).join("")}
    </optgroup>`).join("");
}
function expandRoot(id){
  const group = document.querySelector(`.root-group[data-root="${id}"]`);
  if(!group) return;
  group.classList.add("open");
  const head = $(".root-head", group);
  if(head) head.setAttribute("aria-expanded", "true");
}
function markActive(){
  document.querySelectorAll(".file-item").forEach(el => {
    el.classList.toggle("active", el.dataset.path === state.path);
  });
  document.querySelectorAll(".root-head").forEach(el => {
    const group = el.closest(".root-group");
    const active = group && [...group.querySelectorAll(".file-item")].some(f => f.dataset.path === state.path);
    el.classList.toggle("active", !!active);
  });
  const mobile = $("#mobile-file");
  if(mobile && mobile.value !== state.path) mobile.value = state.path;
}
function renderBlock(file, idx){
  const b = file.blocks[idx];
  if(!b) return "";
  return `<button class="block-card" data-idx="${idx}">
    <div class="block-top">
      <span class="badge ${b.kc}">${esc(b.kind)}</span>
      <span class="block-name">${esc(b.name)}</span>
      <span class="block-line">${esc(b.line)}</span>
    </div>
    <p class="block-logic">${esc(b.logic)}</p>
    ${b.pseudo ? '<span class="key-flag">伪代码</span>' : ""}
  </button>`;
}
function renderMain(){
  const file = fileMap.get(state.path);
  if(!file){ $("#main").innerHTML = '<p class="file-sub">选择一个文件开始。</p>'; return; }
  const classGroups = file.classGroups.map(group => `
    <article class="class-group">
      ${renderBlock(file, group.classIdx)}
      ${group.methodIdxs.length ? `<div class="methods">${group.methodIdxs.map(i => renderBlock(file, i)).join("")}</div>` : ""}
    </article>`).join("");
  const funcs = file.funcIdxs.map((idx, n) => `
    ${n ? '<span class="down-arrow">↓</span>' : ""}
    ${renderBlock(file, idx)}`).join("");
  const entries = file.entryIdxs.map(idx => renderBlock(file, idx)).join("");
  const constChip = file.constIdx == null ? "" : `
    <button class="const-chip" data-idx="${file.constIdx}">
      <span>导入与常量</span><span>${esc(file.blocks[file.constIdx].line)}</span>
    </button>`;
  const stages = file.stages.map((idx, n) => {
    const b = file.blocks[idx];
    if(!b) return "";
    return `${n ? '<span class="flow-arrow">→</span>' : ""}
      <button class="stage" data-idx="${idx}">
        ${b.pseudo ? '<span class="key-dot"></span>' : ""}
        <span class="stage-num">${String(n + 1).padStart(2, "0")}</span>
        <span class="stage-name">${esc(b.name)}</span>
        <span class="stage-sub">${esc(b.logic)}</span>
      </button>`;
  }).join("");
  $("#main").innerHTML = `
    <div class="crumbs"><span>${esc(file.rootId)}</span><b>/</b><span>${esc(file.dir)}</span><b>/</b><b>${esc(file.name)}</b></div>
    <h2 class="file-title">${esc(file.name)}</h2>
    <p class="file-sub">${esc(file.purpose)}</p>
    <div class="meta-row">
      <span class="pill"><b>${file.lines}</b> 行</span>
      <span class="pill"><b>${file.blocks.length}</b> 逻辑块</span>
      <span class="pill"><b>${file.classGroups.length}</b> 类</span>
      <span class="pill"><b>${file.blocks.filter(b => b.pseudo).length}</b> 个伪代码入口</span>
      <span class="pill">点击块查看详情</span>
    </div>
    ${file.missingSrc ? '<p class="src-warn">源文件当前不在磁盘上，页面保留已提取的块逻辑；源码视图不可用。</p>' : ""}
    <div class="section-label">主流程</div>
    ${stages ? `<div class="pipeline${file.stages.length === 1 ? " single" : ""}">${stages}</div>` : '<p class="file-sub">这个文件没有可展开的执行阶段。</p>'}
    ${constChip}
    <div class="section-label">类与函数逻辑</div>
    ${classGroups}
    ${funcs ? `<div class="func-flow">${funcs}</div>` : ""}
    ${entries ? `<div class="entry-wrap">${entries}</div>` : ""}
  `;
}
function selectFile(path, blockIdx){
  if(!fileMap.has(path)) return;
  state.path = path;
  state.block = null;
  expandRoot(fileMap.get(path).rootId);
  renderMain();
  markActive();
  updateHash(path, null);
  $("#main").scrollTop = 0;
  closeDrawer();
  if(blockIdx != null) setTimeout(() => openDrawer(blockIdx), 0);
}
function openDrawer(idx){
  const file = fileMap.get(state.path);
  if(!file) return;
  const b = file.blocks[idx];
  if(!b) return;
  state.block = idx;
  $("#drawer-kind").className = "badge " + b.kc;
  $("#drawer-kind").textContent = b.kind;
  $("#drawer-name").textContent = b.name;
  $("#drawer-path").textContent = `${file.path} · ${b.line}`;
  $("#drawer-logic").textContent = b.logic;
  const pseudoSection = $("#tab-pseudo");
  const pseudoTab = document.querySelector('.tab[data-tab="pseudo"]');
  if(b.pseudo){
    pseudoSection.hidden = false;
    pseudoTab.hidden = false;
    $("#pseudo-code").textContent = b.pseudo;
  } else {
    pseudoSection.hidden = true;
    pseudoTab.hidden = true;
    $("#pseudo-code").textContent = "";
  }
  const src = b.src && b.src.length
    ? b.src.map((line, i) => String(b.srcStart + i).padStart(4, " ") + "  " + line).join("\n")
    : (file.missingSrc
        ? "源文件当前不在磁盘上。这里只保留已提取的逻辑注释与伪代码，源码视图不可用。"
        : "（源码未嵌入）");
  $("#source-code").textContent = src;
  setTab(b.pseudo ? "pseudo" : "source");
  $("#drawer").classList.add("open");
  $("#drawer").setAttribute("aria-hidden", "false");
  $("#drawer-backdrop").hidden = false;
  document.body.style.overflow = "hidden";
  updateHash(state.path, idx);
  setTimeout(() => $("#drawer-close").focus(), 20);
}
function closeDrawer(){
  $("#drawer").classList.remove("open");
  $("#drawer").setAttribute("aria-hidden", "true");
  $("#drawer-backdrop").hidden = true;
  document.body.style.overflow = "";
  state.block = null;
  if(state.path) updateHash(state.path, null);
}
function setTab(name){
  state.tab = name;
  document.querySelectorAll(".tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.tab === name);
    tab.setAttribute("aria-selected", tab.dataset.tab === name ? "true" : "false");
  });
  $("#tab-pseudo").hidden = name !== "pseudo";
  $("#tab-source").hidden = name !== "source";
}
function search(query){
  const q = query.trim().toLowerCase();
  const box = $("#search-results");
  if(!q){ box.hidden = true; box.innerHTML = ""; return; }
  const hits = [];
  for(const item of DATA.search){
    if(hits.length >= 40) break;
    const text = (item.path + " " + item.name + " " + item.logic).toLowerCase();
    if(text.includes(q)) hits.push(item);
  }
  if(!hits.length){
    box.innerHTML = '<div class="search-empty">没有匹配的文件或代码块。</div>';
    box.hidden = false;
    return;
  }
  box.innerHTML = hits.map(item => `
    <button class="result" data-path="${esc(item.path)}" data-idx="${item.idx}">
      <span class="result-path">${esc(item.path)} · ${esc(item.kind)} · ${esc(item.name)}</span>
      <span class="result-name"><span class="result-kind">${esc(item.name)}</span>${esc(item.logic)}</span>
    </button>`).join("");
  box.hidden = false;
}
function bind(){
  $("#nav").addEventListener("click", event => {
    const toggle = event.target.closest("[data-root-toggle]");
    if(toggle){
      const group = toggle.closest(".root-group");
      const open = group.classList.toggle("open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
      return;
    }
    const file = event.target.closest("[data-path]");
    if(file) selectFile(file.dataset.path);
  });
  $("#main").addEventListener("click", event => {
    const target = event.target.closest("[data-idx]");
    if(target) openDrawer(Number(target.dataset.idx));
  });
  $("#search").addEventListener("input", event => search(event.target.value));
  $("#search-results").addEventListener("click", event => {
    const hit = event.target.closest(".result");
    if(!hit) return;
    $("#search").value = "";
    $("#search-results").hidden = true;
    selectFile(hit.dataset.path, Number(hit.dataset.idx));
  });
  $("#mobile-file").addEventListener("change", event => selectFile(event.target.value));
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#drawer-backdrop").addEventListener("click", closeDrawer);
  document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => setTab(tab.dataset.tab));
  });
  document.addEventListener("keydown", event => {
    if(event.key === "Escape"){
      if(!$("#drawer").classList.contains("open")){
        $("#search-results").hidden = true;
      } else closeDrawer();
    }
    if(event.key === "/" && document.activeElement !== $("#search")){
      event.preventDefault();
      $("#search").focus();
    }
  });
}
function boot(){
  renderNav();
  bind();
  const raw = decodeURIComponent(location.hash.slice(1));
  const [hashPath, hashIdx] = raw.split("::");
  const fallback = "10_TD3/Pendulum/agents.py";
  const path = fileMap.has(hashPath) ? hashPath : (fileMap.has(fallback) ? fallback : DATA.roots[0].files[0].path);
  selectFile(path, hashIdx != null && hashIdx !== "" ? Number(hashIdx) : null);
}
function runAudit(){
  const out = {files:0, badCards:[], pseudoMiss:[], zeroStage:[], overflow:[]};
  const original = state.path;
  DATA.roots.forEach(root => root.files.forEach(file => {
    state.path = file.path;
    renderMain();
    out.files += 1;
    const cards = document.querySelectorAll("#main .block-card").length;
    const expected = file.blocks.filter(b => b.kind !== "导入与常量").length;
    if(cards !== expected) out.badCards.push(`${file.path}:${cards}/${expected}`);
    file.blocks.forEach((block, i) => {
      if(!block.pseudo) return;
      openDrawer(i);
      if(!$("#pseudo-code").textContent.trim()) out.pseudoMiss.push(`${file.path}#${i}`);
      closeDrawer();
    });
    if(!file.stages.length) out.zeroStage.push(file.path);
    if(document.documentElement.scrollWidth > innerWidth + 1){
      out.overflow.push(`${file.path}:${document.documentElement.scrollWidth}`);
    }
  }));
  state.path = original;
  renderMain();
  markActive();
  const pre = document.createElement("pre");
  pre.id = "audit-output";
  pre.textContent = JSON.stringify(out);
  document.body.appendChild(pre);
}
function runSelfTest(){
  const out = {fileItems: document.querySelectorAll(".file-item").length};
  selectFile("10_TD3/Pendulum/agents.py");
  out.stages = document.querySelectorAll(".stage").length;
  const cards = [...document.querySelectorAll(".block-card")];
  const update = cards.find(card => $(".block-name", card)?.textContent === "TD3.update");
  out.hasUpdate = !!update;
  update?.click();
  out.drawer = $("#drawer").classList.contains("open");
  out.pseudo = $("#pseudo-code").textContent.includes("target");
  closeDrawer();
  const input = $("#search");
  input.value = "SAC.update";
  input.dispatchEvent(new Event("input", {bubbles:true}));
  out.viewport = innerWidth;
  out.docScroll = document.documentElement.scrollWidth;
  out.overflow = [...document.querySelectorAll("body *")]
    .filter(el => !el.closest("#drawer, #drawer-backdrop"))
    .filter(el => el.getBoundingClientRect().right > innerWidth + 1)
    .slice(0, 8)
    .map(el => `${el.tagName}.${String(el.className).replace(/\s+/g, ".")}@${Math.round(el.getBoundingClientRect().right)}`);
  setTimeout(() => {
    out.search = document.querySelectorAll("#search-results .result").length;
    const pre = document.createElement("pre");
    pre.id = "selftest-output";
    pre.textContent = JSON.stringify(out);
    document.body.appendChild(pre);
  }, 150);
}
document.addEventListener("DOMContentLoaded", () => {
  boot();
  const params = new URLSearchParams(location.search);
  if(params.has("audit")) setTimeout(runAudit, 120);
  else if(params.has("selftest")) setTimeout(runSelfTest, 120);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
