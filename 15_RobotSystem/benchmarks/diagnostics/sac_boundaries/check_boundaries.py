"""Fixed-input checks for the Panda SAC environment and bootstrap boundaries."""

import argparse
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


class FixedPolicy(torch.nn.Module):
    def forward(self, states):
        return states.new_zeros((len(states), 2)), states.new_full((len(states), 1), -0.5)


class FixedCritic(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, states, actions):
        return states.new_full((len(states), 1), self.value)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.repo / "15_RobotSystem"
    if not project.is_dir():
        project = args.repo / "hands-on-reinforcement-learning" / "15_RobotSystem"
    pendulum = project.parent / "11_SAC" / "Pendulum"
    sources = (project / "panda_reach.py", project / "panda_two_joint_env.py",
               project / "panda_sac_smoke.py", pendulum / "agents.py")
    source_hashes = {str(path.relative_to(args.repo)): digest(path) for path in sources}
    sys.path.insert(0, str(project))
    from panda_reach import EVAL_CASES, command_controller, run_episode
    from panda_sac_smoke import CONFIG, SAC
    from panda_two_joint_env import PandaTwoJointEnv

    torch.set_num_threads(1)
    torch.manual_seed(42)
    env = PandaTwoJointEnv(args.scene)
    baseline = []
    for seed, goal in EVAL_CASES["validation"]:
        trace = StringIO()
        result = run_episode(env, command_controller, goal, seed, 0, trace)
        rows = [json.loads(line) for line in trace.getvalue().splitlines()]
        assert result["outcome"] == "success" and len(rows) == result["steps"]
        assert all(row["reward"] == -1 for row in rows[:-1])
        assert rows[-1]["reward"] == 0 and rows[-1]["terminated"] and not rows[-1]["truncated"]
        assert max(result["final_abs_error_rad"]) < 0.01
        assert max(result["final_abs_speed_rad_s"]) < 0.05
        assert result["return"] == -(len(rows) - 1)
        baseline.append(result)
    successful_next_observation = rows[-1]["next_observation"]
    success_done = rows[-1]["terminated"]

    goal = (0.10, 0.10)
    probes = []
    for command in [(0.0, 0.0), (0.02, 0.02)]:
        env.reset(goal=goal)
        observation, reward, terminated, truncated, _ = env.step(command)
        assert reward == -1 and not terminated and not truncated
        probes.append({"command_rad": command, "reward": reward,
                       "max_error_rad": float(np.max(np.abs(observation[:2] - goal)))})
    assert probes[1]["max_error_rad"] < probes[0]["max_error_rad"]
    crossing_goal = observation[:2].copy()
    env.reset(goal=crossing_goal)
    crossing, crossing_reward, terminated, truncated, _ = env.step((0.02, 0.02))
    assert np.max(np.abs(crossing[:2] - crossing_goal)) < 0.01
    assert np.max(np.abs(crossing[2:4])) >= 0.05
    assert crossing_reward == -1 and not terminated and not truncated

    timed_env = PandaTwoJointEnv(args.scene, goal=goal, max_steps=1)
    timed_env.reset()
    timeout_state, timeout_reward, terminated, truncated, info = timed_env.step((0.02, 0.02))
    assert info["outcome"] == "timeout" and timeout_reward == -1 and not terminated and truncated
    timeout_done = terminated
    timeout_snapshot = timeout_state.copy()
    reset_state, _ = timed_env.reset(goal=(-0.1, 0.1))
    np.testing.assert_array_equal(timeout_state, timeout_snapshot)
    assert not np.array_equal(timeout_state[:2], reset_state[:2])
    assert not np.array_equal(timeout_state[4:6], reset_state[4:6])

    env.reset(goal=goal)
    _, _, _, _, clipped_increment = env.step((0.05, -0.03))
    np.testing.assert_allclose(clipped_increment["applied_increment_rad"], (0.02, -0.02), atol=1e-12, rtol=0)
    env.reset(goal=goal)
    env.data.ctrl[env.actuators[0]] = env.upper[0] - 0.01
    _, _, _, _, clipped_target = env.step((0.02, 0.0))
    np.testing.assert_allclose(clipped_target["applied_increment_rad"], (0.01, 0.0), atol=1e-12, rtol=0)
    np.testing.assert_equal(clipped_target["target_rad"][0], env.upper[0])

    env.reset(goal=goal)
    env.data.qpos[env.qadr[0]] = env.actual_upper[0] + 0.2
    mujoco.mj_forward(env.model, env.data)
    fail_state, fail_reward, terminated, truncated, info = env.step((0.0, 0.0))
    assert info["outcome"] == "failure" and info["actual_limit_violation"]
    assert fail_reward == -1 and terminated and not truncated

    agent = SAC(**CONFIG)
    agent.actor = FixedPolicy()
    agent.target_critic_1 = FixedCritic(2.0)
    agent.target_critic_2 = FixedCritic(3.0)
    rewards = torch.tensor([[0.0], [timeout_reward], [fail_reward]])
    next_states = torch.as_tensor(np.array([successful_next_observation, timeout_state, fail_state]),
                                  dtype=torch.float32)
    dones = torch.tensor([[success_done], [timeout_done], [terminated]], dtype=torch.float32)
    actual = agent.calc_target(rewards, next_states, dones)
    expected = torch.tensor([[0.0], [timeout_reward + CONFIG["gamma"] * (2 + 0.01 * 0.5)], [-1.0]])
    torch.testing.assert_close(actual, expected)
    assert not actual.requires_grad

    report = {
        "status": "PASS", "training_updates": 0, "seed": 42,
        "python": sys.version, "mujoco": mujoco.__version__, "torch": str(torch.__version__),
        "scene_sha256": digest(args.scene), "source_sha256": source_hashes,
        "validation_position_baseline": baseline,
        "non_success_probes": probes,
        "passing_through_goal": {"q_rad": crossing[:2].tolist(),
                                 "abs_max_speed_rad_s": float(np.max(np.abs(crossing[2:4]))),
                                 "reward": crossing_reward},
        "timeout_preserved_next_observation": timeout_state.tolist(),
        "observation_after_reset": reset_state.tolist(),
        "action_rate_limit_applied_rad": clipped_increment["applied_increment_rad"].tolist(),
        "position_limit_applied_rad": clipped_target["applied_increment_rad"].tolist(),
        "joint_limit_failure": True,
        "target_cases": ["success", "timeout", "physical_failure"],
        "target_done_mask": dones.flatten().tolist(),
        "td_targets_fixed_critics": actual.flatten().tolist(),
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
