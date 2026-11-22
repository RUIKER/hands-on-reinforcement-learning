"""Run the saved seven-axis reaching policy across two real ROS 2 processes."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import time

import mujoco
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from sensor_msgs.msg import JointState

from panda_ee_env import CartesianEnvironment, HORIZON, features
from panda_policy_actor import ACTOR_KEYS, actor_action, model_digest
from panda_policy_interfaces.srv import ResetCartesian, StepCartesian
from panda_ros2_policy import call


TOPIC = "/panda_cartesian/joint_states"
RESET = "/panda_cartesian/reset"
STEP = "/panda_cartesian/step"
NAMES = [f"joint{i}" for i in range(1, 8)]
REFERENCE_ATOL = 1e-5


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


class Simulator(Node):
    def __init__(self, scene):
        super().__init__("panda_cartesian_sim")
        self.env = CartesianEnvironment(scene)
        self.model_hash = model_digest(scene)
        self.episode = None
        self.publisher = self.create_publisher(JointState, TOPIC, 100)
        self.create_service(ResetCartesian, RESET, self.reset)
        self.create_service(StepCartesian, STEP, self.step)

    def publish_state(self, state):
        msg = JointState()
        stamp = round(self.env.data.time * 1e9)
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 10**9)
        msg.name = NAMES
        msg.position = state[:7].tolist()
        msg.velocity = state[7:14].tolist()
        self.publisher.publish(msg)

    def reset(self, req, res):
        try:
            state = self.env.reset(req.goal_m)
        except ValueError as exc:
            res.reason = str(exc)
            return res
        self.episode = secrets.randbelow(2**64 - 1) + 1
        res.episode, res.accepted = self.episode, True
        res.model_digest = self.model_hash
        res.observation = state.tolist()
        self.publish_state(state)
        return res

    def step(self, req, res):
        if req.episode != self.episode:
            res.reason = "wrong_episode"
        elif self.env.done:
            res.reason = "episode_finished"
        elif req.step != self.env.steps + 1:
            res.reason = "unexpected_step"
        else:
            try:
                state, reward, terminal, truncated, outcome = self.env.step(req.action)
            except ValueError as exc:
                res.reason = str(exc)
                return res
            res.episode, res.step, res.accepted = self.episode, self.env.steps, True
            res.observation, res.reward = state.tolist(), reward
            res.terminated, res.truncated, res.outcome = terminal, truncated, outcome
            res.sim_time_s = self.env.data.time
            self.publish_state(state)
        return res


def joint_sample(state, time_s):
    return (round(time_s * 1e9), list(state[:7]), list(state[7:14]))


def verify_recording(storage, expected):
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, rosbag2_py.ConverterOptions("", ""))
    count = 0
    while reader.has_next():
        topic, data, _, _ = reader.read_next_ext()
        msg = deserialize_message(data, JointState)
        stamp = msg.header.stamp
        row = (stamp.sec * 10**9 + stamp.nanosec, list(msg.position), list(msg.velocity))
        if topic != TOPIC or list(msg.name) != NAMES or count >= len(expected) or row != expected[count]:
            raise RuntimeError(f"MCAP sample {count} differs from the offline joint state")
        count += 1
    if count != len(expected):
        raise RuntimeError(f"MCAP contains {count} joint states, expected {len(expected)}")
    return count


def check_rejection(node, step, request, reason):
    rejected = call(node, step, request)
    if rejected.accepted or rejected.reason != reason:
        raise RuntimeError(f"Expected service rejection {reason}, received {rejected.reason}")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(node, scene, run, output):
    protocol = read_json(run / "protocol.json")
    if protocol["model_sha256"] != {p.name: digest(p) for p in (scene, scene.with_name("panda.xml"))}:
        raise ValueError("Model XML differs from the policy training model")
    actor_path = run / "sac_her_bc.npz"
    with np.load(actor_path, allow_pickle=False) as archive:
        weights = {key: archive[key] for key in ACTOR_KEYS}
    goals = read_json(run / "goals.json")
    saved_paths = read_json(run / "trajectories.json")["sac_her_bc"]
    saved_results = read_json(run / "evaluation.json")["sac_her_bc"]
    offline = CartesianEnvironment(scene)
    model_hash = model_digest(scene)
    reset = node.create_client(ResetCartesian, RESET)
    step = node.create_client(StepCartesian, STEP)
    for client in (reset, step):
        if not client.wait_for_service(timeout_sec=20):
            raise TimeoutError(client.srv_name)

    output.mkdir(parents=True, exist_ok=False)
    storage = rosbag2_py.StorageOptions(uri=str(output / "joint_states"), storage_id="mcap")
    writer = rosbag2_py.SequentialWriter()
    writer.open(storage, rosbag2_py.ConverterOptions("", ""))
    writer.create_topic(rosbag2_py.TopicMetadata(id=0, name=TOPIC,
                        type="sensor_msgs/msg/JointState", serialization_format="cdr"))
    received = 0

    def record(msg):
        nonlocal received
        writer.write(TOPIC, serialize_message(msg), node.get_clock().now().nanoseconds)
        received += 1

    sub = node.create_subscription(JointState, TOPIC, record, 100)
    expected_samples = []
    cases = []
    previous = None
    max_q, max_xyz = 0., 0.
    deadline = time.monotonic() + 10
    try:
        while not sub.get_publisher_count():
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint-state publisher discovery timed out")
            rclpy.spin_once(node, timeout_sec=.1)
        for split in ("frozen", "fresh"):
            for i, goal in enumerate(goals[split]):
                reference = saved_paths[split][i]
                original = saved_results[split][i]
                res = call(node, reset, ResetCartesian.Request(goal_m=goal))
                if not res.accepted or res.model_digest != model_hash:
                    raise RuntimeError(f"Invalid reset at {split}/{i}: {res.reason}")
                if previous is not None:
                    check_rejection(node, step, StepCartesian.Request(episode=previous, step=1),
                                    "wrong_episode")
                episode = previous = res.episode
                state = offline.reset(goal)
                ros_state = np.asarray(res.observation)
                for index in range(HORIZON + 1):
                    if not np.array_equal(ros_state, state):
                        raise RuntimeError(f"State mismatch at {split}/{i}/{index}")
                    expected_samples.append(joint_sample(state, offline.data.time))
                    ref = reference[index]
                    max_q = max(max_q, float(np.max(np.abs(state[:7] - ref["q"]))))
                    max_xyz = max(max_xyz, float(np.max(np.abs(state[21:24] - ref["xyz"]))))
                    if offline.done:
                        break
                    action = actor_action(weights, features(ros_state))
                    req = StepCartesian.Request(episode=episode, step=index+1, action=action.tolist())
                    if index == 0:
                        check_rejection(node, step, StepCartesian.Request(episode=episode, step=2),
                                        "unexpected_step")
                    res = call(node, step, req)
                    offline_action = actor_action(weights, features(state))
                    state, reward, terminal, truncated, outcome = offline.step(offline_action)
                    ros_state = np.asarray(res.observation)
                    if (not res.accepted or (res.episode, res.step) != (episode, index+1)
                            or res.sim_time_s != offline.data.time
                            or (res.reward, res.terminated, res.truncated, res.outcome)
                            != (reward, terminal, truncated, outcome)):
                        raise RuntimeError(f"Transition mismatch at {split}/{i}/{index+1}")
                    if index == 0 and not offline.done:
                        check_rejection(node, step, req, "unexpected_step")
                if (not offline.done or original["steps"] != offline.steps
                        or len(reference) != offline.steps + 1 or original["outcome"] != outcome):
                    raise RuntimeError(f"Stored trajectory differs at {split}/{i}")
                check_rejection(node, step, StepCartesian.Request(episode=episode, step=offline.steps+1),
                                "episode_finished")
                cases.append(dict(split=split, index=i, steps=offline.steps, outcome=outcome))
        if max(max_q, max_xyz) > REFERENCE_ATOL:
            raise RuntimeError(f"Training trajectory mismatch: {max_q} rad, {max_xyz} m")
        deadline = time.monotonic() + 10
        while received < len(expected_samples) and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=.1)
    finally:
        node.destroy_subscription(sub)
        writer.close()
    count = verify_recording(storage, expected_samples)
    summary = dict(status="PASS", ros_distro=os.environ.get("ROS_DISTRO"), mujoco=mujoco.__version__,
                   numpy=np.__version__, policy="sac_her_bc", training_seed=protocol["training_seed"],
                   actor_sha256=digest(actor_path), goals_sha256=digest(run / "goals.json"),
                   model_xml_sha256=protocol["model_sha256"], matched_control_steps=sum(c["steps"] for c in cases),
                   exact_fields=["33-state observation", "simulation time", "reward", "terminated", "truncated", "outcome"],
                   rejected=["stale episode", "duplicate step", "skipped step", "step after episode"],
                   matched_bag_messages=count, topic=TOPIC,
                   saved_pytorch_max_abs_joint_rad=max_q, saved_pytorch_max_abs_xyz_m=max_xyz,
                   saved_pytorch_atol=REFERENCE_ATOL, cases=cases)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"PASS: {len(cases)} cases, {summary['matched_control_steps']} steps, {count} MCAP samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    server = commands.add_parser("serve")
    server.add_argument("scene", type=Path)
    client = commands.add_parser("evaluate")
    client.add_argument("scene", type=Path)
    client.add_argument("run", type=Path)
    client.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        with rclpy.init(args=[]):
            if args.mode == "serve":
                node = Simulator(args.scene)
                try:
                    rclpy.spin(node)
                finally:
                    node.destroy_node()
            else:
                node = Node("panda_cartesian_evaluator")
                try:
                    evaluate(node, args.scene, args.run, args.output)
                finally:
                    node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        if args.mode != "serve":
            raise


if __name__ == "__main__":
    main()
