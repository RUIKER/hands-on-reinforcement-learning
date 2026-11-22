"""Render saved Panda experiment trajectories as an offline browser replay."""

import argparse
import base64
import csv
from io import BytesIO
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from panda_two_joint_env import PandaTwoJointEnv

HERE = Path(__file__).resolve().parent
WIDTH = 640
HEIGHT = 400


def fixed_trajectory(path, model):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    home = model.key("home").qpos.copy()
    addresses = [model.jnt_qposadr[model.joint(f"joint{axis}").id] for axis in range(1, 8)]
    states = [home.copy()]
    series = [[0, home[addresses[0]], home[addresses[0]], None]]
    for row in rows:
        pose = home.copy()
        pose[addresses] = [float(row[f"q_j{axis}_rad"]) for axis in range(1, 8)]
        states.append(pose)
        series.append([float(row["sample_time_s"]), pose[addresses[0]], float(row["target_j1_rad"]), None])
    return states, series


def policy_trajectory(path, scene):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    goal = rows[0]["observation"][4:6]
    env = PandaTwoJointEnv(scene, goal=goal)
    observation, _ = env.reset()
    states = [env.data.qpos.copy()]
    series = [[0, observation[0], observation[6], goal[0]]]
    for step, row in enumerate(rows, 1):
        np.testing.assert_allclose(observation, row["observation"], atol=1e-9, rtol=0)
        observation, reward, terminated, truncated, info = env.step(row["requested_rad"])
        np.testing.assert_allclose(observation, row["next_observation"], atol=1e-9, rtol=0)
        assert step == row["step"] and info["outcome"] == row["outcome"]
        assert (reward, terminated, truncated) == (row["reward"], row["terminated"], row["truncated"])
        states.append(env.data.qpos.copy())
        series.append([row["sim_time_s"], observation[0], observation[6], goal[0]])
    return states, series


def render_frames(model, states, stride):
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, WIDTH)
    model.vis.global_.offheight = max(model.vis.global_.offheight, HEIGHT)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.20, 0, 0.43)
    camera.distance = 1.65
    camera.azimuth = 124
    camera.elevation = -16
    indices = list(range(0, len(states), stride))
    if indices[-1] != len(states) - 1:
        indices.append(len(states) - 1)
    with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
        for index in indices:
            data.qpos[:] = states[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            yield index, Image.fromarray(renderer.render())


def webp_data(image):
    stream = BytesIO()
    image.save(stream, format="WEBP", quality=79, method=4)
    return "data:image/webp;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def generate(scene, results, output, assets):
    model = mujoco.MjModel.from_xml_path(str(scene))
    tracking = fixed_trajectory(results / "20261117_fixed_tracking/tracking.csv", model)
    policy = results / "20261119_panda_policy"
    success = policy_trajectory(policy / "30104-ros2.jsonl", scene)
    timeout = policy_trajectory(policy / "30124-ros2.jsonl", scene)
    clips = []
    specs = [
        ("七轴跟踪", "固定轨迹 · 位置控制", "跟踪完成", tracking, 3),
        ("策略成功", "SAC + HER · ROS 2 · 用例 30104", "第 16 步成功", success, 1),
        ("策略超时", "SAC + HER · ROS 2 · 用例 30124", "第 250 步超时", timeout, 5),
    ]
    for title, subtitle, outcome, (states, series), stride in specs:
        frames = list(render_frames(model, states, stride))
        clips.append({
            "title": title, "subtitle": subtitle, "outcome": outcome,
            "samples": [[None if value is None else round(float(value), 8) for value in row] for row in series],
            "frames": [[index, webp_data(image)] for index, image in frames],
        })
        if assets and title == "七轴跟踪":
            assets.mkdir(parents=True, exist_ok=True)
            images = [image for _, image in frames]
            images[0].save(
                assets / "panda-tracking.webp", save_all=True, append_images=images[1:],
                duration=[180] * (len(images) - 1) + [750], loop=0, quality=78, method=4,
            )
    template = (HERE / "replay_template.html").read_text(encoding="utf-8")
    data = json.dumps(clips, ensure_ascii=False, separators=(",", ":"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__PANDA_REPLAY_DATA__", data), encoding="utf-8", newline="\n")
    print(f"Verified two policy trajectories against saved states. Rendered {sum(len(c['frames']) for c in clips)} frames: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path)
    args = parser.parse_args()
    generate(args.scene, args.results, args.output, args.assets)


if __name__ == "__main__":
    main()
