"""MuJoCo evaluation runtime shared by seven-axis training and ROS 2 deployment."""

import mujoco
import numpy as np

from panda_ik import HandPositionIK


LIMIT = 0.02
DISTANCE = 0.02
SPEED = 0.03
HORIZON = 250
ACTION_SCALE = 0.008
JOINT_MARGIN = 0.05
DAMPING = 0.02
PANDA_HOME = np.array([0., 0., 0., -1.57079, 0., 1.57079, -0.7853])
DISCOUNT = .99
# All non-failure unshaped rewards lie in [-1, 0].
FAILURE_REWARD = -1.0 - 1.0 / (1.0 - DISCOUNT)


def success(obs, goal):
    return bool(np.linalg.norm(obs[21:24] - goal) < DISTANCE
                and np.linalg.norm(obs[24:27]) < SPEED)


class Environment:
    def __init__(self, scene):
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.ik = HandPositionIK(self.model)
        self.act = np.array([self.model.actuator(f'actuator{i}').id for i in range(1, 8)])
        self.qadr = self.ik.qadr
        self.dadr = self.ik.dadr
        self.jac = np.empty((3, self.model.nv))
        self.done = True

    def observation(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        rot = self.data.xmat[self.ik.base].reshape(3, 3).T
        pos = rot @ (self.data.xpos[self.ik.hand] - self.data.xpos[self.ik.base])
        mujoco.mj_jacBody(self.model, self.data, self.jac, None, self.ik.hand)
        vel = rot @ self.jac @ self.data.qvel
        return np.r_[self.data.qpos[self.qadr], self.data.qvel[self.dadr],
                     self.data.ctrl[self.act], pos, vel, self.goal].copy()

    def reset(self, goal):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('Expected finite 3D goal')
        self.goal = goal.copy()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.done = False
        return self.observation()

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode ended')
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError('Expected seven finite normalized actions')
        self.data.ctrl[self.act] = np.clip(self.data.ctrl[self.act] + LIMIT * action,
                                         self.ik.lower, self.ik.upper)
        mujoco.mj_step(self.model, self.data, nstep=10)
        self.steps += 1
        obs = self.observation()
        q = self.data.qpos[self.qadr]
        failed = (not np.isfinite(obs).all() or any(w.number for w in self.data.warning)
                  or np.any(q < self.ik.lower - .001) or np.any(q > self.ik.upper + .001))
        reached = not failed and success(obs, self.goal)
        terminated = bool(failed or reached)
        truncated = bool(not terminated and self.steps >= HORIZON)
        self.done = terminated or truncated
        return obs, (0. if reached else -1.), terminated, truncated, ('failure' if failed else 'success' if reached else 'timeout' if truncated else 'running')


def distance_reward(state, goal, reached):
    return 0. if reached else -float(np.clip(np.linalg.norm(state[21:24] - goal) / .13, .01, 1.))


class CartesianEnvironment(Environment):
    def __init__(self, scene):
        super().__init__(scene)
        if not np.isclose(self.model.opt.timestep, .002) or not np.allclose(self.ik.home, PANDA_HOME):
            raise ValueError('Expected the pinned Panda model home pose and 2 ms physics timestep')

    def observation(self):
        state = super().observation()
        # Track the Cartesian location of the held joint setpoint as well as
        # the measured hand. This lets the policy stop advancing the command
        # while the physical servo is still approaching it.
        command_position, _ = self.ik.position(state[14:21])
        return np.r_[state, command_position]

    def project_action(self, command, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError('Expected three finite normalized Cartesian actions')
        self.ik.position(command)
        mujoco.mj_jacBody(self.model, self.ik.data, self.ik.jacobian, None, self.ik.hand)
        rotation = self.ik.data.xmat[self.ik.base].reshape(3, 3)
        jac = rotation.T @ self.ik.jacobian[:, self.dadr]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + DAMPING**2 * np.eye(3),
                                       ACTION_SCALE * action)
        delta *= min(1., LIMIT / max(np.max(np.abs(delta)), 1e-12))
        return np.clip(command + delta, self.ik.lower + JOINT_MARGIN, self.ik.upper - JOINT_MARGIN)

    def step(self, action):
        command = self.data.ctrl[self.act]
        target = self.project_action(command, action)
        state, _, terminal, timeout, outcome = super().step((target - command) / LIMIT)
        reward = (FAILURE_REWARD if outcome == 'failure'
                  else distance_reward(state, self.goal, outcome == 'success'))
        return state, reward, terminal, timeout, outcome


def features(state):
    state = np.asarray(state)
    return np.concatenate(((state[..., 27:30] - state[..., 30:33]) / .10,
                           (state[..., 27:30] - state[..., 21:24]) / .10,
                           state[..., 24:27] / .5,
                           state[..., 14:21] - PANDA_HOME), axis=-1)
