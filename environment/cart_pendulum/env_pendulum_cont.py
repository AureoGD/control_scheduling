import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_animation import PendulumLiveRenderer
import time


class InvPendulumEnv(gym.Env):

    def __init__(self, env_id=None, dt=0.002, max_step=5000, rendering=False, frame_rate=30):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering
        self.frame_rate = frame_rate
        self.max_step = max_step
        self.dt = dt

        self.inv_pendulum = InvePendulum(dt=self.dt)

        if self.rendering:
            self.pendulum_renderer = PendulumLiveRenderer(self.inv_pendulum)

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)
        # For now, a simple observation space. The satates must be normalized
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)

        self.scale_factor = 0.75
        self.ep_reward = 0
        self.current_step = 0
        self.ep = 0

    def step(self, action):
        self.current_step += 1
        force = self.inv_pendulum.f_max * np.clip(action[0], -1, 1)
        sys_sts = self.inv_pendulum.step_sim(force)
        new_state = self._norm(sys_sts)
        if self.rendering and (self.current_step % self.frame_rate == 0 or self.current_step == 0):
            self.render()
            time.sleep(self.dt * self.frame_rate)

        reward = self._reward(sys_sts)
        done = self._done()
        self.ep_reward += reward

        # For Gym consistency
        terminated = done
        truncated = done
        info = {}

        if not np.all(np.isfinite(new_state)):
            print(f"!!! Invalid state detected in step: {new_state}")

        # Make sure reward is also a valid number
        if not np.isfinite(reward):
            print(f"!!! Invalid reward detected: {reward}")

        return new_state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ep += 1
        ep_r = self.ep_reward
        self.current_step = 0
        self.ep_reward = 0
        x0 = np.array([
            self.scale_factor * np.random.uniform(-self.inv_pendulum.x_max, self.inv_pendulum.x_max), 0,
            self.scale_factor * np.random.uniform(-np.pi, np.pi), 0
        ])
        state = self._norm(self.inv_pendulum.reset(x0))

        if self.rendering:
            self.pendulum_renderer.init_live_render()
            self.render()

        return state, {"Episode": self.ep, "Episode reward": ep_r}

    def render(self):
        self.pendulum_renderer.update_live_render()

    def close(self):
        if self.rendering:
            self.pendulum_renderer.close_render()

    def _done(self):
        if self.current_step >= self.max_step:
            return True

    def _norm(self, states):
        x = states[0] / self.inv_pendulum.x_max
        dx = states[1] / self.inv_pendulum.v_max
        a = states[2]

        cos_a = np.cos(a)
        sin_a = np.sin(a)
        da = states[3] / self.inv_pendulum.da_max
        return np.array([x, dx, cos_a, sin_a, da]).reshape(5, )

    def _reward(sel, state):
        pos = abs(state[0])
        angle_reward = np.cos(state[2])
        r = -0.0001

        if angle_reward >= np.cos(0.2):
            r += 1 + (2 - pos) * 0.5

        return r
