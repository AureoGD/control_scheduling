import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_animation import PendulumLiveRenderer
# from envirioment.cart_pendulum.pendulum_controllers import LQR, SlidingMode, SwingUp
import time


class SchedullerRule():

    def __init__(self):
        self.n_controllers = 1

    def update_control_action(self, controller_index, state):

        return 0


class InvPendulumEnv(gym.Env):

    def __init__(self, env_id=None, dt=0.002, max_step=5000, rendering=False, frame_rate=30, sw_rule=None):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering
        self.frame_rate = frame_rate
        self.max_step = max_step
        self.dt = dt

        self.inv_pendulum = InvePendulum(dt=self.dt)

        if self.rendering:
            self.pendulum_renderer = PendulumLiveRenderer(self.inv_pendulum)

        # Controllers:

        self.sw_rule = sw_rule

        self.action_space = gym.spaces.Discrete(n=self.sw_rule.n_controllers)

        # For now, a simple observation space. The satates must be normalized
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)

        self.scale_factor = 0.75
        self.ep_reward = 0
        self.current_step = 0
        self.ep = 0
        self.st = None

        self.current_mode = None
        self.last_mode = None
        self.difficult_start = None

    def step(self, action):
        self.current_mode = action
        self.current_step += 1

        # in this case, the control action is the index o the internal controller
        force = self.sw_rule.update_control_action(action, self.st)

        self.st = self.inv_pendulum.step_sim(force)
        new_state = self._norm(self.st)
        if self.rendering and (self.current_step % self.frame_rate == 0 or self.current_step == 0):
            self.render()
            time.sleep(self.dt * self.frame_rate)

        reward = self._reward(self.st)
        done = self._done()
        self.ep_reward += reward

        # For Gym consistency
        terminated = done
        truncated = done
        info = {}

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
        # x0 = np.array([0, 0, 0, 0])
        # print(x0)
        self.st = self.inv_pendulum.reset(x0)
        state = self._norm(self.st)

        self.difficult_start = 0.01 + 1 - state[2]

        if self.rendering:
            self.pendulum_renderer.init_live_render()

        return state, {"Episode": self.ep, "Episode reward": ep_r}

    def render(self):
        self.pendulum_renderer.update_live_render()

    def close(self):
        if self.rendering:
            self.pendulum_renderer.close_render()

    def _done(self):
        if self.current_step >= self.max_step:
            return True
        return False

    def _norm(self, states):
        x = states[0] / self.inv_pendulum.x_max
        dx = states[1] / self.inv_pendulum.v_max
        a = states[2]

        cos_a = np.cos(a)
        sin_a = np.sin(a)
        da = states[3] / self.inv_pendulum.da_max
        return np.array([x, dx, cos_a, sin_a, da]).reshape(5, )

    def _reward(self, state):
        pos = abs(state[0])
        cos_angle = np.cos(state[2])
        r = 0

        if self.current_mode != self.last_mode:
            self.last_mode = self.current_mode
            return r

        if cos_angle >= np.cos(0.2):
            r += 1 + (2 - pos) * 0.5

        return r
