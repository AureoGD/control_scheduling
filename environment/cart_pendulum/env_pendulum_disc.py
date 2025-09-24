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

    def __init__(self,
                 env_id=None,
                 dt=0.001,
                 max_step=500,
                 disturbance=False,
                 noise=False,
                 rendering=False,
                 frame_rate=30,
                 sw_rule=None):
        super().__init__()
        self.env_id = env_id
        self.rendering = rendering
        self.frame_rate = frame_rate
        self.max_step = max_step
        self.dt = dt
        self.disturbance = disturbance
        self.noise = noise

        self.inv_pendulum = InvePendulum(dt=self.dt, noise=self.noise, disturbance=self.disturbance, soft_wall=True)

        if self.rendering:
            self.pendulum_renderer = PendulumLiveRenderer(self.inv_pendulum)

        # Controllers:

        self.sw_rule = sw_rule

        self.action_space = gym.spaces.Discrete(n=self.sw_rule.n_controllers)

        # For now, a simple observation space. The satates must be normalized
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)

        self.scale_factor = 1.0
        self.ep_reward = 0
        self.current_step = 0
        self.ep = 0
        self.st = None

        self.current_mode = None
        self.last_mode = None

        self.P = np.array([[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]])
        self.max_vals = np.array([1.5, 2.75, 0.525, 1.5])
        self.theta_max = np.pi/2

        self.force = 0

    def step(self, action):
        self.current_mode = action
        self.current_step += 1

        # in this case, the control action is the index o the internal controller
        force = self.sw_rule.update_control_action(action, self.st)
        self.force = force

        self.st = self.inv_pendulum.step_sim(force)
        new_state = self._norm(self.st)
        if self.rendering and (self.current_step % (self.frame_rate / 10) == 0 or self.current_step == 0):
            self.render()
            time.sleep(self.frame_rate * self.dt)

        reward = self._reward(self.st)
        self.ep_reward += reward

        # For Gym consistency
        terminated = (abs(self.st[2]) > self.theta_max) or self.health < 0

        truncated = self.current_step >= self.max_step

        info = {}
        if truncated and not terminated:
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = self.st
        info["raw_state"] = self.st
        info["control_effort"] = force
        info["pred_state"] = self.inv_pendulum.st_
        info["dist_detected"] = self.inv_pendulum.disturb_detected
        info["scores"] = self.scores_values

        return new_state, reward, terminated, truncated, info

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def reset(self, *, seed=None, options=None, x0=None, disturb=None):
        super().reset(seed=seed)
        self.ep += 1
        ep_r = self.ep_reward
        self.current_step = 0
        self.ep_reward = 0
        self.health = 20
        self.last_mode = None
        self.mode_steps = 0
        self.min_dwell = 20

        self._define_constants()

        if x0 is None:
            x0 = self._sample_x0_rad()

        self.st = self.inv_pendulum.reset(x0, disturb=disturb)
        state = self._norm(self.st)

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
        if self.current_step >= self.max_step or self.health < 0:
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

    def _define_constants(self):
        self.a_states = 1e-3 * self.scale_factor
        self.a_theta = 3e-4 * self.scale_factor
        self.a_center = 4e-4 * self.scale_factor
        self.a_prog = 5e-3 * self.scale_factor
        self.a_alive = 20e-3* self.scale_factor
        self.a_force = 1e-3* self.scale_factor
        self.a_pref = 0.025* self.scale_factor
        self.a_origin = 1.5e-3* self.scale_factor

        sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
        x_tol = 0.5  # at |x|=x_tol -> 0.5 bonus
        a_tol = 0.2  # at |a|=a_tol -> 0.5 bonus
        self.sx = max(1e-9, x_tol / sqrt_2ln2)
        self.sa = max(1e-9, a_tol / sqrt_2ln2)

        self.a_lqr = 10.0
        self.b_lqr = 7.0
        self.c_near = 1.0
        self.rho_lqr = 2.0
        self.mu_x = 0.0

        self.a_sm = 10
        self.b_sm = self.a_sm * 0.40

        self.v_ref = 0.6
        self.tau_pref = 0.5

        self.a_vf = 10.0
        self.b_vf = self.a_vf * 0.5
        self.kappa_theta = (1.0 - 0.5) / self.v_ref  # (1-T_angle)/v_ref

        self.k_s = 0.55

        self.prev_V = None
        self.scores_values = None

    def _sig(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _controller_scores(self, st):

        x, dx, a, da = map(float, st)
        E_theta = abs(a) /self.inv_pendulum.a_max
        E_x = abs(x) / self.inv_pendulum.x_max
        E_v_raw = np.sqrt((dx / self.inv_pendulum.v_max)**2 + (da / self.inv_pendulum.da_max)**2)
        E_v = float(np.clip(E_v_raw / self.v_ref, 0.0, 1.0))

        s_SM = self._sig(self.a_sm * (E_theta + self.mu_x * E_x) - self.b_sm)

        s_VF = self._sig(self.a_vf * (E_v - self.kappa_theta * E_theta) - self.b_vf)

        arg_LQR = self.a_lqr * (self.c_near - (E_theta + self.mu_x * E_x)) - self.b_lqr
        one_minus_v = max(0.0, 1.0 - E_v)
        s_LQR = self._sig(arg_LQR) * (one_minus_v + 1e-8)**self.rho_lqr

        return np.array([s_SM, s_VF, s_LQR], dtype=np.float32)

    def _reward(self, st):
        st = np.asarray(st, dtype=np.float32)
        x, a = float(st[0]), float(st[2])
        r = 0
        r = self.a_alive

        V = float(st.T @ self.P @ st)
        r -= self.a_states * V

        if self.prev_V is None: self.prev_V = V
        r += self.a_prog * (self.prev_V - V)
        self.prev_V = V

        # center = (x * 10 / (self.inv_pendulum.x_max))**2
        # r += self.a_center if abs(x) < 0.025 else -self.a_center * center

        # r -= self.a_theta * abs(np.sin(a))**2

        hx = np.exp(-0.5 * (x / self.sx)**2)
        ha = np.exp(-0.5 * (a / self.sa)**2)

        r += self.a_center * hx + self.a_theta * ha + self.a_origin * (hx * ha)

        scores = self._controller_scores(st)
        m = int(self.current_mode)
        z = np.asarray(scores, dtype=np.float64) / max(1e-8, self.tau_pref)
        z -= z.max()
        q = np.exp(z)
        q /= q.sum()
        rs = float(self.a_pref * (np.log(q[m] + 1e-8) + self.k_s))

        r += rs

        self.scores_values = (scores[0], scores[1], scores[2], rs)

        # r -= self.a_force * np.clip(self.force, -self.inv_pendulum.f_max, self.inv_pendulum.f_max)**2

        return float(r)
    
    def _sample_x0_rad(self):
        direction = np.random.normal(0, 1, 4)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        distance = np.random.uniform(0, self.scale_factor)
        state = direction * distance * self.max_vals
        x0 = np.array([
            float(np.clip(state[0], -self.max_vals[0], self.max_vals[0])),
            float(np.clip(state[1], -self.max_vals[1], self.max_vals[1])),
            float(np.clip(state[2], -self.max_vals[2], self.max_vals[2])),
            float(np.clip(state[3], -self.max_vals[3], self.max_vals[3]))
        ])

        return x0
