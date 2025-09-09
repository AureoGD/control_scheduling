import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from environment.cart_pendulum.inverted_pendulum_dynamics import InvePendulum
from environment.cart_pendulum.pendulum_animation import PendulumLiveRenderer
import time


class InvPendulumEnv(gym.Env):

    def __init__(self, env_id=None, dt=0.001, max_step=500, rendering=False, frame_rate=30):
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

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(5, ), dtype=np.float32)

        self.scale_factor = 1
        self.ep_reward = 0
        self.current_step = 0
        self.ep = 0

        self.P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 5, 0], [0, 0, 0, 1]])
        self.theta_max = np.pi / 2

    # def step(self, action):
    #     self.current_step += 1
    #     self.force = self.inv_pendulum.f_max * np.clip(action[0], -1, 1)
    #     sys_sts = self.inv_pendulum.step_sim(self.force)
    #     new_state = self._norm(sys_sts)
    #     if self.rendering and (self.current_step % (self.frame_rate / 10) == 0 or self.current_step == 0):
    #         self.render()
    #         time.sleep(self.dt * self.frame_rate)

    #     reward = self._reward(sys_sts)

    #     terminated = (abs(sys_sts[2]) > self.theta_max)
    #     truncated = self.current_step >= self.max_step

    #     info = {}
    #     if truncated and not terminated:
    #         info["TimeLimit.truncated"] = True
    #         info["terminal_observation"] = sys_sts
    #     info["raw_state"] = sys_sts
    #     info["control_effort"] = self.force

    #     return new_state, reward, terminated, truncated, info
    def step(self, action):
        # --- 1) Action handling (robust to shapes) ---
        a = float(np.asarray(action).squeeze())
        a = np.clip(a, -1.0, 1.0)
        self.force = float(self.inv_pendulum.f_max) * a

        # --- 2) Bookkeeping BEFORE horizon check to avoid off-by-one ---
        self.current_step += 1

        # --- 3) Integrate dynamics ---
        sys_sts = self.inv_pendulum.step_sim(self.force)

        # (Optional) render at ~10 Hz of your configured frame_rate
        if self.rendering and (self.current_step % max(1, int(self.frame_rate / 10)) == 0 or self.current_step == 0):
            self.render()
            # avoid negative sleeps if frame_rate/dt changes unexpectedly
            time.sleep(max(0.0, self.dt * self.frame_rate))

        # --- 4) Safety: NaN/Inf guard (treat as failure) ---
        if not np.all(np.isfinite(sys_sts)):
            terminated = True
            truncated = False
            info = {
                "nan_terminated": True,
                "raw_state": sys_sts,
                "control_effort": self.force,
                "episode_progress": self.current_step / self.max_step,
            }
            reward = self._reward(sys_sts)  # or a specific catastrophic penalty if you prefer
            new_state = self._norm(np.nan_to_num(sys_sts, copy=False))
            return new_state, reward, terminated, truncated, info

        # --- 5) Termination / truncation logic (mutually exclusive) ---
        theta = float(sys_sts[2])
        terminated = (abs(theta) > self.theta_max)

        horizon_reached = (self.current_step >= self.max_step)
        truncated = (horizon_reached and not terminated)  # critical: do NOT truncate if already terminated

        # --- 6) Reward after state update ---
        reward = self._reward(sys_sts)

        # --- 7) Info dict (Gymnasium-friendly) ---
        info = {
            "raw_state": sys_sts,
            "control_effort": self.force,
            "episode_progress": self.current_step / self.max_step,  # optional, nice for TB
        }
        if truncated:
            # Gymnasium time-limit convention; SB3 VecMonitor will expose this in final_info
            info["TimeLimit.truncated"] = True
            info["terminal_observation"] = sys_sts

        # (Optional) If you have a separate success criterion:
        # info["is_success"] = bool(success_condition)

        # --- 8) Normalized observation out ---
        new_state = self._norm(sys_sts)
        return new_state, reward, terminated, truncated, info

    def _define_constants(self):
        self.a_states = 5e-3
        self.a_theta = 2.0e-3
        self.a_center = 2.5e-3
        self.a_origin = 1.5e-3
        self.a_prog = 3e-3
        self.a_force = 2.5e-4
        self.a_alive = 8e-3
        self.a_df = 1.5e-4

        sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
        x_tol = 0.5  # at |x|=x_tol -> 0.5 bonus
        a_tol = 0.2  # at |a|=a_tol -> 0.5 bonus
        self.sx = max(1e-9, x_tol / sqrt_2ln2)
        self.sa = max(1e-9, a_tol / sqrt_2ln2)

        self.force = 0
        self.prev_V = None
        self.prev_force = None

        self.current_step = 0
        self.ep_reward = 0

    def set_difficulty(self, value: float):
        self.scale_factor = float(value)

    def reset(self, *, seed=None, options=None, x0=None):
        super().reset(seed=seed)
        self.ep += 1
        ep_r = self.ep_reward
        self._define_constants()
        if x0 is None:
            # x0 = self.scale_factor * np.array([
            #     np.random.uniform(-1.5, 1.5),
            #     np.random.uniform(-2.5, 2.5),
            #     np.random.uniform(-0.5, 0.5),
            #     np.random.uniform(-2.5, 2.5)
            # ])
            x0 = self._sample_x0(self.scale_factor)
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

    def _norm(self, states):
        x = states[0] / self.inv_pendulum.x_max
        dx = states[1] / self.inv_pendulum.v_max
        a = states[2]

        cos_a = np.cos(a)
        sin_a = np.sin(a)
        da = states[3] / self.inv_pendulum.da_max
        return np.array([x, dx, cos_a, sin_a, da]).reshape(5, )

    def _reward(self, st):
        st = np.asarray(st, dtype=np.float32)
        x, a = float(st[0]), float(st[2])

        r = self.a_alive

        V = float(st.T @ self.P @ st)
        r -= self.a_states * V

        if self.prev_V is None: self.prev_V = V
        r += self.a_prog * (self.prev_V - V)
        self.prev_V = V

        # center = (x * 10 / (self.inv_pendulum.x_max))**2
        # r += self.a_center if abs(x) < 0.025 else -self.a_center * center

        # r += self.a_theta if abs(a) < 0.025 else -self.a_theta * (np.sin(a)**2)

        hx = np.exp(-0.5 * (x / self.sx)**2)
        ha = np.exp(-0.5 * (a / self.sa)**2)

        r += self.a_center * hx + self.a_theta * ha + self.a_origin * (hx * ha)

        force = np.clip(self.force, -self.inv_pendulum.f_max, self.inv_pendulum.f_max)

        r -= self.a_force * force**2

        # if self.prev_force is None: self.prev_force = force

        # df = force - self.prev_force

        # r -= self.a_df * df**2

        return float(r)

    def _sample_x0(self, s: float) -> np.ndarray:
        # Max ranges per dimension
        R = np.array([1.5, 2.5, 0.5, 2.5], dtype=float)  # [x, dx, theta, dtheta]

        # Draw base uniform in [-1,1]^4
        u = np.random.uniform(-1.0, 1.0, size=4)

        # Per-dimension difficulty gains: positions scale ~ s, velocities ~ s^0.5
        g = np.array([s, s**0.5, s, s**0.5], dtype=float)

        x0 = (R * g) * u

        # Keep theta inside termination margin
        theta_lim = min(self.theta_max * 0.95, R[2])
        x0[2] = np.clip(x0[2], -theta_lim, theta_lim)

        return x0
