# from stable_baselines3.common.callbacks import BaseCallback
# import numpy as np

# class CurriculumCallback(BaseCallback):

#     def __init__(self, total_timesteps: int, step=0.025, ramp_portion=0.80):
#         super().__init__()
#         self.T = int(total_timesteps)
#         self.step = float(step)  # e.g., 0.025
#         self.ramp = float(ramp_portion)  # fraction of training to ramp
#         self.last = None
#         self.venv = None

#     def _on_training_start(self) -> None:
#         self.venv = self.model.get_env()
#         self._apply_if_changed(force=True)

#     def _current_diff(self) -> float:
#         # progress normalized to the ramp window [0,1]
#         denom = max(1, int(self.ramp * self.T))
#         p = min(1.0, self.num_timesteps / denom)
#         # floor bucket index
#         k = int(np.floor(p / self.step))
#         # add +1 so we start at 1*step = 0.025 instead of 0.0
#         d = min(1.0, (k + 1) * self.step)
#         return float(d)

#     def _apply_if_changed(self, force=False):
#         d = self._current_diff()
#         if force or (self.last is None) or (d != self.last):
#             for env in self.venv.envs:
#                 env.unwrapped.set_difficulty(d)
#             self.last = d
#             self.logger.record("curriculum/scale_factor", d)

#     def _on_step(self) -> bool:
#         self._apply_if_changed(force=False)
#         return True
# sb3_framework/curriculum_callback.py

# sb3_framework/curriculum_callback.py
# sb3_framework/curriculum_callback.py

from __future__ import annotations
from typing import Deque, Optional, Dict, Any, List
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumByPerformance(BaseCallback):
    """
    Performance-driven curriculum with:
      • Robust truncation detection (flag OR episode_len ≈ horizon)
      • One-step hysteresis floor (can fall back at most one level)
      • Minimum residency per level (e.g., 5% of total timesteps)

    Works with PPO/A2C/DQN (vectorized).
    Env must implement: env.unwrapped.set_difficulty(scale: float)
    Expects Gymnasium step signature: (obs, reward, terminated, truncated, info)
    """

    def __init__(
        self,
        *,
        total_timesteps: int,
        init_scale: float = 0.10,
        max_scale: float = 1.00,
        min_scale: float = 0.10,  # base floor (initial allowed minimum)
        delta_up: float = 0.10,  # level step size
        delta_down: float = 0.05,  # backoff step size (respects hysteresis floor)
        window_episodes: int = 250,
        trunc_mastery_ratio: float = 0.70,
        fail_backoff_ratio: float = 0.70,
        min_episodes_before_change: int = 50,
        # DQN exploration bump on level-up
        dqn_eps_bump: float = 0.20,
        # Logging/customization
        log_prefix: str = "curriculum",
        # Episode horizon (pass if fixed, e.g., 500). If None, auto-detect from env.unwrapped.max_step
        horizon_hint: Optional[int] = None,
        # Additional generic cooldown (in timesteps). Often keep at 0 when using residency.
        min_steps_between_changes: int = 0,
        # Minimum residency per level as a fraction of TOTAL timesteps (e.g., 0.05 -> 5%)
        residency_frac: float = 0.05,
        # Quantize scales to the delta_up grid to avoid float drift
        snap_to_step_grid: bool = True,
        # Debug prints
        debug_print: bool = False,
    ):
        super().__init__()
        self.T = int(total_timesteps)
        self.scale: float = float(init_scale)
        self.max_scale = float(max_scale)
        self.min_scale = float(min_scale)  # static base floor provided by user
        self.delta_up = float(delta_up)
        self.delta_down = float(delta_down)
        self.window_episodes = int(window_episodes)
        self.trunc_mastery_ratio = float(trunc_mastery_ratio)
        self.fail_backoff_ratio = float(fail_backoff_ratio)
        self.min_episodes_before_change = int(min_episodes_before_change)
        self.dqn_eps_bump = float(dqn_eps_bump)
        self.log_prefix = log_prefix
        self.horizon_hint = horizon_hint
        self.min_steps_between_changes = int(min_steps_between_changes)
        self.residency_frac = float(residency_frac)
        self.min_residency_steps = int(self.residency_frac * self.T)
        self.snap_to_step_grid = bool(snap_to_step_grid)
        self.debug_print = debug_print

        # Rolling window of outcomes: +1 (truncation), 0 (success), -1 (failure)
        self._window: Deque[int] = deque(maxlen=self.window_episodes)

        # Stats
        self._n_levels: int = 0
        self._last_change_step: int = 0
        self._has_changed_once: bool = False  # residency applies only after the first change

        # Hysteresis state
        self._best_scale: float = float(init_scale)  # highest scale ever achieved
        self._min_dynamic: float = float(min_scale)  # moving floor; starts at base min

        # Vec env + horizon + per-env episode length counters
        self.venv = None
        self._horizon: Optional[int] = None
        self._n_envs: int = 0
        self._ep_len: Optional[np.ndarray] = None  # shape (n_envs,)

        # Snap initial scales to grid
        self.scale = self._snap(self.scale)
        self._best_scale = self._snap(self._best_scale)
        self._min_dynamic = max(self.min_scale, self._snap(self._best_scale - self.delta_up))

    # -------------------------------------------------------------------------
    # SB3 hooks
    # -------------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self.venv = self.model.get_env()

        # Detect episode horizon
        detected = None
        try:
            envs = getattr(self.venv, "envs", [])
            if envs:
                detected = getattr(envs[0].unwrapped, "max_step", None)
        except Exception:
            detected = None

        self._horizon = self.horizon_hint if self.horizon_hint is not None else detected

        # Determine number of envs and init counters
        try:
            self._n_envs = int(getattr(self.venv, "num_envs", len(getattr(self.venv, "envs", [])) or 1))
        except Exception:
            self._n_envs = 1
        self._ep_len = np.zeros(self._n_envs, dtype=np.int64)

        if self.debug_print:
            print(f"[Curriculum] horizon_hint={self.horizon_hint}, detected_horizon={detected}, using={self._horizon}")
            print(f"[Curriculum] n_envs={self._n_envs}, min_residency_steps={self.min_residency_steps}")

        self._apply_scale()
        self._log_scalars()

    def _on_step(self) -> bool:
        """
        Called after each env step. We:
          1) Increment all per-env episode lengths.
          2) For envs where `dones[i]` is True, classify truncation vs. failure (or success).
          3) Decide promotions/demotions subject to residency + cooldown.
        """
        if self._ep_len is None:
            return True

        # 1) increment lengths
        self._ep_len += 1

        # 2) check which envs ended this step
        dones = self.locals.get("dones", None)
        infos: List[Dict[str, Any]] = self.locals.get("infos", [])

        if dones is not None:
            dones = np.asarray(dones).reshape(-1)
            for i in range(min(self._n_envs, dones.shape[0])):
                if not dones[i]:
                    continue

                finfo = {}
                if isinstance(infos, (list, tuple)) and i < len(infos) and isinstance(infos[i], dict):
                    finfo = infos[i]
                    if "final_info" in finfo and isinstance(finfo["final_info"], dict):
                        finfo = finfo["final_info"]

                classified = False
                # Prefer explicit flags when present
                if self._has_trunc_flag(finfo):
                    self._window.append(+1)
                    classified = True
                    if self.debug_print:
                        print(f"[Curriculum] TRUNC (flag) env={i}, len={self._ep_len[i]}")
                elif self._has_success_flag(finfo):
                    self._window.append(0)
                    classified = True
                    if self.debug_print:
                        print(f"[Curriculum] SUCCESS (flag) env={i}, len={self._ep_len[i]}")

                # Fallback: use length vs horizon
                if not classified:
                    H = self._horizon
                    if H is not None and abs(int(self._ep_len[i]) - int(H)) <= 1:
                        self._window.append(+1)
                        if self.debug_print:
                            print(f"[Curriculum] TRUNC (len≈H) env={i}, len={self._ep_len[i]}, H={H}")
                    else:
                        self._window.append(-1)
                        if self.debug_print:
                            print(f"[Curriculum] FAIL (len={self._ep_len[i]}) env={i} (no flags)")

                # reset this env's counter for next episode
                self._ep_len[i] = 0

        # 3) decide changes if we have enough data
        n = len(self._window)
        if n >= self.min_episodes_before_change:
            arr = np.asarray(self._window, dtype=np.int32)
            trunc_ratio = float((arr == +1).mean()) if n > 0 else 0.0
            fail_ratio = float((arr == -1).mean()) if n > 0 else 0.0

            if trunc_ratio >= self.trunc_mastery_ratio and self._can_change_level():
                self._level_up()
                self._window.clear()

            elif self.delta_down > 0.0 and fail_ratio >= self.fail_backoff_ratio and self.scale > self._effective_min(
            ) and self._can_change_level():
                self._level_down()
                for _ in range(self.window_episodes // 2):
                    if self._window:
                        self._window.popleft()

            self.logger.record(f"{self.log_prefix}/trunc_ratio", trunc_ratio)
            self.logger.record(f"{self.log_prefix}/fail_ratio", fail_ratio)

        self._log_scalars()
        return True

    # -------------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------------

    def _log_scalars(self) -> None:
        self.logger.record(f"{self.log_prefix}/scale_factor", self.scale)
        self.logger.record(f"{self.log_prefix}/levels_achieved", self._n_levels)
        self.logger.record(f"{self.log_prefix}/min_floor", self._min_dynamic)
        self.logger.record(f"{self.log_prefix}/best_scale", self._best_scale)
        # Residency diagnostics
        elapsed = self.num_timesteps - self._last_change_step
        needed = self.min_residency_steps if self._has_changed_once else 0
        frac = (elapsed / needed) if needed > 0 else 1.0
        self.logger.record(f"{self.log_prefix}/residency_elapsed_steps", elapsed)
        self.logger.record(f"{self.log_prefix}/residency_needed_steps", needed)
        self.logger.record(f"{self.log_prefix}/residency_elapsed_frac", frac)

    def _can_change_level(self) -> bool:
        """Enforce both generic cooldown and minimum residency after a level change."""
        # Residency applies only AFTER the first change
        if self._has_changed_once and self.min_residency_steps > 0:
            if (self.num_timesteps - self._last_change_step) < self.min_residency_steps:
                return False
        # Optional extra cooldown
        if self.min_steps_between_changes > 0:
            if (self.num_timesteps - self._last_change_step) < self.min_steps_between_changes:
                return False
        return True

    def _apply_scale(self) -> None:
        if self.venv is None:
            return
        for e in getattr(self.venv, "envs", []):
            try:
                e.unwrapped.set_difficulty(self.scale)
            except Exception as exc:
                if self.debug_print:
                    print(f"[Curriculum] set_difficulty failed on one env: {exc}")

    def _maybe_bump_dqn_exploration(self) -> None:
        model = self.model
        if hasattr(model, "set_exploration_rate"):
            try:
                current = getattr(model, "exploration_rate", None)
                target = float(self.dqn_eps_bump)
                if current is None or current < target:
                    model.set_exploration_rate(target)
                    if self.debug_print:
                        print(f"[Curriculum] DQN exploration bumped to >= {target}")
            except Exception:
                pass

    def _snap(self, x: float) -> float:
        if not self.snap_to_step_grid or self.delta_up <= 0:
            return float(x)
        k = round(float(x) / self.delta_up)
        return float(np.clip(k * self.delta_up, 0.0, self.max_scale))

    def _effective_min(self) -> float:
        return float(max(self.min_scale, self._min_dynamic))

    def _update_min_floor_after_level_up(self) -> None:
        # After leveling up to S_k, min floor = max(old_floor, S_k - delta_up)
        candidate = self._snap(self._best_scale - self.delta_up)
        self._min_dynamic = max(self._min_dynamic, candidate, self.min_scale)

    def _level_up(self) -> None:
        old = self.scale
        new = self._snap(self.scale + self.delta_up)
        new = float(np.clip(new, self._effective_min(), self.max_scale))
        self.scale = new
        if self.scale != old:
            self._n_levels += 1
            self._last_change_step = self.num_timesteps
            self._has_changed_once = True
            if self.scale > self._best_scale:
                self._best_scale = self.scale
            self._update_min_floor_after_level_up()
            self._apply_scale()
            self._maybe_bump_dqn_exploration()
            if self.debug_print:
                print(f"[Curriculum] LEVEL UP -> scale={self.scale:.3f}, "
                      f"min_floor={self._min_dynamic:.3f} at step={self.num_timesteps}")

    def _level_down(self) -> None:
        old = self.scale
        eff_min = self._effective_min()
        new = self._snap(self.scale - self.delta_down)
        new = float(np.clip(new, eff_min, self.max_scale))
        self.scale = new
        if self.scale != old:
            self._last_change_step = self.num_timesteps
            self._has_changed_once = True
            self._apply_scale()
            if self.debug_print:
                print(f"[Curriculum] LEVEL DOWN -> scale={self.scale:.3f}, "
                      f"min_floor={self._min_dynamic:.3f} at step={self.num_timesteps}")

    # -------------------------------------------------------------------------
    # Flags & helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _has_trunc_flag(info: Dict[str, Any]) -> bool:
        if info.get("TimeLimit.truncated", False):
            return True
        if info.get("truncated", False) and not info.get("terminated", False):
            return True
        if "final_info" in info and isinstance(info["final_info"], dict):
            fi = info["final_info"]
            if fi.get("TimeLimit.truncated", False):
                return True
            if fi.get("truncated", False) and not fi.get("terminated", False):
                return True
        return False

    @staticmethod
    def _has_success_flag(info: Dict[str, Any]) -> bool:
        if "is_success" in info:
            try:
                return bool(info["is_success"])
            except Exception:
                return False
        if "final_info" in info and isinstance(info["final_info"], dict):
            fi = info["final_info"]
            if "is_success" in fi:
                try:
                    return bool(fi["is_success"])
                except Exception:
                    return False
        return False
