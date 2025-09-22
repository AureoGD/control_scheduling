import numpy as np
from typing import List
from collections import deque  # kept only if you still want it elsewhere

class SelfAdaptingCurriculum:
    def __init__(
        self,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        cooldown_period: int = 20,
        variance_threshold: float = 0.01,
        ema_weight: float = 0.75,          # w in f(t) = w*f(t-1) + (1-w)*data[t]
        slope_threshold: float = 5e-2,    # |ΔEMA| below this -> treat as “flat”
        warmup_generations: int = 20,      # wait a few gens before acting on EMA
        use_ema_variance: bool = False    # optionally smooth variance as well
    ):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = min_difficulty

        # Adaptive state
        self.generation_count = 0
        self.cooldown_counter = 0

        # EMA parameters
        self.ema_weight = float(ema_weight)
        self.slope_threshold = float(slope_threshold)
        self.warmup_generations = int(warmup_generations)
        self.use_ema_variance = bool(use_ema_variance)
        self.slope = 0

        # EMA trackers (initialized on first update)
        self.ema_mean = None
        self.prev_ema_mean = None
        self.ema_variance = None

        # Convergence threshold
        self.variance_threshold = variance_threshold

        # Bounds for initial condition sampling
        self.max_vals = np.array([1.5, 2.75, 0.525, 1.5])

        # Cooldown
        self.cooldown_period = cooldown_period

    def get_initial_conditions(self, num_conditions: int) -> List[List[float]]:
        """Generate initial conditions based on current difficulty."""
        conditions = []
        for _ in range(num_conditions):
            direction = np.random.normal(0, 1, 4)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            distance = np.random.uniform(0, self.current_difficulty)
            state = direction * distance * self.max_vals

            conditions.append([
                float(np.clip(state[0], -self.max_vals[0], self.max_vals[0])),
                float(np.clip(state[1], -self.max_vals[1], self.max_vals[1])),
                float(np.clip(state[2], -self.max_vals[2], self.max_vals[2])),
                float(np.clip(state[3], -self.max_vals[3], self.max_vals[3]))
            ])
        return conditions

    def _ema_update(self, prev_ema: float | None, value: float) -> float:
        """
        EMA update: f(t) = w * f(t-1) + (1 - w) * value
        If prev_ema is None (first sample), return value to bootstrap.
        """
        if prev_ema is None:
            return float(value)
        w = self.ema_weight
        return float(w * prev_ema + (1.0 - w) * value)

    def update_difficulty(self, evaluated_population, extra_metrics=None):
        """
        Self-adapting difficulty update using EMA for trend (slope) detection.
        We estimate the local slope by ΔEMA = EMA_t - EMA_{t-1}.
        """
        self.generation_count += 1

        # --- Extract metrics ---
        fitness_scores = np.asarray([ind[1] for ind in evaluated_population], dtype=np.float64)
        avg_fitness = float(np.mean(fitness_scores))
        raw_variance = float(np.var(fitness_scores))

        # --- Update EMAs ---
        self.prev_ema_mean = self.ema_mean
        self.ema_mean = self._ema_update(self.ema_mean, avg_fitness)

        if self.use_ema_variance:
            self.ema_variance = self._ema_update(self.ema_variance, raw_variance)
            fitness_variance_for_logic = self.ema_variance
        else:
            fitness_variance_for_logic = raw_variance
        self.slope = self.ema_mean
        # --- Slope estimation via EMA difference ---
        slope_ready = (self.prev_ema_mean is not None) and (self.generation_count >= self.warmup_generations)
        slope = 0.0
        if slope_ready:
            slope = self.ema_mean - self.prev_ema_mean  # per-generation slope proxy

        
        
        # --- Difficulty logic ---
        if slope_ready:
            # If trend is flat AND cooldown elapsed -> increase difficulty
            if abs(slope) < self.slope_threshold and self.cooldown_counter >= self.cooldown_period:
                if fitness_variance_for_logic < self.variance_threshold:
                    # Converged population -> smaller bump
                    self.current_difficulty = min(self.current_difficulty + 0.05, self.max_difficulty)
                else:
                    # Still diverse -> larger bump
                    self.current_difficulty = min(self.current_difficulty + 0.10, self.max_difficulty)
                self.cooldown_counter = 0
            else:
                # Keep counting until next opportunity
                self.cooldown_counter += 1
        else:
            # Warmup phase: do not change difficulty based on slope yet
            self.cooldown_counter += 1

        # Optional “anti-stagnation” nudge:
        # if variance is too low (over-converged), reduce difficulty slightly to
        # keep training stable (this mirrors your original behavior).
        # if fitness_variance_for_logic < self.variance_threshold:
        #     self.current_difficulty = max(self.current_difficulty - 0.05, self.min_difficulty)

        return self.current_difficulty

    def get_difficulty(self) -> float:
        return self.current_difficulty
