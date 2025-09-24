import numpy as np
from typing import List

class SelfAdaptingCurriculum:
    def __init__(
        self,
        min_difficulty: float = 0.1,
        max_difficulty: float = 1.0,
        cooldown_period: int = 20,
        variance_threshold: float = 0.001,
        ema_weight: float = 0.95,          
        slope_threshold: float = 5e-2,    
        warmup_generations: int = 20,    
        use_ema_variance: bool = True,
        slope_stable_period: int = 5,
        require_positive_slope: bool = True  # New: require positive slope (improving performance)
    ):
        self.min_difficulty = min_difficulty
        self.max_difficulty = max_difficulty
        self.current_difficulty = min_difficulty

        self.generation_count = 0
        self.cooldown_counter = 0
        self.stable_slope_counter = 0

        self.ema_weight = float(ema_weight)
        self.slope_threshold = float(slope_threshold)
        self.warmup_generations = int(warmup_generations)
        self.use_ema_variance = bool(use_ema_variance)
        self.slope_stable_period = int(slope_stable_period)
        self.require_positive_slope = bool(require_positive_slope)  # New parameter
        self.slope = 0.0

        self.ema_mean = None
        self.prev_ema_mean = None
        self.ema_variance = None

        self.variance_threshold = variance_threshold
        self.max_vals = np.array([1.5, 2.75, 0.525, 1.5])
        self.cooldown_period = cooldown_period

    def get_initial_conditions(self, num_conditions: int) -> List[List[float]]:
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
        if prev_ema is None:
            return float(value)
        w = self.ema_weight
        return float(w * prev_ema + (1.0 - w) * value)

    def update_difficulty(self, evaluated_population, extra_metrics=None):
        self.generation_count += 1

        return 1

        # fitness_scores = np.asarray([ind[1] for ind in evaluated_population], dtype=np.float64)
        # avg_fitness = float(np.mean(fitness_scores))
        # raw_variance = float(np.var(fitness_scores))

        # # Store previous EMA before updating
        # self.prev_ema_mean = self.ema_mean
        
        # # Update EMA mean
        # self.ema_mean = self._ema_update(self.ema_mean, avg_fitness)

        # if self.use_ema_variance:
        #     self.ema_variance = self._ema_update(self.ema_variance, raw_variance)
        #     fitness_variance_for_logic = self.ema_variance
        # else:
        #     fitness_variance_for_logic = raw_variance

        # # Calculate slope as difference between current and previous EMA
        # slope_ready = (self.prev_ema_mean is not None) and (self.generation_count >= self.warmup_generations)
        
        # if slope_ready:
        #     # Slope is the change in EMA mean (proxy for derivative)
        #     self.slope = self.ema_mean - self.prev_ema_mean
            
        #     # Check conditions for stable/improving performance
        #     if self.require_positive_slope:
        #         # Require positive slope (improving performance) that's not too large
        #         slope_condition = (self.slope > 0) and (self.slope < self.slope_threshold)
        #         slope_status = "positive and stable"
        #     else:
        #         # Original condition: slope near zero (stable performance)
        #         slope_condition = abs(self.slope) < self.slope_threshold
        #         slope_status = "stable"
            
        #     if slope_condition:
        #         self.stable_slope_counter += 1
        #     else:
        #         self.stable_slope_counter = 0  # Reset if condition is not met
            
        #     print(f"Gen {self.generation_count}: Difficulty={self.current_difficulty:.3f}, "
        #           f"EMA={self.ema_mean:.3f}, Slope={self.slope:.6f}, "
        #           f"StableCount={self.stable_slope_counter}/{self.slope_stable_period}, "
        #           f"Cooldown={self.cooldown_counter}/{self.cooldown_period}, "
        #           f"Condition={slope_status}")

        #     # Check if slope has met conditions for required period AND cooldown period has passed
        #     if (self.stable_slope_counter >= self.slope_stable_period and 
        #         self.cooldown_counter >= self.cooldown_period):
                
        #         if fitness_variance_for_logic < self.variance_threshold:
        #             # Low variance: small increase
        #             new_difficulty = min(self.current_difficulty + 0.01, self.max_difficulty)
        #             print(f"  -> Small increase (low variance): {self.current_difficulty:.3f} -> {new_difficulty:.3f}")
        #             self.current_difficulty = new_difficulty
        #         else:
        #             # High variance: larger increase
        #             new_difficulty = min(self.current_difficulty + 0.05, self.max_difficulty)
        #             print(f"  -> Large increase (high variance): {self.current_difficulty:.3f} -> {new_difficulty:.3f}")
        #             self.current_difficulty = new_difficulty
                
        #         # Reset both counters after difficulty increase
        #         self.cooldown_counter = 0
        #         self.stable_slope_counter = 0
                
        #     else:
        #         self.cooldown_counter += 1
        # else:
        #     self.cooldown_counter += 1
        #     if self.generation_count < self.warmup_generations:
        #         print(f"Gen {self.generation_count}: Warming up... ({self.generation_count}/{self.warmup_generations})")
        #     else:
        #         print(f"Gen {self.generation_count}: Waiting for EMA history...")

        # return self.current_difficulty

    def get_difficulty(self) -> float:
        return self.current_difficulty