import numpy as np
import torch
from typing import List, Tuple
from es_framework.commons.nn_parameters import flatten_nn_parameters


class CEMOptimizer:

    def __init__(self,
                 param_dim: int,
                 population_size: int = 50,
                 elite_fraction: float = 0.5,
                 initial_std_dev: float = 0.5,
                 update_rule_type: str = "standard",
                 elite_weighting_type: str = "uniform",
                 noise_decay_factor: float = 0.995,
                 min_std_dev: float = 1e-3,
                 extra_noise_scale: float = 0.01):

        self.param_dim = param_dim
        self.population_size = population_size
        self.num_elites = max(1, int(population_size * elite_fraction))

        self.mean = np.zeros(param_dim, dtype=np.float32)
        self.std_devs = np.full(param_dim, initial_std_dev, dtype=np.float32)

        self.update_rule_type = update_rule_type
        self.elite_weighting_type = elite_weighting_type

        self.noise_decay_factor = noise_decay_factor
        self.min_std_dev = min_std_dev
        self.epsilon = extra_noise_scale

        # For "cmaes_type" rule, need the mean used for sampling the current generation
        self.old_mean = np.copy(self.mean)

    def set_initial_mean_params(self, initial_model: torch.nn.Module):
        self.mean = flatten_nn_parameters(initial_model)
        self.old_mean = np.copy(self.mean)  # Update mu_old as well

    def sample_population(self) -> List[np.ndarray]:
        # Store the mean used for this generation's sampling if using cmaes_type rule
        if self.update_rule_type == "cmaes_type":
            self.old_mean = np.copy(self.mean)

        population = []
        for _ in range(self.population_size):
            individual_params = self.mean + self.std_devs * np.random.randn(self.param_dim).astype(np.float32)
            population.append(individual_params)
        return population

    def _calculate_elite_weights(self) -> np.ndarray:
        """Calculates weights for elite individuals based on the configured type."""

        if self.elite_weighting_type == "logarithmic":
            # gives more importance to better individuals
            ranks = np.arange(1, self.num_elites + 1)
            raw_weights = np.log(self.num_elites + 1) - np.log(ranks)
            if np.sum(raw_weights) <= 0:
                return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

            weights = raw_weights / np.sum(raw_weights)
            return weights.astype(np.float32)
        else:
            # "each individual is given the same importance"
            return np.full(self.num_elites, 1.0 / self.num_elites, dtype=np.float32)

    def update_distribution(self, evaluated_population: List[Tuple[np.ndarray, float]]):
        """
        Upddate the weights distribution.
        
        Args:
            evaluated_population (List[Tuple[np.ndarray, float]]): _description_
        """

        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        elite_individuals_params = [ind[0] for ind in evaluated_population[:self.num_elites]]

        elite_params_array = np.array(elite_individuals_params, dtype=np.float32)

        # Calculate elite weights (lambda_i)
        lambda_ = self._calculate_elite_weights()

        # 1. Update mean using weighted average of elites. Eq. 1 from the paper.
        self.mean = np.average(elite_params_array, axis=0, weights=lambda_)

        # 2. Update std_devs using Eq. 3 (element-wise squaring) from the paper, avoiding the n×n outer product matrix
        if self.update_rule_type == "standard":
            squared_diffs = np.square(elite_params_array - self.mean)
        elif self.update_rule_type == "cmaes_type":
            squared_diffs = np.square(elite_params_array - self.old_mean)

        # First, calculate the weighted average to get the new variance
        new_variances = np.average(squared_diffs, axis=0, weights=lambda_)

        # Then, add the noise to the final calculated variance
        self.std_devs = np.sqrt(new_variances + self.epsilon)

        # 4. Ensure std_devs do not collapse
        self.std_devs = np.maximum(self.std_devs, self.min_std_dev)

        # 4. Decay the extra noise scale
        self.epsilon *= self.noise_decay_factor
        self.epsilon = max(self.epsilon, self.min_std_dev)

        best_fitness_this_gen = evaluated_population[0][1]
        print(f"CEM Distribution Updated. Best Fitness: {best_fitness_this_gen:.4f}, "
              f"Mean StdDev: {np.mean(self.std_devs):.6f}, "
              f"Current Extra Noise Scale: {self.epsilon:.6f}")

    def get_best_params(self) -> np.ndarray:
        return self.mean
