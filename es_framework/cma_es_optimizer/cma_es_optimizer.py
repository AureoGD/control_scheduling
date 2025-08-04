import numpy as np
from typing import List, Tuple


class CMAESOptimizer:

    def __init__(self, param_dim: int, population_size: int = 50, elite_fraction=0.5, sigma: float = 0.5):
        self.param_dim = param_dim
        self.sigma = sigma
        self.population_size = population_size
        self.elite_fraction = elite_fraction

        self.generation = 0
        self.mu = int(self.population_size * self.elite_fraction)

        # Search distribution mean
        self.mean = np.zeros(param_dim)

        # Stores standard normal samples z ~ N(0, I)
        self._z_vectors = []

        # Covariance matrix and its decomposition
        self.C = np.identity(param_dim)
        self.B = np.identity(param_dim)
        self.D = np.ones(param_dim)
        self.inv_sqrt_C = np.identity(param_dim)

        # Evolution paths
        self.p_sigma = np.zeros(param_dim)
        self.p_c = np.zeros(param_dim)

        # Recombination weights and effective mass
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1. / np.sum(self.weights**2)

        # Learning rates and damping factors
        self.c_sigma = (self.mu_eff + 2) / (param_dim + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (param_dim + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / param_dim) / (param_dim + 4 + 2 * self.mu_eff / param_dim)
        self.c1 = 2 / ((param_dim + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((param_dim + 2)**2 + self.mu_eff))

        # Expected value of ||N(0,I)|| for sigma adaptation
        self.chi_N = np.sqrt(param_dim) * (1 - 1 / (4 * param_dim) + 1 / (21 * (param_dim**2)))

        self.eig_update_counter = 0

    def set_initial_mean_params(self, initial_params: np.ndarray) -> None:
        if initial_params.shape[0] != self.param_dim:
            raise ValueError(f"Expected param_dim = {self.param_dim}, but got {initial_params.shape[0]}")
        self.mean = initial_params.copy()

    def sample_population(self) -> List[np.ndarray]:
        self._z_vectors = []
        population = []

        for _ in range(self.population_size):
            # Sample from standard normal distribution — Eq. (38)
            z = np.random.randn(self.param_dim).astype(np.float32)
            self._z_vectors.append(z)

            # Compute correlated mutation — Eq. (39)
            y = self.B @ (self.D * z)

            # Generate new candidate — Eq. (40)
            individual_params = self.mean + self.sigma * y
            population.append(individual_params)

        return population

    def update_distribution(self, evaluated_population: List[Tuple[np.ndarray, float]]) -> None:
        evaluated_population.sort(key=lambda x: x[1], reverse=True)
        elite_params = [ind[0] for ind in evaluated_population[:self.mu]]
        elite_z = [self._z_vectors[i] for i in range(self.mu)]

        # Step 1: Update mean — Eq. (43)
        elite_params_array = np.array(elite_params)
        self.mean = np.sum(self.weights[:, None] * elite_params_array, axis=0)

        # Step 2: Compute weighted average of z vectors
        Z = np.array(elite_z)
        z_mean = np.sum(self.weights[:, None] * Z, axis=0)

        # Step 3: Update evolution path for sigma — Eq. (47)
        self.p_sigma = (1 - self.c_sigma) * self.p_sigma + \
                       np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * z_mean

        # Step 4: Adapt global step-size sigma — Eq. (48)
        norm_ps = np.linalg.norm(self.p_sigma)
        self.sigma *= np.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chi_N - 1))

        # Step 5: Compute y_mean = B D z_mean
        y_mean = self.B @ (self.D * z_mean)

        # Step 6: Update evolution path for covariance — Eq. (45)
        threshold = 1.4 + 2 / (self.param_dim + 1)
        expected_norm = np.sqrt(1 - (1 - self.c_sigma)**(2 * self.eig_update_counter))
        expected_norm = max(expected_norm, 1e-8)
        h_sigma = int((norm_ps / expected_norm) < threshold)

        self.p_c = (1 - self.c_c) * self.p_c + \
                   h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff) * y_mean

        # Step 7: Update covariance matrix — Eq. (44)
        C_mu = np.zeros((self.param_dim, self.param_dim))
        for i in range(self.mu):
            y_i = self.B @ (self.D * elite_z[i])
            C_mu += self.weights[i] * np.outer(y_i, y_i)

        self.C = (1 - self.c1 - self.c_mu) * self.C + \
                 self.c1 * np.outer(self.p_c, self.p_c) + \
                 self.c_mu * C_mu

        # Step 8: Periodic eigen decomposition of C
        self.eig_update_counter += 1
        if self.eig_update_counter % (self.param_dim // 10 + 1) == 0:
            self.C = (self.C + self.C.T) / 2  # Enforce symmetry
            self.D, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(self.D, 1e-20))
            self.inv_sqrt_C = self.B @ np.diag(1. / self.D) @ self.B.T

        # Logging
        best_fitness = evaluated_population[0][1]
        print(f"CMA-ES Updated. Best Fitness: {best_fitness:.4f}, "
              f"Step size (sigma): {self.sigma:.5f}, "
              f"||p_sigma||: {norm_ps:.4f}")

    def get_best_params(self) -> np.ndarray:
        return self.mean
