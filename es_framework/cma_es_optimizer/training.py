import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import torch
import multiprocessing
from functools import partial
from es_framework.commons.worker import init_worker, run_worker
from es_framework.commons.nn_parameters import flatten_nn_parameters, unflatten_nn_parameters
from es_framework.commons.logger import TrainingLogger
from es_framework.cma_es_optimizer.cma_es_optimizer import CMAESOptimizer
from es_framework.commons.control_rule import ControlRule

# Set this to True to train using the discrete environment
is_discrete = True

# CMA-ES constants
POPULATION_SIZE = 100
GENERATIONS = 10000
SIGMA = 1.5
ELITE_FRACTION = 0.5

# NN model variables
fc1_dim = 64
fc2_dim = 64

# Logging variables
alg = 'cmaes'


def main():
    logger = TrainingLogger(discrete=is_discrete, alg=alg)

    config = {'model_config': {'fc1_dim': fc1_dim, 'fc2_dim': fc2_dim, 'discrete': is_discrete}}

    if is_discrete:
        from scheduller_rules.schl_rule1 import SchedullerRule
        sw_rule = SchedullerRule()
        config['is_discrete'] = True
        config['env_sw_config'] = {'sw_rule': sw_rule}
        output_dim = sw_rule.n_controllers
    else:
        config['is_discrete'] = False
        output_dim = 1

    reference_model = ControlRule(observation_dim=5, output_dim=output_dim, **config['model_config'])

    logger.set_reference_model(reference_model)
    param_dim = flatten_nn_parameters(reference_model).size

    optimizer = CMAESOptimizer(param_dim=param_dim, population_size=POPULATION_SIZE, elite_fraction=ELITE_FRACTION, sigma=SIGMA)
    optimizer.set_initial_mean_params(flatten_nn_parameters(reference_model))

    num_generations = GENERATIONS
    num_workers = min(15, POPULATION_SIZE)
    logger.log_message(f"Starting CMA-ES training with {num_workers} persistent parallel workers.")

    pool = multiprocessing.Pool(processes=num_workers, initializer=partial(init_worker, config=config))

    try:
        for gen in range(1, num_generations + 1):
            population_params = optimizer.sample_population()
            diff = 1
            tasks = [(i, params, diff) for i, params in enumerate(population_params)]
            results = pool.map(run_worker, tasks)
            results.sort(key=lambda x: x[0])
            fitness_scores = [score for _, score in results]

            evaluated_population = list(zip(population_params, fitness_scores))
            optimizer.update_distribution(evaluated_population)

            logger.log_generation(generation=gen, evaluated_population=evaluated_population, extra_metrics={"sigma": optimizer.sigma, "mean_stddev": np.mean(optimizer.D)})

    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")
    finally:
        logger.log_message("Closing worker pool and saving final model...")
        pool.close()
        pool.join()

        final_best_weights = optimizer.get_best_params()
        final_model_state_dict = unflatten_nn_parameters(final_best_weights, reference_model)
        final_model_path = os.path.join(logger.models_save_dir, "cmaes_model_final_mean.pth")
        torch.save(final_model_state_dict, final_model_path)
        logger.log_message(f"Final CMA-ES mean weights saved to {final_model_path}")
        logger.close()


if __name__ == "__main__":
    main()
