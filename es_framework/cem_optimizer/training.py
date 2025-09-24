import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import multiprocessing
from functools import partial

from es_framework.commons.worker import init_worker, run_worker
from es_framework.commons.nn_parameters import flatten_nn_parameters, unflatten_nn_parameters
from es_framework.commons.logger import TrainingLogger
from es_framework.cem_optimizer.cem_optimizer import CEMOptimizer
from es_framework.commons.control_rule import ControlRule
from es_framework.commons.initial_conditions import SelfAdaptingCurriculum

# --- Experiment Setup ---
is_discrete = False

# # --- CEM Hyperparameters ---
# POPULATION_SIZE = 100
# GENERATIONS = 1500
# ELITE_FRACTION = 0.25
# INITIAL_STD_DEV = 1.5
# EXTRA_NOISE_SCALE = 0.5
# NOISE_DECAY_FACTOR = 0.99
# MIN_STD_DEV = 0.001
# UPDATE_RULE = "standard"  # "standard" or "cmaes_type"
# ELITE_WEIGHTING = "uniform"  # "uniform" or "logarithmic"

# --- CEM Hyperparameters ---
POPULATION_SIZE = 100
GENERATIONS = 2500
ELITE_FRACTION = 0.25
INITIAL_STD_DEV = 2.5
EXTRA_NOISE_SCALE = 0.5
NOISE_DECAY_FACTOR = 0.99
MIN_STD_DEV = 0.001
UPDATE_RULE = "standard"  # "standard" or "cmaes_type"
ELITE_WEIGHTING = "uniform"  # "uniform" or "logarithmic"


# --- Neural Net Setup ---
fc1_dim = 64
fc2_dim = 64

# --- Logging Setup ---
log_dir = "es_framework"
algorithm = 'cem'


def difficulty_for_gen(gen: int, step_every: int = 200, step_size: float = 0.1, start: float = 0.0, end: float = 1.0):
    k = (gen - 1) // step_every
    diff = start + k * step_size
    return float(max(0.0, min(end, diff)))


def main():
    logger = TrainingLogger(discrete=is_discrete, alg=algorithm)

    # --- Environment / Model Config ---
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

    cem = CEMOptimizer(param_dim=param_dim,
                       population_size=POPULATION_SIZE,
                       elite_fraction=ELITE_FRACTION,
                       initial_std_dev=INITIAL_STD_DEV,
                       update_rule_type=UPDATE_RULE,
                       elite_weighting_type=ELITE_WEIGHTING,
                       noise_decay_factor=NOISE_DECAY_FACTOR,
                       min_std_dev=MIN_STD_DEV,
                       extra_noise_scale=EXTRA_NOISE_SCALE)
    cem.set_initial_mean_params(reference_model)

    # --- Parallel Setup ---
    num_workers = min(20, POPULATION_SIZE)
    logger.log_message(f"Starting CEM training with {num_workers} persistent parallel workers.")
    initializer_with_args = partial(init_worker, config=config)
    pool = multiprocessing.Pool(processes=num_workers, initializer=initializer_with_args)

    curriculum = SelfAdaptingCurriculum(min_difficulty=0.1, max_difficulty=1.0, use_ema_variance=True)

    try:
        for gen in range(1, GENERATIONS + 1):
            # 1. Sample population
            population_params = cem.sample_population()

            diff = difficulty_for_gen(gen, step_every=100, step_size=0.05, start=0.02, end=1.0)
                 # curriculum = SelfAdaptingCurriculum(min_difficulty=0.1, max_difficulty=1.0)
            curriculum.current_difficulty = diff
            initial_conditions = curriculum.get_initial_conditions(10)
            
            slope = curriculum.slope

            # 2. Dispatch tasks
            tasks = [(i, params, initial_conditions,diff) for i, params in enumerate(population_params)]
            results = pool.map(run_worker, tasks)
            results.sort(key=lambda x: x[0])
            fitness_scores = [score for _, score in results]
            evaluated_population = list(zip(population_params, fitness_scores))

            

            # 3. Update distribution
            cem.update_distribution(evaluated_population)
            
            curriculum.update_difficulty(evaluated_population)

            # 4. Log generation
            logger.log_generation(generation=gen,
                                  evaluated_population=evaluated_population,
                                  extra_metrics={
                                      "Mean_StdDev_Params": float(getattr(cem, "mean_std_devs", float('nan'))),
                                      "Extra_Noise_Scale": getattr(cem, "epsilon", float('nan')),
                                      "Difficulty": diff,
                                      "Slope": slope
                                  })

    except KeyboardInterrupt:
        logger.log_message("Training interrupted by user.")

    finally:
        logger.log_message("Closing worker pool and saving final model...")
        pool.close()
        pool.join()

        final_weights = cem.get_best_params()
        final_state_dict = unflatten_nn_parameters(final_weights, reference_model)
        final_path = os.path.join(logger.models_save_dir, "cem_model_final_mean.pth")
        torch.save(final_state_dict, final_path)
        logger.log_message(f"Final CEM mean weights saved to {final_path}")
        logger.close()


if __name__ == "__main__":
    main()
