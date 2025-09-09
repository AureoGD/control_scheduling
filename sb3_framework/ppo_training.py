import os
import glob
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from sb3_framework.curriculum_callback import CurriculumByPerformance

# Set this to True to test the discrete environment
IS_DISCRETE = False
# Number of parallel envirioments
N_ENVS = 4
# Steps collected per environment before an update
N_STEPS = 1024
# Frequency to save intermediate models (in steps)
CHECKPOINT_FREQ = 100_000
# Frequency to run evaluation
EVAL_FREQ = 10_000
# Total timesteps for training
TOTAL_TIMESTEPS = 100_000_000

# Option to continue training from the latest checkpoint
CONTINUE_TRAINING = False
# Root folder for saved models
MODEL_ROOT = "models/ppo"

if IS_DISCRETE:
    from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
    from scheduller_rules.schl_rule1 import SchedullerRule
    sw_rule = SchedullerRule()
else:
    from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv


def main(args):
    continue_training = CONTINUE_TRAINING
    if IS_DISCRETE:
        prefix = 'D_'
    else:
        prefix = 'C_'

    tensorboard_log = "logs/ppo"
    run_dir = None
    checkpoint_path = None

    # --- 0. Handle Continue Training ---
    if continue_training:
        # Search for the latest checkpoint inside the model folder
        checkpoints = glob.glob(os.path.join(MODEL_ROOT, prefix + "*/rl_model_*.zip"))
        if checkpoints:
            checkpoints.sort()
            checkpoint_path = checkpoints[-1]
            run_dir = os.path.dirname(checkpoint_path)
            run_name = os.path.basename(run_dir)
            print(f"[INFO] Continuing training from: {checkpoint_path}")
        else:
            print("[INFO] No checkpoints found. Starting a new training run.")
            continue_training = False

    if not continue_training:
        # Get the time string for a new run
        time_str = datetime.now().strftime("%d%H%M%S")

        # Combine them in the new order
        run_name = prefix + time_str
        run_dir = os.path.join(MODEL_ROOT, run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Run directory: {run_dir}")

    # --- 1. Define Environment Arguments ---
    env_kwargs = {}
    if IS_DISCRETE:
        env_kwargs['sw_rule'] = sw_rule

    # --- 2. Create Environments with Arguments ---

    train_env = make_vec_env(InvPendulumEnv, n_envs=N_ENVS, seed=0, env_kwargs=env_kwargs)
    eval_env = make_vec_env(InvPendulumEnv, n_envs=1, seed=1, env_kwargs=env_kwargs)

    # --- 3. Setup Callbacks ---
    # curriculum_cb = CurriculumCallback(total_timesteps=TOTAL_TIMESTEPS, step=0.025, ramp_portion=0.80)

    curriculum_cb = CurriculumByPerformance(
        total_timesteps=TOTAL_TIMESTEPS,
        init_scale=0.10,
        min_scale=0.10,
        max_scale=1.00,
        delta_up=0.10,
        delta_down=0.05,
        window_episodes=5_000,
        trunc_mastery_ratio=0.70,
        fail_backoff_ratio=0.70,
        horizon_hint=500,
        residency_frac=0.05,
        min_steps_between_changes=0,
        snap_to_step_grid=True,
        debug_print=False,
    )

    # Callback for saving the best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(EVAL_FREQ, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Save a model every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ, 1),
        save_path=run_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # --- 4. Create or Load the Model ---
    if continue_training and checkpoint_path:
        # Load the last checkpoint and attach the new environment
        print(f"Loading model from {checkpoint_path}...")
        model = PPO.load(checkpoint_path, env=train_env)
    else:
        print("Creating new model...")
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=1,
            n_steps=N_STEPS,
            batch_size=N_STEPS // N_ENVS,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.02,
            use_sde=not IS_DISCRETE,
            sde_sample_freq=4,
            normalize_advantage=True,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    print("\nStarting model training...")

    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[eval_callback, checkpoint_callback, curriculum_cb],
                tb_log_name=run_name,
                reset_num_timesteps=not continue_training)

    # --- 5. Final Save ---
    print("Training finished. Saving final model.")
    model.save(os.path.join(run_dir, "final_model"))

    # --- 6. Updated Training Summary ---
    print("\n--- Training Summary ---")
    print(f"Saved final model to: {run_dir}/final_model.zip")
    print("----------------------")


if __name__ == "__main__":
    main(None)  # Pass None or handle arguments properly if you use them
