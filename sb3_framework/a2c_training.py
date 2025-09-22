import os
import glob
from datetime import datetime

from stable_baselines3 import A2C  # Changed from PPO to A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from sb3_framework.curriculum_callback import CurriculumByPerformance

# Set this to True to test the discrete environment
IS_DISCRETE = False
# Number of parallel environments
N_ENVS = 32
# Steps collected per environment before an update (n_steps for A2C)
N_STEPS = 5
# Frequency to save intermediate models (in steps)
CHECKPOINT_FREQ = 1_000_000
# Frequency to run evaluation
EVAL_FREQ = 10_000
# Total timesteps for training
TOTAL_TIMESTEPS = 100_000_000

# Option to continue training from the latest checkpoint
CONTINUE_TRAINING = False
# Root folder for saved models, updated for A2C
MODEL_ROOT = "models/a2c"

if IS_DISCRETE:
    from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
    from scheduller_rules.schl_rule1 import SchedullerRule
    sw_rule = SchedullerRule()
else:
    from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv


def main(args):
    """Main training function."""
    continue_training = CONTINUE_TRAINING
    if IS_DISCRETE:
        prefix = 'D_'
    else:
        prefix = 'C_'

    # Updated log directory for A2C
    tensorboard_log = "logs/a2c"
    run_dir = None
    checkpoint_path = None

    # --- 0. Handle Continue Training ---
    if continue_training:
        # Search for the latest checkpoint inside the model folder
        checkpoints = glob.glob(os.path.join(MODEL_ROOT, prefix + "*/rl_model_*.zip"))
        if checkpoints:
            # Sort checkpoints by modification time to get the latest
            checkpoints.sort(key=os.path.getmtime)
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
        # This will fail if the rule object isn't defined.
        # Ensure SchedullerRule is available.
        try:
            env_kwargs['sw_rule'] = sw_rule
        except NameError:
            print("[WARN] SchedullerRule not defined. Continuing without it.")

    # --- 2. Create Vectorized Environments ---
    train_env = make_vec_env(InvPendulumEnv, n_envs=N_ENVS, seed=0, env_kwargs=env_kwargs)
    eval_env = make_vec_env(InvPendulumEnv, n_envs=1, seed=1, env_kwargs=env_kwargs)

    # Important: expose episode stats/final_info (TimeLimit.truncated) to callbacks
    train_env = VecMonitor(train_env)
    eval_env = VecMonitor(eval_env)

    # --- 3. Setup Callbacks ---

    # curriculum_cb = CurriculumCallback(total_timesteps=TOTAL_TIMESTEPS, step=0.025, ramp_portion=0.60)
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

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path=run_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # --- 4. Create or Load the Model ---
    if continue_training and checkpoint_path:
        print(f"Loading model from {checkpoint_path}...")
        model = A2C.load(checkpoint_path, env=train_env, tensorboard_log=tensorboard_log)
    else:
        print("Creating new A2C model...")
        model = A2C(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=1,
            n_steps=N_STEPS,
            gamma=0.99,
            learning_rate=7e-4,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )

    print("\nStarting A2C model training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[eval_callback, checkpoint_callback, curriculum_cb],
                tb_log_name=run_name,
                reset_num_timesteps=not continue_training)

    # --- 5. Final Save ---
    print("Training finished. Saving final model.")
    model.save(os.path.join(run_dir, "final_model"))

    # --- 6. Updated Training Summary ---
    print("\n--- Training Summary ---")
    print(f"Algorithm: A2C")
    print(f"Environment: {'Discrete' if IS_DISCRETE else 'Continuous'}")
    print(f"Total Timesteps: {TOTAL_TIMESTEPS}")
    print(f"Saved final model to: {os.path.join(run_dir, 'final_model.zip')}")
    print("----------------------")


if __name__ == "__main__":
    main(None)
