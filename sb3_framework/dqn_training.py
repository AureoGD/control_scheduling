import os
import glob
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# Import the discrete environment directly
from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
from sb3_framework.curriculum_callback import CurriculumByPerformance
from scheduller_rules.schl_rule1 import SchedullerRule

# --- DQN Hyperparameters ---
# Number of parallel environments
N_ENVS =  16
# Size of the replay buffer
BUFFER_SIZE = 1_000_000
# Number of steps to collect before starting training
LEARNING_STARTS = 50_000
# Batch size for each training update
BATCH_SIZE = 128
# The soft update coefficient for the target network
TAU = 0.005
# Update the model every TRAIN_FREQ steps or episodes
TRAIN_FREQ = 2
# How many gradient steps to do after each update
GRADIENT_STEPS = 1
# Exploration fraction
EXPLORATION_FRACTION = 0.1
# Final value of epsilon
EXPLORATION_FINAL_EPS = 0.05

# --- Training Control ---
# Frequency to save intermediate models (in steps)
CHECKPOINT_FREQ = 1_000_000
# Frequency to run evaluation
EVAL_FREQ = 10_000
# Total timesteps for training
TOTAL_TIMESTEPS = 100_000_000

# Option to continue training from the latest checkpoint
CONTINUE_TRAINING = False
# Root folder for saved models
MODEL_ROOT = "models/dqn"


def main(args):
    continue_training = CONTINUE_TRAINING
    # Set prefix for discrete environment models
    prefix = 'D_'

    tensorboard_log = "logs/dqn"
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
    sw_rule = SchedullerRule()
    env_kwargs = {'sw_rule': sw_rule}

    # --- 2. Create Environments with Arguments ---
    train_env = make_vec_env(InvPendulumEnv, n_envs=N_ENVS, seed=0, env_kwargs=env_kwargs)
    eval_env = make_vec_env(InvPendulumEnv, n_envs=1, seed=1, env_kwargs=env_kwargs)

    # --- 3. Setup Callbacks ---
    # Callback for saving the best model
    curriculum_cb = CurriculumByPerformance(
        total_timesteps=TOTAL_TIMESTEPS,
        init_scale=0.10,
        min_scale=0.10,
        max_scale=1.00,
        delta_up=0.10,
        delta_down=0.05,
        window_episodes=5_000,
        dqn_eps_bump=0.2,
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
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),  # Eval freq is based on steps per env
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Save a model every CHECKPOINT_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq=max(CHECKPOINT_FREQ // N_ENVS, 1),  # Save freq is based on steps per env
        save_path=run_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    # --- 4. Create or Load the Model ---
    if continue_training and checkpoint_path:
        # Load the last checkpoint and attach the new environment
        print(f"Loading model from {checkpoint_path}...")
        model = DQN.load(checkpoint_path, env=train_env)
        # Load the replay buffer if it exists
        replay_buffer_path = os.path.join(run_dir, "rl_model_replay_buffer.pkl")
        if os.path.exists(replay_buffer_path):
            print(f"Loading replay buffer from {replay_buffer_path}...")
            model.load_replay_buffer(replay_buffer_path)
    else:
        print("Creating new model...")
        model = DQN(
            policy="MlpPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=1,
            buffer_size=BUFFER_SIZE,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=0.99,
            learning_rate=1e-4,
            train_freq=(TRAIN_FREQ, "step"),
            gradient_steps=GRADIENT_STEPS,
            exploration_fraction=EXPLORATION_FRACTION,
            exploration_final_eps=EXPLORATION_FINAL_EPS,
            target_update_interval=1000,
        )

    print("\nStarting model training...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS,
                callback=[eval_callback, checkpoint_callback, curriculum_cb],
                tb_log_name=run_name,
                reset_num_timesteps=not continue_training)

    # --- 5. Final Save ---
    print("Training finished. Saving final model.")
    model.save(os.path.join(run_dir, "final_model"))
    model.save_replay_buffer(os.path.join(run_dir, "final_replay_buffer"))

    # --- 6. Updated Training Summary ---
    print("\n--- Training Summary ---")
    print(f"Saved final model to: {os.path.join(run_dir, 'final_model.zip')}")
    print("----------------------")


if __name__ == "__main__":
    main(None)
