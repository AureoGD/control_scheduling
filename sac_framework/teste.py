import os
import gymnasium as gym
import multiprocessing as mp
from datetime import datetime
import torch
import numpy as np
import random

# ---------------------------
# SHARED HYPERPARAMETERS
# ---------------------------
TOTAL_TIMESTEPS = 200_000
BATCH_SIZE = 64
START_TRAIN_AFTER = 5_000
REPLAY_BUFFER_SIZE = 50_000
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 3
SEED = 42


# -----------------------------------
# TRAIN SB3 SAC
# -----------------------------------
def train_sb3_sac():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    log_dir = "logs/sb3_sac"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])

    model = SAC("MlpPolicy",
                env,
                verbose=1,
                seed=SEED,
                learning_rate=3e-4,
                batch_size=BATCH_SIZE,
                buffer_size=REPLAY_BUFFER_SIZE,
                learning_starts=START_TRAIN_AFTER,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
                ent_coef='auto',
                tensorboard_log=log_dir)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name="run")
    model.save(os.path.join(log_dir, "final_model"))


# -----------------------------------
# TRAIN CUSTOM SAC
# -----------------------------------
def train_custom_sac():
    import gymnasium as gym
    from sac_framework.custom_sac.agents.continuous_agent import ContinuousAgent
    from sac_framework.custom_sac.common.replay_buffer import ReplayBuffer
    from sac_framework.custom_sac.common.logger import Logger
    from sac_framework.custom_sac.common.learning import train_agent

    log_dir = "logs/custom_sac"
    os.makedirs(log_dir, exist_ok=True)

    # Seeding
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent = ContinuousAgent(
        obs_dim=obs_dim,
        action_dim=n_actions,
        alpha='auto',  # fixed entropy coefficient, change if needed
        device="cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(obs_dim=obs_dim, buffer_size=REPLAY_BUFFER_SIZE)
    logger = Logger(is_discrete=False)

    config = {
        "total_timesteps": TOTAL_TIMESTEPS,
        "learning_starts": START_TRAIN_AFTER,
        "batch_size": BATCH_SIZE,
        "eval_freq": EVAL_FREQ,
        "n_eval_episodes": N_EVAL_EPISODES
    }

    train_agent(env=env,
                eval_env=eval_env,
                agent=agent,
                buffer=buffer,
                logger=logger,
                config=config,
                buffer_zize=REPLAY_BUFFER_SIZE)


# -----------------------------------
# RUN BOTH IN PARALLEL
# -----------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")

    sb3_proc = mp.Process(target=train_sb3_sac)
    custom_proc = mp.Process(target=train_custom_sac)

    sb3_proc.start()
    custom_proc.start()

    sb3_proc.join()
    custom_proc.join()

    print("✅ Both trainings finished.")
