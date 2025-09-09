import os
import gymnasium
import torch
import numpy as np

from sac_framework.custom_sac.common.replay_buffer import ReplayBuffer
from sac_framework.custom_sac.common.learning import train_agent
from sac_framework.custom_sac.common.logger import Logger

# ---------------------------
# CONFIG
# ---------------------------

# Set this to True to test the discrete environment
IS_DISCRETE = False
# Steps collected per environment before an update
N_STEPS = 2048
# Frequency to save intermediate models (in steps)
CHECKPOINT_FREQ = 100_000
# Frequency to run evaluation
EVAL_FREQ = 10_000
# Total timesteps for training
TOTAL_TIMESTEPS = 15_000_000
# Batch size
BATCH_SIZE = 64
# Start training after this many steps to fill the buffer
START_TRAIN_AFTER = 10_000
# Replay buffer size
REPLAY_BUFFER_SIZE = 50_000
# Number of evaluation episodes
N_EVAL_EPISODES = 5
# Seed
SEED = 44

# ---------------------------
# SEEDING
# ---------------------------
import random

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------------------
# ENVIRONMENT SELECTION
# ---------------------------
if IS_DISCRETE:
    from sac_framework.custom_sac.agents.discrete_agent import DiscreteAgent
    from environment.cart_pendulum.env_pendulum_disc import InvPendulumEnv
    from scheduller_rules.schl_rule2 import SchedullerRule

    sw_rule = SchedullerRule()
    env_kwargs = {}
    env_kwargs['sw_rule'] = sw_rule

    env = InvPendulumEnv(**env_kwargs, rendering=False)
    eval_env = InvPendulumEnv(**env_kwargs, rendering=False)
    output_dim = env.action_space.n

    obs_dim = env.observation_space.shape[0]

    agent = DiscreteAgent(obs_dim=obs_dim, n_actions=output_dim, device="cuda" if torch.cuda.is_available() else "cpu")
else:
    from sac_framework.custom_sac.agents.continuous_agent import ContinuousAgent
    from environment.cart_pendulum.env_pendulum_cont import InvPendulumEnv

    # Corrected: pass the environment class instead of the agent
    env = InvPendulumEnv(rendering=False)
    eval_env = InvPendulumEnv(rendering=False)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    agent = ContinuousAgent(obs_dim=obs_dim,
                            action_dim=n_actions,
                            alpha='0.01',
                            device="cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# REPLAY BUFFER
# ---------------------------
# Fix: use the correct action dimension (n_actions)
buffer = ReplayBuffer(obs_dim=obs_dim, buffer_size=REPLAY_BUFFER_SIZE)

config = {
    "total_timesteps": TOTAL_TIMESTEPS,
    "learning_starts": START_TRAIN_AFTER,
    "batch_size": BATCH_SIZE,
    "eval_freq": EVAL_FREQ,
    "n_eval_episodes": N_EVAL_EPISODES
}

logger = Logger(is_discrete=IS_DISCRETE)

# ---------------------------
# TRAINING LOOP
# ---------------------------
train_agent(env=env,
            eval_env=eval_env,
            agent=agent,
            buffer=buffer,
            logger=logger,
            config=config,
            buffer_zize=REPLAY_BUFFER_SIZE)
