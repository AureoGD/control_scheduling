# dsac_framework/replay_buffer.py

import numpy as np
import torch


class ReplayBuffer:

    def __init__(self, obs_dim, buffer_size):
        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0

        self.state_memory = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.next_state_memory = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.action_memory = np.zeros(buffer_size, dtype=np.int64)  # <--- single integer
        self.reward_memory = np.zeros(buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(buffer_size, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        self.state_memory[self.ptr] = state
        self.next_state_memory[self.ptr] = next_state
        self.action_memory[self.ptr] = int(action)  # <--- store as scalar
        self.reward_memory[self.ptr] = reward
        self.terminal_memory[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        states = torch.tensor(self.state_memory[idxs], dtype=torch.float32)
        actions = torch.tensor(self.action_memory[idxs], dtype=torch.long).view(-1, 1)  # <---
        rewards = torch.tensor(self.reward_memory[idxs], dtype=torch.float32).view(-1, 1)
        next_states = torch.tensor(self.next_state_memory[idxs], dtype=torch.float32)
        dones = torch.tensor(self.terminal_memory[idxs], dtype=torch.float32).view(-1, 1)

        return {"obs": states, "act": actions, "rew": rewards, "next_obs": next_states, "done": dones}
