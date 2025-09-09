import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DiscreteActor(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], n_actions)  # Output: action logits
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return logits, probs, log_probs


class DiscreteCritic(nn.Module):

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], n_actions)  # Output: Q(s, a) for all a
        )

    def forward(self, obs):
        return self.net(obs)  # shape: [batch_size, n_actions]


class ContinuousActor(nn.Module):
    """
    Stochastic Gaussian actor with Tanh squashing for SAC (SB3-style initialization).
    """

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_sizes=(256, 256),
                 log_std_min=-20,
                 log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Build hidden layers
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_sizes[0]),
                                 nn.ReLU(),
                                 nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                                 nn.ReLU())

        # Output layers
        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)

        # Apply SB3-style orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.constant_(self.mean_layer.bias, 0.0)

        nn.init.orthogonal_(self.log_std_layer.weight, gain=0.01)
        nn.init.constant_(self.log_std_layer.bias, 0.0)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # Tanh correction for log-prob
        log_prob = normal.log_prob(x_t).sum(-1)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(-1)
        log_prob = log_prob.unsqueeze(-1)

        return action, log_prob

    def deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class ContinuousCritic(nn.Module):
    """
    Q-network for SAC: Takes state and action as input.
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_sizes[0]), nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1))

        self._init_weights()

    def _init_weights(self):
        for i, layer in enumerate(self.q_net):
            if isinstance(layer, nn.Linear):
                is_output = (i == len(self.q_net) - 1)
                gain = 1.0 if is_output else np.sqrt(2)
                nn.init.orthogonal_(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q_net(x)
