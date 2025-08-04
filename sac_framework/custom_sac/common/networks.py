import torch
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
    Stochastic Gaussian actor with Tanh squashing for SAC
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.net = nn.Sequential(nn.Linear(obs_dim, hidden_sizes[0]), nn.ReLU(),
                                 nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU())

        self.mean_layer = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[1], action_dim)

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        """
        Reparameterized sampling with tanh squashing and log_prob correction.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # sample from N(mean, std)
        y_t = torch.tanh(x_t)  # squash with tanh
        action = y_t

        # Log probability with Tanh correction
        log_prob = normal.log_prob(x_t).sum(-1)  # [B]
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(-1)  # Tanh correction
        log_prob = log_prob.unsqueeze(-1)  # [B,1]

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
        self.q_net = nn.Sequential(nn.Linear(obs_dim + action_dim, hidden_sizes[0]), nn.ReLU(),
                                   nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(),
                                   nn.Linear(hidden_sizes[1], 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q_net(x)
