import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
from sac_framework.custom_sac.agents.base_agent import BaseAgent
from sac_framework.custom_sac.common.networks import DiscreteActor, DiscreteCritic


class DiscreteAgent(BaseAgent):

    def __init__(self,
                 obs_dim,
                 n_actions,
                 device="gpu",
                 gamma=0.99,
                 tau=0.005,
                 alpha=0.2,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 tune_alpha=True):

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = device
        self._auto_alpha = tune_alpha

        # Hyperparameters
        self.gamma = gamma  # discount
        self.tau = tau  # target smoothing
        self.alpha = alpha  # entropy temperature (fixed for now)

        # Networks
        self.actor = DiscreteActor(obs_dim, n_actions).to(device)
        self.q1 = DiscreteCritic(obs_dim, n_actions).to(device)
        self.q2 = DiscreteCritic(obs_dim, n_actions).to(device)

        # Target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self._freeze_target_networks()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        if self._auto_alpha:
            # Target entropy is a heuristic, often set to a fraction of the max entropy.
            self.target_entropy = -0.98 * np.log(1.0 / self.n_actions)
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=actor_lr)
            self.alpha = self.log_alpha.exp().detach()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)

    def _freeze_target_networks(self):
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

    def select_action(self, obs, deterministic=True):
        obs = np.array(obs, dtype=np.float32)  # efficient conversion
        obs = torch.from_numpy(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, probs, _ = self.actor(obs)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action_dist = torch.distributions.Categorical(probs=probs)
            action = action_dist.sample()

        return action.item()

    def update_critic(self, obs, act, next_obs, rew, done):
        act = act.view(-1, 1)
        # Current Q-values for the pair (s, a) of the batch
        q1_pred = self.q1(obs).gather(1, act)
        q2_pred = self.q2(obs).gather(1, act)

        # Eq(16)
        with torch.no_grad():
            # from the batch, use the next_obs to find the next action distribuition
            _, probs, _ = self.actor(next_obs)
            greedy_action = torch.argmax(probs, dim=1, keepdim=True)  # a = argmax_a pi(a|s′)

            # using the target network, estimate the Q-values using the next_obs
            q1_next = self.q1_target(next_obs).gather(1, greedy_action)
            q2_next = self.q2_target(next_obs).gather(1, greedy_action)
            # evaluate the mean of the Q_values
            q_target = 0.5 * (q1_next + q2_next)

            target_q = rew + (1 - done) * self.gamma * q_target
            target_q = target_q.view(-1, 1)

        # DiscreteCritic losses
        loss_q1 = F.mse_loss(q1_pred, target_q)
        loss_q2 = F.mse_loss(q2_pred, target_q)

        # Gradient steps
        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        return loss_q1.item(), loss_q2.item()

    def update_actor(self, obs):
        # Forward pass through DiscreteActor to get probabilities and log-probs (entropy)
        _, probs, log_probs = self.actor(obs)

        if self._auto_alpha:
            alpha_loss, entropy = self.update_alpha(log_probs)
        else:
            alpha_loss, entropy = 0.0, -torch.sum(probs * log_probs, dim=1).mean().item()

        # Q-values from both critics
        q1 = self.q1(obs)
        q2 = self.q2(obs)
        q_avg = 0.5 * (q1 + q2)

        # DiscreteActor loss: Eq. (10)
        actor_loss = (probs * (self.alpha * log_probs - q_avg)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), alpha_loss, self.alpha, entropy

    def update_alpha(self, log_probs):
        # Compute policy entropy
        entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1)
        entropy = entropy.detach()

        # Alpha loss
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Update self.alpha from log_alpha
        self.alpha = self.log_alpha.exp().item()
        return alpha_loss.item(), entropy.mean().item()

    def update(self, batch):
        obs = torch.as_tensor(batch['obs'], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch['act'], dtype=torch.long, device=self.device).long()
        rew = torch.as_tensor(batch['rew'], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch['next_obs'], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch['done'], dtype=torch.float32, device=self.device)

        # Flatten action if necessary
        if act.ndim == 2:
            act = act.squeeze(-1)

        # --- Update Critics ---
        q1_loss, q2_loss = self.update_critic(obs, act, next_obs, rew, done)
        # --- Update Actor and Alpha ---
        actor_loss, alpha_loss, alpha, entropy = self.update_actor(obs)
        # --- Soft Update Target Networks ---
        self.soft_update_target()

        return {
            "critic_loss": (q1_loss + q2_loss) * 0.5,
            "actor_loss": actor_loss,
            "ent_coef_loss": alpha_loss,
            "ent_coef": alpha,
            "learning_rate": self.actor_optimizer.param_groups[0]['lr']
        }
