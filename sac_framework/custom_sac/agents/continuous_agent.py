import torch
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

# --- Corrected Import Paths ---
from custom_sac.agents.base_agent import BaseAgent
from custom_sac.common.networks import ContinuousActor, ContinuousCritic


class ContinuousAgent(BaseAgent):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 device="cuda",
                 gamma=0.99,
                 tau=0.005,
                 alpha="auto",
                 actor_lr=3e-4,
                 critic_lr=3e-4):

        super().__init__(tau=tau)  # Call the parent constructor

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma

        # Networks
        self.actor = ContinuousActor(obs_dim, action_dim).to(device)
        self.q1 = ContinuousCritic(obs_dim, action_dim).to(device)
        self.q2 = ContinuousCritic(obs_dim, action_dim).to(device)

        # Target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        self._freeze_target_networks()

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)

        # Entropy tuning
        if isinstance(alpha, str) and alpha.lower() == "auto":
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=actor_lr)
            self._auto_alpha = True
            self.alpha = self.log_alpha.exp().detach()
        else:
            self._auto_alpha = False
            self.alpha = torch.tensor(float(alpha), device=self.device)

    def _freeze_target_networks(self):
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs)
        else:
            action, _ = self.actor.sample(obs)
        return action.cpu().numpy()[0]

    def update(self, batch):
        """
        A single update step for the actor and critic.
        Returns a dictionary of all relevant losses and metrics.
        """
        # --- Unpack batch and move to device ---
        obs = batch['obs'].to(self.device)
        act = batch['act'].to(self.device)
        rew = batch['rew'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device)

        # --- Update Critic ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            target_q_min = torch.min(self.q1_target(next_obs, next_action), self.q2_target(
                next_obs, next_action)) - self.alpha * next_log_prob
            target_q = rew + (1.0 - done) * self.gamma * target_q_min

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        loss_q1 = F.mse_loss(q1_pred, target_q)
        loss_q2 = F.mse_loss(q2_pred, target_q)
        critic_loss = loss_q1 + loss_q2

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # --- Update Actor and Alpha ---
        # Freeze Q-networks to save computation
        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        action, log_prob = self.actor.sample(obs)
        q_min = torch.min(self.q1(obs, action), self.q2(obs, action))
        actor_loss = (self.alpha * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze Q-networks
        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        # --- Update Alpha (Entropy Temperature) ---
        alpha_loss = torch.tensor(0.0).to(self.device)
        if self._auto_alpha:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()

        # --- Soft Update Target Networks ---
        self.soft_update_target()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "ent_coef_loss": alpha_loss.item(),
            "ent_coef": self.alpha.item(),
            "learning_rate": self.actor_optimizer.param_groups[0]['lr']
        }
