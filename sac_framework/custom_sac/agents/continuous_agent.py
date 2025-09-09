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
                 tau=0.0005,
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
            self.log_alpha = torch.zeros(1,
                                         requires_grad=True,
                                         device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha],
                                              lr=actor_lr,
                                              eps=1e-4)
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
        obs = torch.as_tensor(obs, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(obs)
        else:
            action, _ = self.actor.sample(obs)
        return action.cpu().numpy()[0]

    def update(self, batch):
        """
        A single update step for the actor, critic, and entropy coefficient (alpha).
        Returns a dictionary of all relevant losses and metrics.
        """
        # --- Unpack batch and move to device ---
        obs = batch['obs'].to(self.device)
        act = batch['act'].to(self.device)
        rew = batch['rew'].to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        done = batch['done'].to(self.device).float()

        # --- Check for NaNs ---
        if torch.isnan(obs).any() or torch.isnan(act).any() or torch.isnan(
                rew).any() or torch.isnan(next_obs).any():
            print("!!! NaN detected in the input batch from replay buffer !!!")
            import sys
            sys.exit()

        # ==================== 1. Update Alpha First ====================
        action_pi, log_prob = self.actor.sample(obs)
        log_prob = log_prob.view(-1, 1)  # ensure shape

        alpha_loss = torch.tensor(0.0).to(self.device)
        if self._auto_alpha:
            alpha_loss = -(self.log_alpha *
                           (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().detach()
        ent_coef = self.alpha.detach()  # used for critic and actor

        # ==================== 2. Update Critic ====================
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            next_log_prob = next_log_prob.view(-1, 1)
            target_q1 = self.q1_target(next_obs, next_action)
            target_q2 = self.q2_target(next_obs, next_action)
            target_q_min = torch.min(target_q1, target_q2)
            target_q = rew + (1.0 - done) * self.gamma * (
                target_q_min - ent_coef * next_log_prob)
            target_q = torch.clamp(target_q, -10.0, 10.0)

        # Compute critic losses
        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        loss_q1 = F.mse_loss(q1_pred, target_q)
        loss_q2 = F.mse_loss(q2_pred, target_q)
        critic_loss = 0.5 * (loss_q1 + loss_q2)

        # Optimize both Q-networks
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        # NEW: Clip critic gradients
        torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 1.0)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # ==================== 3. Update Actor ====================
        # Freeze critic parameters to avoid unnecessary gradient computation
        for p in self.q1.parameters():
            p.requires_grad = False
        for p in self.q2.parameters():
            p.requires_grad = False

        action_pi, log_prob = self.actor.sample(obs)
        log_prob = log_prob.view(-1, 1)
        q1_pi = self.q1(obs, action_pi)
        q2_pi = self.q2(obs, action_pi)
        q_min = torch.min(q1_pi, q2_pi)
        actor_loss = (ent_coef * log_prob - q_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Unfreeze critic parameters
        for p in self.q1.parameters():
            p.requires_grad = True
        for p in self.q2.parameters():
            p.requires_grad = True

        # ==================== 4. Soft Update Target Networks ====================
        self.soft_update_target()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "ent_coef_loss": alpha_loss.item(),
            "ent_coef": ent_coef.item(),
            "learning_rate": self.actor_optimizer.param_groups[0]['lr'],
            "entropy": -log_prob.mean().item()
        }
