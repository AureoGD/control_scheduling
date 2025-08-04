# custo_sac/common/logger.py
import os
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    A wrapper for TensorBoard's SummaryWriter that also handles saving
    model checkpoints periodically and tracking the best-performing model.
    """

    def __init__(self, base_log_dir="logs/", base_model_dir="models/", project_name="sac", is_discrete=False):
        """
        Initializes the logger and model saving functionality.

        Args:
            base_log_dir (str): The base directory for TensorBoard logs (e.g., 'logs/').
            base_model_dir (str): The base directory for saved models (e.g., 'models/').
            project_name (str): The name of the algorithm/project (e.g., 'sac').
            is_discrete (bool): Flag for naming the run directory.
        """
        # --- Create unique run name ---
        if is_discrete:
            prefix = 'D_'
        else:
            prefix = 'C_'

        # Get the time string
        time_str = datetime.now().strftime("%d%H%M%S")

        # Combine them in the new order
        run_name = prefix + time_str

        # --- Define and create directories based on the new structure ---
        # self.log_dir is for TensorBoard
        self.log_dir = os.path.join(base_log_dir, project_name, run_name)
        # self.models_dir is for model checkpoints
        self.models_dir = os.path.join(base_model_dir, project_name, run_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        # --- Initialize TensorBoard Writer ---
        self.writer = SummaryWriter(self.log_dir)
        print(f"TensorBoard logs will be saved to: {self.log_dir}")
        print(f"Models will be saved to: {self.models_dir}")

        # --- Initialize model saving tracking ---
        self.best_avg_reward = -np.inf

    def log_scalar(self, tag, value, step):
        """Logs a single scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)

    def _save_checkpoint(self, agent, file_name):
        """Internal method to save a model checkpoint."""
        checkpoint_path = os.path.join(self.models_dir, file_name)

        # It's best practice to save state_dicts for flexibility
        checkpoint = {
            'actor_state_dict': agent.actor.state_dict(),
            'q1_state_dict': agent.q1.state_dict(),
            'q2_state_dict': agent.q2.state_dict(),
            'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
            'q1_optimizer_state_dict': agent.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': agent.q2_optimizer.state_dict(),
        }
        # Also save alpha state if it's being tuned automatically
        if agent._auto_alpha:
            checkpoint['log_alpha'] = agent.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = agent.alpha_optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        # print(f"Saved model checkpoint to: {checkpoint_path}")

    def save_periodic_model(self, agent, step):
        """Saves the model at a specific timestep."""
        file_name = f"model_step_{step}.pth"
        self._save_checkpoint(agent, file_name)
        print(f"Saved periodic model at step {step}.")

    def save_best_model(self, agent, current_avg_reward):
        """
        Saves the model if the current average reward is the best seen so far.
        """
        if current_avg_reward > self.best_avg_reward:
            self.best_avg_reward = current_avg_reward
            print(f"New best average reward: {self.best_avg_reward:.2f}. Saving best model...")
            self._save_checkpoint(agent, "best_model.pth")

    def close(self):
        """Closes the SummaryWriter to ensure all logs are written."""
        self.writer.close()
