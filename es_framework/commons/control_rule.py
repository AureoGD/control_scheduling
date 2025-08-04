import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ControlRule(nn.Module):

    def __init__(self, observation_dim: int, output_dim: int, discreate=False, **model_cfg):
        """
            Two linear layers NN                                 
        """
        super().__init__()

        self.discreate = discreate
        self.observation_dim = observation_dim
        self.output_dim = output_dim

        # Extract layer dimensions from the model_cfg dictionary.
        fc1_dim = model_cfg.get('fc1_dim', 64)
        fc2_dim = model_cfg.get('fc2_dim', 64)

        # The NN layers are defined based on the extracted dimensions
        self.fc1 = torch.nn.Linear(observation_dim, fc1_dim)
        self.fc2 = torch.nn.Linear(fc1_dim, fc2_dim)
        self.fc_out = torch.nn.Linear(fc2_dim, output_dim)

    def forward(self, normalized_state_tensor: torch.Tensor) -> torch.Tensor:
        """The forward pass expects a pre-normalized tensor."""
        x = torch.relu(self.fc1(normalized_state_tensor))
        x = torch.relu(self.fc2(x))
        logits = self.fc_out(x)

        if not self.discreate:
            mode_logits = torch.tanh(logits)
            return mode_logits
        else:
            probabilities = F.softmax(logits, dim=-1)
            return probabilities
