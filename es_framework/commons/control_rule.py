import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlRule(nn.Module):

    def __init__(self, observation_dim: int, output_dim: int, **model_cfg):
        super().__init__()

        self.discrete = bool(model_cfg.pop("discrete", False))
        fc1_dim = int(model_cfg.pop("fc1_dim", 64))
        fc2_dim = int(model_cfg.pop("fc2_dim", 64))
        action_low = model_cfg.pop("action_low", None)
        action_high = model_cfg.pop("action_high", None)

        # layers
        self.fc1 = nn.Linear(observation_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc_out = nn.Linear(fc2_dim, output_dim)

        # optional continuous-action scaling
        self._low = None
        self._high = None
        if action_low is not None and action_high is not None:
            low = np.asarray(action_low, dtype=np.float32).ravel()
            high = np.asarray(action_high, dtype=np.float32).ravel()
            if low.shape != high.shape or low.size != output_dim:
                raise ValueError("action_low/high must match output_dim")
            self._low = torch.tensor(low, dtype=torch.float32)
            self._high = torch.tensor(high, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RAW outputs: logits (discrete) or unsquashed (continuous)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc_out(h)

    @torch.no_grad()
    def predict(self, state, device="cpu", deterministic=False):
        x = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else state.float()
        single = False
        if x.ndim == 1:
            x = x.unsqueeze(0)
            single = True
        x = x.to(device)

        out = self.forward(x)
        if self.discrete:
            a = F.softmax(out, dim=-1).argmax(dim=-1).cpu().numpy()
            return int(a[0]) if single else a.astype(np.int64), state

        a = torch.tanh(out)
        if self._low is not None and self._high is not None:
            low = self._low.to(a.device)
            high = self._high.to(a.device)
            a = low + (a + 1.0) * 0.5 * (high - low)
        a = a.cpu().numpy().astype(np.float32)
        return a[0] if single else a, state

    def reset_parameters(self, ortho: bool = True, last_layer_std: float = 0.01, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)

        if ortho:
            gain = nn.init.calculate_gain("relu")
            nn.init.orthogonal_(self.fc1.weight, gain)
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.orthogonal_(self.fc2.weight, gain)
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.orthogonal_(self.fc_out.weight, last_layer_std)
            nn.init.constant_(self.fc_out.bias, 0.0)
        else:
            nn.init.kaiming_uniform_(self.fc1.weight, a=0.0, nonlinearity="relu")
            nn.init.constant_(self.fc1.bias, 0.0)
            nn.init.kaiming_uniform_(self.fc2.weight, a=0.0, nonlinearity="relu")
            nn.init.constant_(self.fc2.bias, 0.0)
            nn.init.xavier_uniform_(self.fc_out.weight)
            nn.init.constant_(self.fc_out.bias, 0.0)
        return self


# ==============================================================================
#  TEST SCRIPT
# ==============================================================================
if __name__ == "__main__":
    OBS_DIM = 4
    NUM_TEST_STATES = 1000  # Increased to 1000 as you tested
    ACTION_DIM_DISCRETE = 3

    print(f"--- Testing with larger initial weights for the final layer ---")

    model_discrete = ControlRule(observation_dim=OBS_DIM, output_dim=ACTION_DIM_DISCRETE, discrete=True)

    # The KEY CHANGE is here: last_layer_std=1.0
    model_discrete.reset_parameters(seed=45, last_layer_std=1)

    random_states_batch = np.random.randn(NUM_TEST_STATES, OBS_DIM).astype(np.float32)
    actions_discrete, _ = model_discrete.predict(random_states_batch)

    # Check the distribution of actions
    actions_taken, counts = np.unique(actions_discrete, return_counts=True)

    print(f"Tested with {NUM_TEST_STATES} random states.")
    print("Action distribution:")
    for action, count in zip(actions_taken, counts):
        print(f"  Action {action}: was chosen {count} times")

    print("\nNote: By increasing the initial weights of the final layer, the logits are more varied,")
    print("leading to a more random initial policy that doesn't always pick 0.")
    # # --- Test Case 2: Continuous Actions (Unscaled) ---
    # print("--- Test Case 2: Continuous Actions (Unscaled) ---")
    # ACTION_DIM_CONTINUOUS = 2  # Example: Pendulum has 1, but we'll use 2 for variety.

    # # 1. Instantiate the model for continuous actions (default behavior)
    # model_continuous_unscaled = ControlRule(observation_dim=OBS_DIM, output_dim=ACTION_DIM_CONTINUOUS)
    # model_continuous_unscaled.reset_parameters(seed=42)

    # # 2. Use the same random state
    # # 3. Get the action
    # action_continuous_unscaled, _ = model_continuous_unscaled.predict(random_state_single)

    # print(f"Input State (shape {random_state_single.shape}):\n{random_state_single}")
    # print(f"Output Action (shape {action_continuous_unscaled.shape}):\n{action_continuous_unscaled}")
    # print("Note: The output is a numpy array with values between -1 and 1 due to the 'tanh' activation.\n")

    # # --- Test Case 3: Continuous Actions (Scaled) ---
    # print("--- Test Case 3: Continuous Actions (Scaled) ---")

    # # 1. Define action space bounds and instantiate the model
    # ACTION_LOW = np.array([-5.0, -0.5], dtype=np.float32)
    # ACTION_HIGH = np.array([5.0, 1.0], dtype=np.float32)

    # model_continuous_scaled = ControlRule(observation_dim=OBS_DIM,
    #                                       output_dim=ACTION_DIM_CONTINUOUS,
    #                                       action_low=ACTION_LOW,
    #                                       action_high=ACTION_HIGH)
    # model_continuous_scaled.reset_parameters(seed=42)

    # # 2. Use the same random state
    # # 3. Get the action
    # action_continuous_scaled, _ = model_continuous_scaled.predict(random_state_single)

    # print(f"Input State (shape {random_state_single.shape}):\n{random_state_single}")
    # print(f"Action bounds: Low={ACTION_LOW}, High={ACTION_HIGH}")
    # print(f"Output Action (shape {action_continuous_scaled.shape}):\n{action_continuous_scaled}")
    # print("Note: The output is scaled to fit within the provided 'action_low' and 'action_high' bounds.\n")

    # # --- Test Case 4: Batch Processing ---
    # print("--- Test Case 4: Batch Processing ---")
    # BATCH_SIZE = 3

    # # 1. Create a batch of random input states
    # random_state_batch = np.random.randn(BATCH_SIZE, OBS_DIM).astype(np.float32)

    # # 2. Get a batch of actions (using the scaled model as an example)
    # actions_batch, _ = model_continuous_scaled.predict(random_state_batch)

    # print(f"Input Batch Shape: {random_state_batch.shape}")
    # print(f"Output Actions Shape: {actions_batch.shape}")
    # print(f"Output Actions:\n{actions_batch}")
    # print(
    #     "\nNote: The model correctly processes a batch of inputs (e.g., shape (3, 4)) and returns a corresponding batch of outputs (e.g., shape (3, 2))."
    # )
