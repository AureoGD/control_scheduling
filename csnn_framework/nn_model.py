import torch
import torch.nn as nn
import torch.optim as optim


# The model definition from your previous question
class NNModel(nn.Module):

    def __init__(self, input_size=4, hidden_size1=256, hidden_size2=128):
        super().__init__()

        self.hidden_path = nn.Sequential(nn.Linear(input_size, hidden_size1), nn.Tanh(), nn.Linear(hidden_size1, hidden_size2), nn.Tanh(), nn.Linear(hidden_size2, 1))
        self.skip_connection = nn.Linear(input_size, 1, bias=True)

    def forward(self, x):
        output = self.hidden_path(x) + self.skip_connection(x)
        # output = self.hidden_path(x)
        return output
