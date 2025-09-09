# custo_sac/agents/base_agent.py
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseAgent(ABC):
    """
    An abstract base class for all agents. It defines the essential methods
    that any agent must implement to be compatible with the training loop.
    """

    def __init__(self, tau=0.005):

        self.tau = tau

    @abstractmethod
    def select_action(self, obs, deterministic=False):
        pass

    @abstractmethod
    def update(self, batch):
        pass

    def soft_update_target(self):
        """
        Performs a soft update (Polyak averaging) of the target networks.
        This implementation is shared across all SAC agents.
        """
        for param, target_param in zip(self.q1.parameters(),
                                       self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(),
                                       self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data +
                                    (1 - self.tau) * target_param.data)

    # def orthogonal_init(self, gain=1.0):

    #     def init_fn(m):
    #         if isinstance(m, nn.Linear):
    #             if hasattr(m, "gain"):
    #                 gain = m.gain
    #             elif m.out_features == 1:
    #                 gain = 1.0  # critic output
    #             elif m.out_features == self.action_dim:
    #                 gain = 0.01  # actor output
    #             else:
    #                 gain = np.sqrt(2)
    #             nn.init.orthogonal_(m.weight, gain=gain)
    #             nn.init.constant_(m.bias, 0)

    #     return init_fn
