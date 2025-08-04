# custo_sac/agents/base_agent.py
from abc import ABC, abstractmethod


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
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
