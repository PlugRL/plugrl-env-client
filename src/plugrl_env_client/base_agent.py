import abc
from typing import Dict


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    @abc.abstractmethod
    def feedback(
        self,
        obs: Dict,
        rewards: float,
        terminated: bool,
        truncated: bool,
        info: Dict,
    ) -> None:
        """Provide feedback to the agent."""

    def reset(self) -> None:
        """Reset the agent to its initial state."""
