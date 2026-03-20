import abc
from typing import Any, Dict


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict, env_indices: Any) -> Dict:
        """Infer actions from observations."""

    @abc.abstractmethod
    def feedback(
        self,
        obs: Dict,
        rewards: Any,
        terminated: Any,
        truncated: Any,
        info: Dict,
        env_indices: Any,
    ) -> None:
        """Provide feedback to the agent."""

    def reset(self) -> None:
        """Reset the agent to its initial state."""
