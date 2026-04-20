import abc
from typing import Any


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
