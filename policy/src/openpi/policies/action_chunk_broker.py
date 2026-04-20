from typing import Any

import numpy as np
import tree
from typing_extensions import override

from openpi.policies import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wrap a policy to return action chunks one step at a time."""

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step = 0
        self._last_results: dict[str, Any] | None = None

    @override
    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
