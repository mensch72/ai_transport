"""
PPO-oriented wrapper for the accessibility-equity project.

At the current stage, the existing `TransportGymWrapper` already matches the
PPO use case reasonably well:

- one external policy controls all vehicles jointly
- the action space is MultiDiscrete
- the observation is a gymnasium Dict space
- the reward is the project-local accessibility / equity reward

So this file currently provides a thin PPO-specific entry point around the
existing wrapper, while keeping a dedicated module path for future PPO-specific
changes such as:

- action masking
- richer PPO-friendly observations
- vectorized environment helpers
"""

from __future__ import annotations

from typing import Any

from accessibility_equity.wrappers.gym_wrapper import (
    TransportGymWrapper,
    create_transport_env,
)


class PPOTransportWrapper(TransportGymWrapper):
    """
    PPO-specific wrapper.

    Right now this class is intentionally thin and inherits the full behavior
    of the project-local `TransportGymWrapper`.
    """


def create_ppo_env(*args: Any, **kwargs: Any) -> PPOTransportWrapper:
    """
    Create a PPO-oriented transport environment.

    This currently reuses the same environment construction logic as the
    project-local gym wrapper.
    """
    env = create_transport_env(*args, **kwargs)
    env.__class__ = PPOTransportWrapper
    return env
