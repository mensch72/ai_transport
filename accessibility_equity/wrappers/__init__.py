"""
Wrappers for the accessibility-equity project.

This module mirrors the structure of `ai_transport.wrappers` while exposing a
project-local wrapper that can evolve independently.
"""

from accessibility_equity.wrappers.gym_wrapper import (
    TransportGymWrapper,
    create_transport_env,
)
from accessibility_equity.wrappers.ppo_wrapper import (
    PPOTransportWrapper,
    create_ppo_env,
)
from accessibility_equity.wrappers.dqn_wrapper import (
    DQNTransportWrapper,
    REWARD_SCALE,
    create_dqn_env,
)

__all__ = [
    "TransportGymWrapper",
    "create_transport_env",
    "PPOTransportWrapper",
    "create_ppo_env",
    "DQNTransportWrapper",
    "REWARD_SCALE",
    "create_dqn_env",
]
