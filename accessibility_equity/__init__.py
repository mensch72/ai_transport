"""AI Transport Accessibility Equity.

This package is a self-contained reinforcement-learning project for transport
accessibility and equity experiments. It includes the transport simulator,
policies, scenario generation, centralized single-agent training wrappers, and
accessibility- and equity-oriented task logic.
"""

__version__ = "0.1.0"

from accessibility_equity.envs import env, parallel_env, raw_env
from accessibility_equity import policies

__all__ = [
    "env",
    "parallel_env",
    "raw_env",
    "policies",
]
