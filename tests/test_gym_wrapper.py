"""
Smoke tests for the RL wrapper.
"""

import numpy as np

from accessibility_equity.config import RewardConfig
from accessibility_equity.wrappers import create_transport_env


def test_wrapper_reset_and_step():
    """The wrapper should support one basic reset-step cycle."""
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=8, seed=1)

    obs, info = env.reset(seed=1)
    assert "step_type" in obs
    assert "real_time" in obs
    assert "vehicle_positions" in obs
    assert "human_positions" in obs
    assert "step_type" in info

    action = np.array([0], dtype=np.int64)
    obs, reward, terminated, truncated, info = env.step(action)

    assert "step_type" in obs
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step_count" in info

    env.close()


def test_default_reward_uses_normalized_time_integrated_utility():
    """Default reward should use normalized U(s_t) * scale * delta_t."""
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=8,
        seed=3,
        reward_config=RewardConfig(normalize_utility=True),
    )

    env.reset(seed=3)
    _obs, reward, _terminated, _truncated, info = env.step(
        np.array([0], dtype=np.int64)
    )

    assert "current_utility" in info
    assert "current_normalized_utility" in info
    assert "delta_t" in info
    assert "time_integrated_utility" in info
    assert "raw_time_integrated_utility" in info
    assert "utility_lower_bound" in info
    assert "utility_upper_bound" in info
    assert np.isclose(
        float(reward),
        (
            float(info["current_normalized_utility"])
            * 100.0
            * float(info["delta_t"])
        ),
    )
    assert np.isclose(float(reward), float(info["time_integrated_utility"]))
    assert np.isclose(
        float(info["raw_time_integrated_utility"]),
        float(info["current_utility"]) * float(info["delta_t"]),
    )

    env.close()


def test_reward_can_use_raw_time_integrated_current_state_utility():
    """Normalization can be disabled for raw U(s_t) * delta_t rewards."""
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=8,
        seed=3,
        reward_config=RewardConfig(normalize_utility=False),
    )

    env.reset(seed=3)
    _obs, reward, _terminated, _truncated, info = env.step(
        np.array([0], dtype=np.int64)
    )

    assert info["current_normalized_utility"] is None
    assert np.isclose(
        float(reward),
        float(info["current_utility"]) * float(info["delta_t"]),
    )

    env.close()
