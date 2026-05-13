"""
Smoke tests for the DQN-compatible wrapper.
"""

import numpy as np
from gymnasium import spaces

from accessibility_equity.wrappers import create_dqn_env


def test_dqn_wrapper_reset_and_step():
    """The DQN wrapper should expose flat observations and Discrete actions."""
    env = create_dqn_env(num_humans=2, num_nodes=8, seed=1)

    assert isinstance(env.action_space, spaces.Discrete)

    obs, info = env.reset(seed=1)
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == env.observation_space.shape
    assert "step_type" in info

    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "step_count" in info

    env.close()


def test_dqn_wrapper_routing_mode_skips_non_decision_steps():
    """Routing mode should only return vehicle routing decisions to DQN."""
    env = create_dqn_env(
        num_humans=2,
        num_nodes=8,
        seed=2,
        use_action_masking=True,
        decision_mode="routing",
    )

    obs, info = env.reset(seed=2)
    assert isinstance(obs, dict)
    assert isinstance(obs["features"], np.ndarray)
    assert obs["features"].shape == env.observation_space["features"].shape
    assert obs["action_mask"].shape == env.observation_space["action_mask"].shape
    assert "action_mask" in info

    routing_mask = env.action_masks()
    assert routing_mask.dtype == np.bool_
    assert routing_mask.shape == (env.action_space.n,)
    assert routing_mask[0]
    assert routing_mask[1:].all()

    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, dict)
    assert isinstance(float(reward), float)
    assert info["dqn_underlying_vehicle_action"] == 0
    assert info["dqn_decision_mode"] == "routing"
    assert info["dqn_underlying_steps"] > 1
    assert reward == info["raw_reward"]
    assert "delta_t" in info
    assert "time_integrated_utility" in info

    assert env.base_env.env.step_type == "routing"
    next_routing_mask = env.action_masks()
    assert next_routing_mask[0]
    assert next_routing_mask[1:].all()

    env.close()


def test_dqn_wrapper_routing_departing_mode_exposes_departing_actions():
    """Routing+departing mode should expose outgoing-edge choices at departing."""
    env = create_dqn_env(
        num_humans=2,
        num_nodes=8,
        seed=2,
        use_action_masking=True,
        decision_mode="routing_departing",
    )

    obs, _info = env.reset(seed=2)
    assert isinstance(obs, dict)
    assert env.base_env.env.step_type == "routing"

    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(float(reward), float)
    assert info["dqn_decision_mode"] == "routing_departing"
    assert env.base_env.env.step_type == "departing"
    departing_mask = env.action_masks()
    current_node = env._position_to_node(
        env.base_env.env.agent_positions.get(env.vehicle_agent)
    )
    neighbor_nodes = {
        target for _source, target in env.base_env.env.network.out_edges(current_node)
    }
    expected_node_actions = {
        env.node_to_index[node] + 1
        for node in neighbor_nodes
    }
    actual_node_actions = set(np.flatnonzero(departing_mask)) - {0}
    assert actual_node_actions == expected_node_actions

    if actual_node_actions:
        chosen_action = min(actual_node_actions)
        obs, reward, terminated, truncated, info = env.step(chosen_action)
        assert isinstance(obs, dict)
        assert isinstance(float(reward), float)
        assert info["dqn_action_was_valid"]
        assert env._at_dqn_decision_point() or terminated or truncated

    env.close()
