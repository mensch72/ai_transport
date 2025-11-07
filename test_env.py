"""Tests for AI Transport PettingZoo parallel environment."""

import numpy as np
import pytest
from ai_transport import parallel_env


def test_environment_creation():
    """Test that environment can be created with default parameters."""
    env = parallel_env()
    assert env is not None
    assert len(env.possible_agents) == 3  # Default num_buses
    assert env.num_stops == 10  # Default num_stops


def test_environment_reset():
    """Test environment reset functionality."""
    env = parallel_env(num_buses=2, num_stops=5)
    observations, infos = env.reset(seed=42)
    
    # Check that we get observations for all agents
    assert len(observations) == 2
    assert "bus_0" in observations
    assert "bus_1" in observations
    
    # Check observation shape
    for obs in observations.values():
        assert obs.shape == (5,)
        assert obs.dtype == np.float32


def test_environment_step():
    """Test environment step functionality."""
    env = parallel_env(num_buses=2, num_stops=5)
    observations, infos = env.reset(seed=42)
    
    # Take a step with random actions
    actions = {agent: 0 for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Check return values
    assert len(observations) == 2
    assert len(rewards) == 2
    assert len(terminations) == 2
    assert len(truncations) == 2
    assert len(infos) == 2
    
    # Check types
    for obs in observations.values():
        assert isinstance(obs, np.ndarray)
    for reward in rewards.values():
        assert isinstance(reward, (int, float, np.number))
    for term in terminations.values():
        assert isinstance(term, bool)
    for trunc in truncations.values():
        assert isinstance(trunc, bool)


def test_action_space():
    """Test action space is correct."""
    env = parallel_env(num_buses=2)
    
    for agent in env.possible_agents:
        action_space = env.action_space(agent)
        # Should have 3 discrete actions
        assert action_space.n == 3


def test_observation_space():
    """Test observation space is correct."""
    env = parallel_env(num_buses=2)
    
    for agent in env.possible_agents:
        obs_space = env.observation_space(agent)
        # Observation shape should be (5,)
        assert obs_space.shape == (5,)


def test_parallel_api_compliance():
    """Test that environment follows PettingZoo parallel API."""
    env = parallel_env(num_buses=2, num_stops=5)
    
    # Test required attributes
    assert hasattr(env, "possible_agents")
    assert hasattr(env, "agents")
    assert hasattr(env, "observation_spaces")
    assert hasattr(env, "action_spaces")
    
    # Test required methods
    assert callable(env.reset)
    assert callable(env.step)
    assert callable(env.observation_space)
    assert callable(env.action_space)
    
    # Test reset returns correct format
    observations, infos = env.reset()
    assert isinstance(observations, dict)
    assert isinstance(infos, dict)
    
    # Test step returns correct format
    actions = {agent: 0 for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    assert isinstance(observations, dict)
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)
    assert isinstance(infos, dict)


def test_episode_termination():
    """Test that episodes terminate correctly."""
    env = parallel_env(num_buses=2, num_stops=5)
    observations, infos = env.reset()
    
    # Run until max timesteps
    for _ in range(env.max_timesteps):
        actions = {agent: 0 for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Should be truncated now
    assert all(truncations.values())


def test_different_actions():
    """Test that different actions produce different results."""
    env = parallel_env(num_buses=1, num_stops=5)
    
    # Test action 0 (move to next stop)
    observations, infos = env.reset(seed=42)
    initial_pos = env.bus_positions["bus_0"]
    actions = {"bus_0": 0}
    env.step(actions)
    assert env.bus_positions["bus_0"] == (initial_pos + 1) % env.num_stops
    
    # Test action 1 (wait)
    observations, infos = env.reset(seed=42)
    initial_pos = env.bus_positions["bus_0"]
    actions = {"bus_0": 1}
    env.step(actions)
    assert env.bus_positions["bus_0"] == initial_pos
    
    # Test action 2 (skip stop)
    observations, infos = env.reset(seed=42)
    initial_pos = env.bus_positions["bus_0"]
    actions = {"bus_0": 2}
    env.step(actions)
    assert env.bus_positions["bus_0"] == (initial_pos + 2) % env.num_stops


def test_passenger_dynamics():
    """Test that passenger pickup and dropoff work correctly."""
    env = parallel_env(num_buses=1, num_stops=5)
    observations, infos = env.reset(seed=42)
    
    # Record initial state
    initial_passengers = env.bus_passengers["bus_0"]
    
    # Move and pick up passengers
    actions = {"bus_0": 0}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    # Bus should have picked up or dropped off passengers
    # (exact count depends on random state, but dynamics should work)
    assert env.bus_passengers["bus_0"] >= 0
    assert env.bus_passengers["bus_0"] <= env.max_passengers_per_vehicle


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
