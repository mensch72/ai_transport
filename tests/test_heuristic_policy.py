import pytest
import numpy as np
import networkx as nx
from ai_transport import parallel_env
from ai_transport.policies import HeuristicRoutingHumanPolicy


def create_simple_network():
    """Create a simple test network."""
    G = nx.DiGraph()
    # Create a linear network: 0 -> 1 -> 2 -> 3
    G.add_node(0, name="A", x=0.0, y=0.0)
    G.add_node(1, name="B", x=10.0, y=0.0)
    G.add_node(2, name="C", x=20.0, y=0.0)
    G.add_node(3, name="D", x=30.0, y=0.0)
    
    # Add edges with length and speed
    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=10.0, speed=5.0, capacity=10)
    G.add_edge(2, 3, length=10.0, speed=5.0, capacity=10)
    
    return G


def create_branching_network():
    """Create a branching network for more complex tests."""
    G = nx.DiGraph()
    # Network structure:
    #     1
    #    / \
    #   0   3
    #    \ /
    #     2
    G.add_node(0, name="Start", x=0.0, y=0.0)
    G.add_node(1, name="North", x=10.0, y=10.0)
    G.add_node(2, name="South", x=10.0, y=-10.0)
    G.add_node(3, name="End", x=20.0, y=0.0)
    
    G.add_edge(0, 1, length=14.14, speed=5.0, capacity=10)  # ~sqrt(10^2 + 10^2)
    G.add_edge(0, 2, length=14.14, speed=5.0, capacity=10)
    G.add_edge(1, 3, length=14.14, speed=5.0, capacity=10)
    G.add_edge(2, 3, length=14.14, speed=5.0, capacity=10)
    
    return G


def test_policy_initialization():
    """Test that the policy can be initialized correctly."""
    network = create_simple_network()
    target_nodes = {3}
    
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes=target_nodes,
        p_wait=0.5,
        seed=42
    )
    
    assert policy.agent_id == 'human_0'
    assert policy.target_nodes == {3}
    assert policy.p_wait == 0.5
    assert policy.network == network


def test_policy_with_single_target():
    """Test policy with a single target node."""
    network = create_simple_network()
    env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        network=network,
        observation_scenario='full'
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    # Create policy with node 3 as target
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.0,  # Never wait, always walk
        seed=42
    )
    
    # Test that policy can generate actions
    action_space_size = env.action_space('human_0').n
    action, justification = policy.get_action(obs['human_0'], action_space_size)
    
    assert isinstance(action, (int, np.integer))
    assert isinstance(justification, str)
    assert action >= 0
    assert action < action_space_size


def test_policy_with_multiple_targets():
    """Test policy with multiple target nodes."""
    network = create_branching_network()
    target_nodes = {2, 3}  # Multiple targets
    
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes=target_nodes,
        p_wait=0.3,
        seed=42
    )
    
    assert policy.target_nodes == {2, 3}


def test_boarding_decision():
    """Test that the policy can make boarding decisions."""
    network = create_simple_network()
    env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        network=network,
        observation_scenario='full'
    )
    
    obs, info = env.reset(seed=42)
    
    # Create policy
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Run simulation until boarding step
    for _ in range(100):  # Limit iterations
        if env.step_type == 'routing':
            # Set vehicle destination
            actions = {}
            for agent in env.agents:
                if agent == 'vehicle_0':
                    # Set destination to node 3 (which is in target set)
                    actions[agent] = 4  # Assuming node 3 is at index 3 (action = index + 1)
                else:
                    actions[agent] = 0
            obs, _, _, _, _ = env.step(actions)
        elif env.step_type == 'boarding':
            # Get boarding action from policy
            action_space_size = env.action_space('human_0').n
            action, justification = policy.get_action(obs['human_0'], action_space_size)
            
            # Action should be valid
            assert 0 <= action < action_space_size
            assert isinstance(justification, str)
            
            # If there's a vehicle available and it's going toward target, might board
            # (depends on the logic)
            break
        else:
            # Pass in other steps
            actions = {agent: 0 for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)


def test_walking_decision_with_p_wait():
    """Test that the policy respects p_wait probability."""
    network = create_simple_network()
    
    # Test with p_wait = 0 (always walk)
    policy_always_walk = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.0,
        seed=42
    )
    
    # Test with p_wait = 1 (always wait)
    policy_always_wait = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=1.0,
        seed=42
    )
    
    # Create mock observation for departing step
    observation = {
        'step_type': 'departing',
        'my_position': 0,  # At node 0
        'agent_attributes': {
            'human_0': {'speed': 1.0}
        },
        'action_mapping': {
            'description': {0: 'pass', 1: 'walk_edge'},
            'details': {0: None, 1: (0, 1)}
        }
    }
    
    # Test always walk (p_wait=0)
    action_walk, justification_walk = policy_always_walk.get_action(observation, 2)
    # Should walk (action > 0) since p_wait=0 and not at target
    
    # Test always wait (p_wait=1)
    action_wait, justification_wait = policy_always_wait.get_action(observation, 2)
    # Should wait (action == 0) since p_wait=1
    assert action_wait == 0
    assert 'Waiting' in justification_wait


def test_at_target_node():
    """Test that policy passes when already at target."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={0},  # Target is node 0
        p_wait=0.5,
        seed=42
    )
    
    # Mock observation at target node
    observation = {
        'step_type': 'departing',
        'my_position': 0,  # At node 0, which is the target
        'agent_attributes': {
            'human_0': {'speed': 1.0}
        },
        'action_mapping': {
            'description': {0: 'pass'},
            'details': {0: None}
        }
    }
    
    action, justification = policy.get_action(observation, 1)
    assert action == 0
    assert 'target' in justification.lower()


def test_routing_step_passes():
    """Test that policy passes during routing step."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    observation = {
        'step_type': 'routing',
        'my_position': 0,
        'agent_attributes': {},
        'action_mapping': {'description': {0: 'pass'}, 'details': {}}
    }
    
    action, justification = policy.get_action(observation, 1)
    assert action == 0
    assert 'routing' in justification.lower()


def test_unboarding_step_passes():
    """Test that policy passes during unboarding step."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    observation = {
        'step_type': 'unboarding',
        'my_position': 0,
        'agent_attributes': {},
        'action_mapping': {'description': {0: 'pass'}, 'details': {}}
    }
    
    action, justification = policy.get_action(observation, 1)
    assert action == 0


def test_shortest_path_computation():
    """Test that shortest path computation works correctly."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Compute path from 0 to 3
    path = policy._compute_shortest_duration_path(0, 3)
    assert path is not None
    assert path == [0, 1, 2, 3]
    
    # Compute path duration
    duration = policy._compute_path_duration(path)
    assert duration > 0
    # Duration should be 3 edges * (10.0 length / 5.0 speed) = 6.0
    assert abs(duration - 6.0) < 0.01


def test_nodes_on_path():
    """Test that we correctly identify nodes on a path."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    path = [0, 1, 2, 3]
    nodes = policy._get_nodes_on_path(path)
    assert nodes == {0, 1, 2, 3}
    
    # Test with None path
    nodes_none = policy._get_nodes_on_path(None)
    assert nodes_none == set()


def test_policy_reset():
    """Test that policy reset works."""
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Reset should not raise an error
    policy.reset()
