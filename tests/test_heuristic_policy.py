"""
Tests for HeuristicRoutingHumanPolicy.

NOTE: To see verbose test output with print statements, run:
    pytest tests/test_heuristic_policy.py -vs

The -s flag disables output capturing so you can see the detailed test output.
Without -s, pytest captures all print statements.
"""

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
    print("\n=== Testing Policy Initialization ===")
    print("Creating a simple linear network: 0 -> 1 -> 2 -> 3")
    network = create_simple_network()
    target_nodes = {3}
    
    print(f"Initializing HeuristicRoutingHumanPolicy with target_nodes={target_nodes}, p_wait=0.5")
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes=target_nodes,
        p_wait=0.5,
        seed=42
    )
    
    print(f"✓ Policy created successfully")
    print(f"  - agent_id: {policy.agent_id}")
    print(f"  - target_nodes: {policy.target_nodes}")
    print(f"  - p_wait: {policy.p_wait}")
    print(f"  - network nodes: {len(policy.nodes)}")
    
    assert policy.agent_id == 'human_0'
    assert policy.target_nodes == {3}
    assert policy.p_wait == 0.5
    assert policy.network == network
    print("✓ All assertions passed")


def test_policy_with_single_target():
    """Test policy with a single target node."""
    print("\n=== Testing Policy with Single Target ===")
    print("Creating environment with simple network")
    network = create_simple_network()
    env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        network=network,
        observation_scenario='full'
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Environment initialized with {len(env.agents)} agents")
    
    # Create policy with node 3 as target
    print("Creating policy with target node 3")
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.0,  # Never wait, always walk
        seed=42
    )
    
    # Test that policy can generate actions
    action_space_size = env.action_space('human_0').n
    print(f"Action space size for human_0: {action_space_size}")
    action, justification = policy.get_action(obs['human_0'], action_space_size)
    print(f"Generated action: {action}")
    print(f"Justification: {justification}")
    
    assert isinstance(action, (int, np.integer))
    assert isinstance(justification, str)
    assert action >= 0
    assert action < action_space_size
    print("✓ Policy generates valid actions")


def test_policy_with_multiple_targets():
    """Test policy with multiple target nodes."""
    print("\n=== Testing Policy with Multiple Targets ===")
    network = create_branching_network()
    target_nodes = {2, 3}  # Multiple targets
    print(f"Creating policy with multiple targets: {target_nodes}")
    
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes=target_nodes,
        p_wait=0.3,
        seed=42
    )
    
    print(f"✓ Policy created with target_nodes: {policy.target_nodes}")
    assert policy.target_nodes == {2, 3}
    print("✓ Multiple targets supported")


def test_boarding_decision():
    """Test that the policy can make boarding decisions."""
    print("\n=== Testing Boarding Decision Logic ===")
    network = create_simple_network()
    env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        network=network,
        observation_scenario='full'
    )
    
    obs, info = env.reset(seed=42)
    print("Environment reset, running simulation to boarding step...")
    
    # Create policy
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Run simulation until boarding step
    for iteration in range(100):  # Limit iterations
        if env.step_type == 'routing':
            # Set vehicle destination
            actions = {}
            for agent in env.agents:
                if agent == 'vehicle_0':
                    # Find the action index for setting destination to node 3
                    action_mapping = obs[agent].get('action_mapping', {})
                    details = action_mapping.get('details', {})
                    # Find action that sets destination to node 3
                    dest_action = 0
                    for action_idx, dest in details.items():
                        if dest == 3:
                            dest_action = action_idx
                            break
                    actions[agent] = dest_action
                    if dest_action > 0:
                        print(f"  Vehicle setting destination to node 3 (action {dest_action})")
                else:
                    actions[agent] = 0
            obs, _, _, _, _ = env.step(actions)
        elif env.step_type == 'boarding':
            # Get boarding action from policy
            print(f"Reached boarding step at iteration {iteration}")
            action_space_size = env.action_space('human_0').n
            action, justification = policy.get_action(obs['human_0'], action_space_size)
            
            print(f"  Boarding decision: action={action}")
            print(f"  Justification: {justification}")
            
            # Action should be valid
            assert 0 <= action < action_space_size
            assert isinstance(justification, str)
            print("✓ Boarding decision is valid")
            
            # If there's a vehicle available and it's going toward target, might board
            # (depends on the logic)
            break
        else:
            # Pass in other steps
            actions = {agent: 0 for agent in env.agents}
            obs, _, _, _, _ = env.step(actions)
    
    print("✓ Boarding logic tested successfully")


def test_walking_decision_with_p_wait():
    """Test that the policy respects p_wait probability."""
    print("\n=== Testing Walking Decision with p_wait ===")
    network = create_simple_network()
    
    # Test with p_wait = 0 (always walk)
    print("Testing p_wait=0 (should always walk)")
    policy_always_walk = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.0,
        seed=42
    )
    
    # Test with p_wait = 1 (always wait)
    print("Testing p_wait=1 (should always wait)")
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
    print(f"  p_wait=0: action={action_walk}, justification={justification_walk}")
    # Should walk (action > 0) since p_wait=0 and not at target
    
    # Test always wait (p_wait=1)
    action_wait, justification_wait = policy_always_wait.get_action(observation, 2)
    print(f"  p_wait=1: action={action_wait}, justification={justification_wait}")
    # Should wait (action == 0) since p_wait=1
    assert action_wait == 0
    assert 'Waiting' in justification_wait
    print("✓ p_wait probability respected")


def test_at_target_node():
    """Test that policy passes when already at target."""
    print("\n=== Testing Behavior at Target Node ===")
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={0},  # Target is node 0
        p_wait=0.5,
        seed=42
    )
    print("Created policy with target at node 0")
    
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
    
    print("Testing action when human is at target node 0")
    action, justification = policy.get_action(observation, 1)
    print(f"  Action: {action}")
    print(f"  Justification: {justification}")
    assert action == 0
    assert 'target' in justification.lower()
    print("✓ Policy correctly passes when at target")


def test_routing_step_passes():
    """Test that policy passes during routing step."""
    print("\n=== Testing Routing Step (Should Pass) ===")
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
    
    print("Testing action during routing step (humans can't act)")
    action, justification = policy.get_action(observation, 1)
    print(f"  Action: {action}")
    print(f"  Justification: {justification}")
    assert action == 0
    assert 'routing' in justification.lower()
    print("✓ Policy correctly passes during routing step")


def test_unboarding_step_passes():
    """Test that policy passes during unboarding step."""
    print("\n=== Testing Unboarding Step (Should Pass) ===")
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
    
    print("Testing action during unboarding step (not aboard vehicle)")
    action, justification = policy.get_action(observation, 1)
    print(f"  Action: {action}")
    print(f"  Justification: {justification}")
    assert action == 0
    print("✓ Policy correctly passes during unboarding step")


def test_shortest_path_computation():
    """Test that shortest path computation works correctly."""
    print("\n=== Testing Shortest Path Computation ===")
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Compute path from 0 to 3
    print("Computing shortest duration path from node 0 to node 3")
    path = policy._compute_shortest_duration_path(0, 3)
    print(f"  Path found: {path}")
    assert path is not None
    assert path == [0, 1, 2, 3]
    print("✓ Correct path computed")
    
    # Compute path duration
    print("Computing path duration")
    duration = policy._compute_path_duration(path)
    print(f"  Duration: {duration:.2f} time units")
    assert duration > 0
    # Duration should be 3 edges * (10.0 length / 5.0 speed) = 6.0
    expected_duration = 6.0
    assert abs(duration - expected_duration) < 0.01
    print(f"✓ Correct duration computed (expected {expected_duration}, got {duration:.2f})")


def test_nodes_on_path():
    """Test that we correctly identify nodes on a path."""
    print("\n=== Testing Nodes on Path Identification ===")
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    path = [0, 1, 2, 3]
    print(f"Testing with path: {path}")
    nodes = policy._get_nodes_on_path(path)
    print(f"  Nodes on path: {nodes}")
    assert nodes == {0, 1, 2, 3}
    print("✓ Correct nodes identified")
    
    # Test with None path
    print("Testing with None path")
    nodes_none = policy._get_nodes_on_path(None)
    print(f"  Nodes on None path: {nodes_none}")
    assert nodes_none == set()
    print("✓ None path handled correctly")


def test_policy_reset():
    """Test that policy reset works."""
    print("\n=== Testing Policy Reset ===")
    network = create_simple_network()
    policy = HeuristicRoutingHumanPolicy(
        agent_id='human_0',
        network=network,
        target_nodes={3},
        p_wait=0.5,
        seed=42
    )
    
    # Reset should not raise an error
    print("Calling policy.reset()")
    policy.reset()
    print("✓ Reset completed without errors")
    
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
                    # Find the action index for setting destination to node 3
                    action_mapping = obs[agent].get('action_mapping', {})
                    details = action_mapping.get('details', {})
                    # Find action that sets destination to node 3
                    dest_action = 0
                    for action_idx, dest in details.items():
                        if dest == 3:
                            dest_action = action_idx
                            break
                    actions[agent] = dest_action
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
