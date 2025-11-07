import pytest
import numpy as np
import networkx as nx
from ai_transport import env, parallel_env, raw_env


def test_import():
    """Test that the module can be imported"""
    assert env is not None
    assert parallel_env is not None
    assert raw_env is not None


def test_parallel_env_creation():
    """Test creating a parallel environment with default parameters"""
    test_env = parallel_env()
    assert test_env is not None
    assert len(test_env.possible_agents) == 3  # 2 humans + 1 vehicle by default
    assert test_env.num_humans == 2
    assert test_env.num_vehicles == 1


def test_parallel_env_custom_agents():
    """Test creating environment with custom number of agents"""
    test_env = parallel_env(num_humans=3, num_vehicles=2)
    assert len(test_env.possible_agents) == 5
    assert test_env.num_humans == 3
    assert test_env.num_vehicles == 2


def test_agent_attributes():
    """Test that agents have correct attributes"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    
    # Check human attributes
    for i in range(2):
        agent = f"human_{i}"
        assert agent in test_env.agent_attributes
        assert 'speed' in test_env.agent_attributes[agent]
        assert test_env.agent_attributes[agent]['speed'] == 1.0
    
    # Check vehicle attributes
    agent = "vehicle_0"
    assert agent in test_env.agent_attributes
    assert 'speed' in test_env.agent_attributes[agent]
    assert 'capacity' in test_env.agent_attributes[agent]
    assert 'fuel_use' in test_env.agent_attributes[agent]
    assert test_env.agent_attributes[agent]['speed'] == 2.0
    assert test_env.agent_attributes[agent]['capacity'] == 4
    assert test_env.agent_attributes[agent]['fuel_use'] == 1.0


def test_custom_agent_attributes():
    """Test creating environment with custom agent attributes"""
    test_env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        human_speed=2.5,
        vehicle_speed=5.0,
        vehicle_capacity=6,
        vehicle_fuel_use=2.0
    )
    
    assert test_env.agent_attributes["human_0"]['speed'] == 2.5
    assert test_env.agent_attributes["vehicle_0"]['speed'] == 5.0
    assert test_env.agent_attributes["vehicle_0"]['capacity'] == 6
    assert test_env.agent_attributes["vehicle_0"]['fuel_use'] == 2.0


def test_default_network():
    """Test that default network is created correctly"""
    test_env = parallel_env()
    assert test_env.network is not None
    assert isinstance(test_env.network, nx.DiGraph)
    assert len(test_env.network.nodes()) > 0
    assert len(test_env.network.edges()) > 0
    
    # Check node attributes
    for node in test_env.network.nodes():
        assert 'name' in test_env.network.nodes[node]
    
    # Check edge attributes
    for u, v in test_env.network.edges():
        edge_data = test_env.network[u][v]
        assert 'length' in edge_data
        assert 'speed' in edge_data
        assert 'capacity' in edge_data


def test_custom_network():
    """Test creating environment with custom network"""
    G = nx.DiGraph()
    G.add_node(0, name="Start")
    G.add_node(1, name="End")
    G.add_edge(0, 1, length=20.0, speed=10.0, capacity=5)
    
    test_env = parallel_env(network=G)
    assert test_env.network == G


def test_network_validation():
    """Test that network validation works"""
    # Network without node name attribute should raise error
    G = nx.DiGraph()
    G.add_node(0)
    with pytest.raises(ValueError, match="missing required 'name' attribute"):
        parallel_env(network=G)
    
    # Network without edge attributes should raise error
    G = nx.DiGraph()
    G.add_node(0, name="A")
    G.add_node(1, name="B")
    G.add_edge(0, 1)
    with pytest.raises(ValueError, match="missing required"):
        parallel_env(network=G)


def test_reset():
    """Test environment reset"""
    test_env = parallel_env()
    observations, infos = test_env.reset()
    
    # Check that all agents are present
    assert len(observations) == len(test_env.possible_agents)
    assert len(infos) == len(test_env.possible_agents)
    
    # Check state components initialized
    assert test_env.real_time == 0.0
    assert test_env.agent_positions is not None
    assert test_env.vehicle_destinations is not None
    
    # Check all agents have positions
    for agent in test_env.agents:
        assert agent in test_env.agent_positions
    
    # Check vehicles have destination (should be None initially)
    for agent in test_env.vehicle_agents:
        assert agent in test_env.vehicle_destinations
        assert test_env.vehicle_destinations[agent] is None


def test_position_types():
    """Test that positions can be either nodes or (edge, coordinate) tuples"""
    test_env = parallel_env()
    test_env.reset()
    
    # Initially all positions should be nodes
    for agent in test_env.agents:
        pos = test_env.agent_positions[agent]
        assert pos in test_env.network.nodes()
    
    # Test setting position as (edge, coordinate)
    edges = list(test_env.network.edges())
    if edges:
        edge = edges[0]
        edge_length = test_env.network[edge[0]][edge[1]]['length']
        test_env.agent_positions[test_env.agents[0]] = (edge, edge_length / 2)
        pos = test_env.agent_positions[test_env.agents[0]]
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert pos[0] == edge


def test_step_basic():
    """Test basic step functionality"""
    test_env = parallel_env()
    test_env.reset()
    
    # Create dummy actions
    actions = {agent: 0 for agent in test_env.agents}
    
    observations, rewards, terminations, truncations, infos = test_env.step(actions)
    
    # Check return values structure
    assert len(observations) == len(test_env.agents)
    assert len(rewards) == len(test_env.agents)
    assert len(terminations) == len(test_env.agents)
    assert len(truncations) == len(test_env.agents)
    assert len(infos) == len(test_env.agents)


def test_step_empty_actions():
    """Test step with empty actions"""
    test_env = parallel_env()
    test_env.reset()
    
    observations, rewards, terminations, truncations, infos = test_env.step({})
    
    # Should return empty dicts and clear agents
    assert observations == {}
    assert rewards == {}
    assert terminations == {}
    assert truncations == {}
    assert infos == {}
    assert len(test_env.agents) == 0


def test_wrapped_env():
    """Test that wrapped env function works"""
    test_env = env()
    assert test_env is not None


def test_raw_env():
    """Test that raw_env (AEC) function works"""
    test_env = raw_env()
    assert test_env is not None
