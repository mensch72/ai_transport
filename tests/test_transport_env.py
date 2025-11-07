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


def test_human_aboard_state():
    """Test that human_aboard state is initialized correctly"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    
    # Check that all humans have aboard status
    for i in range(2):
        agent = f"human_{i}"
        assert agent in test_env.human_aboard
        assert test_env.human_aboard[agent] is None  # Initially not aboard


def test_step_type_state():
    """Test that step_type state is initialized correctly"""
    test_env = parallel_env()
    test_env.reset()
    
    assert test_env.step_type is not None
    assert test_env.step_type in ['routing', 'unboarding', 'boarding', 'departing']


def test_action_space_routing():
    """Test action spaces during routing step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'routing'
    
    # Vehicles at nodes should have actions for each node + None
    num_nodes = len(test_env.network.nodes())
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == num_nodes + 1
    
    # Humans should only have pass action
    human_space = test_env.action_space('human_0')
    assert human_space.n == 1


def test_action_space_unboarding():
    """Test action spaces during unboarding step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'unboarding'
    
    # Human not aboard should only have pass
    test_env.human_aboard['human_0'] = None
    space = test_env.action_space('human_0')
    assert space.n == 1
    
    # Human aboard vehicle at node should have pass and unboard
    test_env.human_aboard['human_1'] = 'vehicle_0'
    test_env.agent_positions['vehicle_0'] = 0  # at node
    space = test_env.action_space('human_1')
    assert space.n == 2
    
    # Vehicles should only have pass
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == 1


def test_action_space_boarding():
    """Test action spaces during boarding step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=2)
    test_env.reset()
    test_env.step_type = 'boarding'
    
    # Place all agents at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.agent_positions['vehicle_1'] = 0
    test_env.human_aboard['human_0'] = None
    test_env.human_aboard['human_1'] = None
    
    # Humans at node with 2 vehicles should have pass + 2 boarding options
    space = test_env.action_space('human_0')
    assert space.n == 3  # pass + 2 vehicles
    
    # Vehicles should only have pass
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == 1


def test_action_space_departing():
    """Test action spaces during departing step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None  # Not aboard
    test_env.human_aboard['human_1'] = 'vehicle_0'  # Aboard
    
    # Count outgoing edges from node 0
    outgoing_edges = list(test_env.network.out_edges(0))
    
    # Vehicle at node should have pass + outgoing edges
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == len(outgoing_edges) + 1
    
    # Human at node not aboard should have pass + outgoing edges
    human_space = test_env.action_space('human_0')
    assert human_space.n == len(outgoing_edges) + 1
    
    # Human aboard should only have pass
    human_aboard_space = test_env.action_space('human_1')
    assert human_aboard_space.n == 1


def test_action_space_agents_on_edges():
    """Test that agents on edges can only pass in all step types"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    
    # Place agents on an edge
    edges = list(test_env.network.edges())
    if edges:
        edge = edges[0]
        test_env.agent_positions['human_0'] = (edge, 5.0)
        test_env.agent_positions['vehicle_0'] = (edge, 3.0)
        test_env.human_aboard['human_0'] = None
        
        for step_type in ['routing', 'unboarding', 'boarding', 'departing']:
            test_env.step_type = step_type
            
            # Both should only have pass action when on edge
            human_space = test_env.action_space('human_0')
            assert human_space.n == 1
            
            vehicle_space = test_env.action_space('vehicle_0')
            assert vehicle_space.n == 1



def test_routing_step_logic():
    """Test routing step changes destinations but not time"""
    test_env = parallel_env(num_humans=1, num_vehicles=2)
    test_env.reset()
    test_env.step_type = 'routing'
    
    initial_time = test_env.real_time
    nodes = list(test_env.network.nodes())
    
    # Action 0 = set destination to None
    # Action 1, 2, 3 = set destination to node 0, 1, 2
    actions = {
        'human_0': 0,  # Should do nothing (human can only pass)
        'vehicle_0': 0,  # Set destination to None
        'vehicle_1': 2  # Set destination to node 1
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check destinations changed
    assert test_env.vehicle_destinations['vehicle_0'] is None
    assert test_env.vehicle_destinations['vehicle_1'] == nodes[1]


def test_unboarding_step_logic():
    """Test unboarding step changes aboard status but not time"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'unboarding'
    
    # Put humans aboard vehicle
    test_env.human_aboard['human_0'] = 'vehicle_0'
    test_env.human_aboard['human_1'] = 'vehicle_0'
    
    initial_time = test_env.real_time
    
    # Action 0 = pass, Action 1 = unboard
    actions = {
        'human_0': 1,  # Unboard
        'human_1': 0,  # Pass (stay aboard)
        'vehicle_0': 0  # Can only pass
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check human_0 unboarded, human_1 still aboard
    assert test_env.human_aboard['human_0'] is None
    assert test_env.human_aboard['human_1'] == 'vehicle_0'


def test_boarding_step_logic():
    """Test boarding step with capacity constraints"""
    test_env = parallel_env(num_humans=3, num_vehicles=1)
    test_env.reset(seed=42)
    test_env.step_type = 'boarding'
    
    # Set vehicle capacity to 2
    test_env.agent_attributes['vehicle_0']['capacity'] = 2
    
    # All at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['human_2'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None
    test_env.human_aboard['human_1'] = None
    test_env.human_aboard['human_2'] = None
    
    initial_time = test_env.real_time
    
    # All humans try to board (action 1 = board vehicle_0)
    actions = {
        'human_0': 1,
        'human_1': 1,
        'human_2': 1,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check that exactly 2 humans boarded (capacity = 2)
    humans_aboard = sum(1 for h in test_env.human_agents 
                       if test_env.human_aboard[h] == 'vehicle_0')
    assert humans_aboard == 2


def test_departing_step_logic_basic():
    """Test departing step moves agents onto edges"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Both at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None
    
    initial_time = test_env.real_time
    
    # Get first outgoing edge from node 0
    outgoing = list(test_env.network.out_edges(0))
    assert len(outgoing) > 0
    
    # Action 1 = depart into first outgoing edge
    actions = {
        'human_0': 1,
        'vehicle_0': 1
    }
    
    test_env.step(actions)
    
    # After departing, agents are placed on edge at coord 0, then time advances
    # and they move. One or both may reach the end.
    human_pos = test_env.agent_positions['human_0']
    vehicle_pos = test_env.agent_positions['vehicle_0']
    
    # At least one should have moved (time should have advanced)
    assert test_env.real_time > initial_time
    
    # Human might be on edge or at destination node
    # Vehicle might be on edge or at destination node
    # At least one should have departed (not both at node 0)
    assert not (human_pos == 0 and vehicle_pos == 0)


def test_departing_step_time_advance():
    """Test that departing step advances time correctly"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents on an edge
    edges = list(test_env.network.edges())
    edge = edges[0]
    edge_data = test_env.network[edge[0]][edge[1]]
    edge_length = edge_data['length']
    edge_speed = edge_data['speed']
    human_speed = test_env.agent_attributes['human_0']['speed']
    
    # Place human at coordinate 5, vehicle at coordinate 3
    test_env.agent_positions['human_0'] = (edge, 5.0)
    test_env.agent_positions['vehicle_0'] = (edge, 3.0)
    test_env.human_aboard['human_0'] = None
    
    initial_time = test_env.real_time
    
    # Both pass (don't depart, just move along edge)
    actions = {
        'human_0': 0,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Calculate expected time advance
    human_remaining = (edge_length - 5.0) / human_speed
    vehicle_remaining = (edge_length - 3.0) / edge_speed
    expected_delta_t = min(human_remaining, vehicle_remaining)
    
    # Check time advanced correctly
    assert abs(test_env.real_time - (initial_time + expected_delta_t)) < 1e-9


def test_departing_step_arrival_at_node():
    """Test that agents arrive at nodes when reaching edge end"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents very close to end of edge
    edges = list(test_env.network.edges())
    edge = edges[0]
    edge_data = test_env.network[edge[0]][edge[1]]
    edge_length = edge_data['length']
    
    # Place both agents same distance from end so they arrive together
    # Use a distance that works for both speeds
    edge_speed = edge_data['speed']
    human_speed = test_env.agent_attributes['human_0']['speed']
    
    # Place them close enough that both will arrive
    # For edge_speed=5.0, human_speed=1.0, if we place at edge_length-0.05,
    # vehicle time = 0.05/5.0 = 0.01, human time = 0.05/1.0 = 0.05
    # min is 0.01, so vehicle arrives but human doesn't
    # Let's place human separately closer to end
    test_env.agent_positions['human_0'] = (edge, edge_length - 0.01)  # Very close
    test_env.agent_positions['vehicle_0'] = (edge, edge_length - 0.01)  # Very close
    test_env.human_aboard['human_0'] = None
    
    actions = {
        'human_0': 0,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Both should now be at the target node (or at least vehicle should be)
    target_node = edge[1]
    # With coord=9.99, human needs 0.01/1.0=0.01, vehicle needs 0.01/5.0=0.002
    # min is 0.002, so vehicle arrives, human goes to 9.99+1.0*0.002=9.992
    # Let's just check vehicle arrives
    assert test_env.agent_positions['vehicle_0'] == target_node


def test_humans_aboard_move_with_vehicle():
    """Test that humans aboard vehicles move with the vehicle"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place human aboard vehicle at node
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = 'vehicle_0'
    
    outgoing = list(test_env.network.out_edges(0))
    
    # Human tries to walk (but is aboard so shouldn't work)
    # Vehicle departs
    actions = {
        'human_0': 1,  # Try to walk (should be ignored)
        'vehicle_0': 1  # Depart
    }
    
    test_env.step(actions)
    
    # Human should be at same position as vehicle (moved with it)
    assert test_env.agent_positions['human_0'] == test_env.agent_positions['vehicle_0']
    
    # Vehicle should have moved (not at node 0 anymore)
    assert test_env.agent_positions['vehicle_0'] != 0
    
    # Human still aboard
    assert test_env.human_aboard['human_0'] == 'vehicle_0'
