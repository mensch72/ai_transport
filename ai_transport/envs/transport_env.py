import functools
from typing import Optional, Union, Tuple, Dict, Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, Tuple as TupleSpace
from gymnasium.utils import seeding
import networkx as nx
from scipy.spatial import Delaunay

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None, num_humans=2, num_vehicles=1, network=None,
        human_speed=1.0, vehicle_speed=2.0, vehicle_capacity=4, vehicle_fuel_use=1.0,
        observation_scenario='full'):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(
        render_mode=internal_render_mode,
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        network=network,
        human_speed=human_speed,
        vehicle_speed=vehicle_speed,
        vehicle_capacity=vehicle_capacity,
        vehicle_fuel_use=vehicle_fuel_use,
        observation_scenario=observation_scenario
    )
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None, num_humans=2, num_vehicles=1, network=None,
            human_speed=1.0, vehicle_speed=2.0, vehicle_capacity=4, vehicle_fuel_use=1.0,
            observation_scenario='full'):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(
        render_mode=render_mode,
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        network=network,
        human_speed=human_speed,
        vehicle_speed=vehicle_speed,
        vehicle_capacity=vehicle_capacity,
        vehicle_fuel_use=vehicle_fuel_use,
        observation_scenario=observation_scenario
    )
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "transport_v0"}
    
    # Float comparison epsilon for coordinate/length comparisons
    FLOAT_EPSILON = 1e-9

    def __init__(
        self,
        render_mode=None,
        num_humans=2,
        num_vehicles=1,
        network=None,
        human_speed=1.0,
        vehicle_speed=2.0,
        vehicle_capacity=4,
        vehicle_fuel_use=1.0,
        observation_scenario='full'
    ):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        
        Args:
            observation_scenario: One of 'full', 'local', or 'statistical'
                - 'full': Every agent observes the full state
                - 'local': Agents observe only agents at same node/edge
                - 'statistical': As local, plus counts of agents at all nodes/edges
        """
        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        
        # Observation scenario
        if observation_scenario not in ['full', 'local', 'statistical']:
            raise ValueError(f"observation_scenario must be 'full', 'local', or 'statistical', got {observation_scenario}")
        self.observation_scenario = observation_scenario
        
        # Create agent names
        human_agents = [f"human_{i}" for i in range(num_humans)]
        vehicle_agents = [f"vehicle_{i}" for i in range(num_vehicles)]
        self.possible_agents = human_agents + vehicle_agents
        
        # Store agent types for easy lookup
        self.human_agents = set(human_agents)
        self.vehicle_agents = set(vehicle_agents)

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # Agent attributes
        self.agent_attributes = {}
        for agent in human_agents:
            self.agent_attributes[agent] = {
                'speed': human_speed
            }
        for agent in vehicle_agents:
            self.agent_attributes[agent] = {
                'speed': vehicle_speed,
                'capacity': vehicle_capacity,
                'fuel_use': vehicle_fuel_use
            }
        
        # Network - if not provided, create a simple default network
        if network is None:
            network = self._create_default_network()
        self.network = network
        
        # Validate network has required attributes
        self._validate_network()
        
        self.render_mode = render_mode
        
        # Initialize np_random_seed for action space
        self.np_random_seed = None
        
        # State components (will be initialized in reset)
        self.real_time = None
        self.agent_positions = None
        self.vehicle_destinations = None
        self.human_aboard = None  # For each human: None or vehicle ID
        self.step_type = None  # One of: 'routing', 'unboarding', 'boarding', 'departing'

    def _create_default_network(self):
        """Create a simple default network for testing"""
        G = nx.DiGraph()
        # Add nodes with name attribute
        G.add_node(0, name="A")
        G.add_node(1, name="B")
        G.add_node(2, name="C")
        # Add edges with required attributes
        G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
        G.add_edge(1, 2, length=15.0, speed=5.0, capacity=10)
        G.add_edge(2, 0, length=12.0, speed=5.0, capacity=10)
        return G
    
    def create_random_2d_network(self, num_nodes=10, bidirectional_prob=0.3, 
                                 speed_mean=5.0, capacity_mean=10.0, 
                                 coord_mean=0.0, coord_std=10.0, seed=None):
        """
        Create a 2D random network using Delaunay triangulation.
        
        Args:
            num_nodes: Number of nodes to generate
            bidirectional_prob: Probability that an edge is bidirectional (otherwise random direction)
            speed_mean: Mean (scale parameter) for exponential distribution of edge speeds.
                       Exponential distribution is used to model varying traffic conditions
                       with occasional high-speed routes.
            capacity_mean: Mean (scale parameter) for exponential distribution of edge capacities.
                          Exponential distribution models varying infrastructure quality
                          with occasional high-capacity routes.
            coord_mean: Mean for 2D Gaussian distribution of node coordinates
            coord_std: Standard deviation for 2D Gaussian distribution of node coordinates
            seed: Random seed for reproducibility
            
        Returns:
            NetworkX DiGraph with nodes having 'name', 'x', 'y' attributes
            and edges having 'length', 'speed', 'capacity' attributes
        """
        # Set random seed if provided
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Generate random 2D coordinates from Gaussian distribution
        coords = rng.normal(loc=coord_mean, scale=coord_std, size=(num_nodes, 2))
        
        # Compute Delaunay triangulation
        tri = Delaunay(coords)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes with coordinates and names
        for i in range(num_nodes):
            G.add_node(i, name=f"Node_{i}", x=float(coords[i, 0]), y=float(coords[i, 1]))
        
        # Process triangulation edges
        edges_set = set()
        for simplex in tri.simplices:
            # Each simplex is a triangle with 3 vertices
            for i in range(3):
                u = simplex[i]
                v = simplex[(i + 1) % 3]
                
                # Store as undirected edge (smaller index first)
                edge = (min(u, v), max(u, v))
                edges_set.add(edge)
        
        # Add edges with attributes
        for u, v in edges_set:
            # Compute Euclidean length from coordinates
            dx = coords[v, 0] - coords[u, 0]
            dy = coords[v, 1] - coords[u, 1]
            length = float(np.sqrt(dx**2 + dy**2))
            
            # Draw speed and capacity from exponential distributions
            speed = float(rng.exponential(scale=speed_mean))
            capacity = float(rng.exponential(scale=capacity_mean))
            
            # Ensure minimum values
            speed = max(speed, 0.1)
            capacity = max(capacity, 1.0)
            
            # Decide direction(s)
            if rng.random() < bidirectional_prob:
                # Bidirectional
                G.add_edge(u, v, length=length, speed=speed, capacity=capacity)
                G.add_edge(v, u, length=length, speed=speed, capacity=capacity)
            else:
                # Unidirectional in random direction
                if rng.random() < 0.5:
                    G.add_edge(u, v, length=length, speed=speed, capacity=capacity)
                else:
                    G.add_edge(v, u, length=length, speed=speed, capacity=capacity)
        
        return G
    
    def initialize_random_positions(self, seed=None):
        """
        Initialize agent positions randomly on the network.
        Some agents at nodes, others on edges.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        nodes = list(self.network.nodes())
        edges = list(self.network.edges())
        
        if not nodes:
            raise ValueError("Network has no nodes")
        
        for agent in self.agents:
            # Randomly decide if agent is at node or on edge
            if rng.random() < 0.5 or not edges:
                # Place at random node
                node = nodes[rng.randint(len(nodes))]
                self.agent_positions[agent] = node
            else:
                # Place on random edge at random coordinate
                edge_idx = rng.randint(len(edges))
                edge = edges[edge_idx]
                edge_length = self.network[edge[0]][edge[1]]['length']
                coord = float(rng.uniform(0, edge_length))
                self.agent_positions[agent] = (edge, coord)

    
    def _validate_network(self):
        """Validate that the network has all required attributes"""
        # Check node attributes
        for node in self.network.nodes():
            if 'name' not in self.network.nodes[node]:
                raise ValueError(f"Node {node} missing required 'name' attribute")
        
        # Check edge attributes
        for u, v in self.network.edges():
            edge_data = self.network[u][v]
            required_attrs = ['length', 'speed', 'capacity']
            for attr in required_attrs:
                if attr not in edge_data:
                    raise ValueError(f"Edge ({u}, {v}) missing required '{attr}' attribute")

    # Observation space should be defined here.
    # Observation spaces change based on observation_scenario, so caching is disabled
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # Observations are returned as dictionaries, so we use DictSpace
        # The exact structure depends on observation_scenario and current state
        # For simplicity, we return a flexible dict space
        # In practice, observations will be Python dicts that can contain various data
        
        # Return a generic dict space - actual observations will be dicts
        # This is a placeholder that allows any dict structure
        return DictSpace({
            'agent': Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })

    # Action space should be defined here.
    # Action spaces change based on step_type, so caching is disabled
    def action_space(self, agent):
        """
        Return the action space for the agent based on current step_type.
        
        Action spaces by step_type:
        - routing: vehicles at nodes can set destination to None or any node
        - unboarding: humans aboard vehicles at nodes can pass or unboard
        - boarding: humans at nodes can pass or board any vehicle at same node
        - departing: vehicles at nodes can pass or choose outgoing edge;
                     humans at nodes (not aboard) can pass or choose outgoing edge
        """
        if self.step_type is None or agent not in self.agents:
            # Default fallback - single action (pass)
            return Discrete(1, seed=self.np_random_seed)
        
        if self.step_type == 'routing':
            if agent in self.vehicle_agents:
                pos = self.agent_positions.get(agent)
                # Only vehicles at nodes can act
                if pos is not None and not isinstance(pos, tuple):
                    # Index 0 = set destination to None
                    # Index 1..N = set destination to node 0, 1, ..., N-1
                    num_nodes = len(self.network.nodes())
                    return Discrete(num_nodes + 1, seed=self.np_random_seed)
            # All other agents can only pass
            return Discrete(1, seed=self.np_random_seed)
        
        elif self.step_type == 'unboarding':
            if agent in self.human_agents:
                aboard = self.human_aboard.get(agent)
                if aboard is not None:
                    vehicle_pos = self.agent_positions.get(aboard)
                    # Only humans aboard vehicles at nodes can act
                    if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
                        # Actions: 0=pass, 1=unboard
                        return Discrete(2, seed=self.np_random_seed)
            # All other agents can only pass
            return Discrete(1, seed=self.np_random_seed)
        
        elif self.step_type == 'boarding':
            if agent in self.human_agents:
                pos = self.agent_positions.get(agent)
                aboard = self.human_aboard.get(agent)
                # Only humans at nodes and not aboard can act
                if pos is not None and not isinstance(pos, tuple) and aboard is None:
                    # Find vehicles at the same node
                    vehicles_at_node = [
                        v for v in self.vehicle_agents 
                        if self.agent_positions.get(v) == pos
                    ]
                    # Actions: 0=pass, 1..N=board vehicle 0, 1, ..., N-1
                    return Discrete(len(vehicles_at_node) + 1, seed=self.np_random_seed)
            # All other agents can only pass
            return Discrete(1, seed=self.np_random_seed)
        
        elif self.step_type == 'departing':
            pos = self.agent_positions.get(agent)
            # Check if agent is at a node
            if pos is not None and not isinstance(pos, tuple):
                if agent in self.vehicle_agents:
                    # Vehicles at nodes can choose outgoing edges
                    outgoing_edges = list(self.network.out_edges(pos))
                    # Actions: 0=pass, 1..N=depart into edge 0, 1, ..., N-1
                    return Discrete(len(outgoing_edges) + 1, seed=self.np_random_seed)
                elif agent in self.human_agents:
                    aboard = self.human_aboard.get(agent)
                    # Humans at nodes and not aboard can choose outgoing edges
                    if aboard is None:
                        outgoing_edges = list(self.network.out_edges(pos))
                        # Actions: 0=pass, 1..N=walk into edge 0, 1, ..., N-1
                        return Discrete(len(outgoing_edges) + 1, seed=self.np_random_seed)
            # All other agents can only pass
            return Discrete(1, seed=self.np_random_seed)
        
        # Default fallback
        return Discrete(1, seed=self.np_random_seed)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) > 0:
            print(f"Current state: real_time={self.real_time:.2f}, step_type={self.step_type}")
            for agent in self.agents:
                pos = self.agent_positions[agent]
                if isinstance(pos, tuple):
                    edge, coord = pos
                    print(f"  {agent}: on edge {edge}, coordinate {coord:.2f}")
                else:
                    print(f"  {agent}: at node {pos}")
                if agent in self.vehicle_agents:
                    dest = self.vehicle_destinations[agent]
                    print(f"    destination: {dest}")
                elif agent in self.human_agents:
                    aboard = self.human_aboard[agent]
                    print(f"    aboard: {aboard}")
        else:
            print("Environment terminated")

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.agents = self.possible_agents[:]
        
        # Initialize state components
        self.real_time = 0.0
        
        # Initialize agent positions - all agents start at first node
        nodes = list(self.network.nodes())
        self.agent_positions = {agent: nodes[0] for agent in self.agents}
        
        # Initialize vehicle destinations - all start with None
        self.vehicle_destinations = {agent: None for agent in self.vehicle_agents}
        
        # Initialize human aboard status - all start with None (not aboard)
        self.human_aboard = {agent: None for agent in self.human_agents}
        
        # Initialize step type - start with routing
        self.step_type = 'routing'
        
        # Create observations based on scenario
        observations = self._generate_observations()
        infos = {agent: {} for agent in self.agents}

        return observations, infos
    
    def _generate_observations(self):
        """Generate observations for all agents based on observation_scenario"""
        observations = {}
        for agent in self.agents:
            observations[agent] = self._generate_observation_for_agent(agent)
        return observations
    
    def _generate_observation_for_agent(self, agent):
        """Generate observation for a single agent based on observation_scenario"""
        if self.observation_scenario == 'full':
            return self._generate_full_observation(agent)
        elif self.observation_scenario == 'local':
            return self._generate_local_observation(agent)
        elif self.observation_scenario == 'statistical':
            return self._generate_statistical_observation(agent)
        else:
            return {}
    
    def _generate_full_observation(self, agent):
        """Full observation: agent observes the complete state"""
        obs = {
            'real_time': float(self.real_time),
            'step_type': self.step_type,
            'agent_positions': dict(self.agent_positions),
            'vehicle_destinations': dict(self.vehicle_destinations),
            'human_aboard': dict(self.human_aboard),
            'agent_attributes': dict(self.agent_attributes),
            'network_nodes': list(self.network.nodes()),
            'network_edges': [(u, v, dict(data)) for u, v, data in self.network.edges(data=True)]
        }
        return obs
    
    def _generate_local_observation(self, agent):
        """
        Local observation: agent observes only agents at same node or on same edge,
        along with their state components and attributes.
        """
        agent_pos = self.agent_positions[agent]
        
        # Find agents at same location
        agents_at_location = []
        for other_agent in self.agents:
            other_pos = self.agent_positions[other_agent]
            # Check if at same location (node or edge)
            if agent_pos == other_pos:
                agents_at_location.append(other_agent)
        
        # Build observation with info about agents at same location
        obs = {
            'real_time': float(self.real_time),
            'step_type': self.step_type,
            'my_position': agent_pos,
            'agents_here': {}
        }
        
        for other_agent in agents_at_location:
            agent_info = {
                'position': self.agent_positions[other_agent],
                'attributes': dict(self.agent_attributes[other_agent])
            }
            
            # Add type-specific state
            if other_agent in self.vehicle_agents:
                agent_info['destination'] = self.vehicle_destinations[other_agent]
            elif other_agent in self.human_agents:
                agent_info['aboard'] = self.human_aboard[other_agent]
            
            obs['agents_here'][other_agent] = agent_info
        
        return obs
    
    def _generate_statistical_observation(self, agent):
        """
        Statistical observation: as local, plus counts of humans and vehicles
        at every node and on every edge.
        """
        # Start with local observation
        obs = self._generate_local_observation(agent)
        
        # Add statistical information
        node_counts = {}
        edge_counts = {}
        
        for node in self.network.nodes():
            node_counts[node] = {'humans': 0, 'vehicles': 0}
        
        for edge in self.network.edges():
            edge_counts[edge] = {'humans': 0, 'vehicles': 0}
        
        # Count agents at each location
        for other_agent in self.agents:
            pos = self.agent_positions[other_agent]
            agent_type = 'vehicles' if other_agent in self.vehicle_agents else 'humans'
            
            if isinstance(pos, tuple) and len(pos) == 2:
                # Agent on edge (validate it's a 2-element tuple)
                edge, coord = pos
                if edge in edge_counts:
                    edge_counts[edge][agent_type] += 1
            else:
                # Agent at node
                if pos in node_counts:
                    node_counts[pos][agent_type] += 1
        
        obs['node_counts'] = node_counts
        obs['edge_counts'] = edge_counts
        
        return obs

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Process actions based on current step_type
        if self.step_type == 'routing':
            self._process_routing_actions(actions)
        elif self.step_type == 'unboarding':
            self._process_unboarding_actions(actions)
        elif self.step_type == 'boarding':
            self._process_boarding_actions(actions)
        elif self.step_type == 'departing':
            self._process_departing_actions(actions)
        
        # Generate observations based on scenario
        observations = self._generate_observations()
        
        # All rewards are constantly zero
        rewards = {agent: 0.0 for agent in self.agents}
        
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminations, truncations, infos
    
    def _process_routing_actions(self, actions):
        """
        Process routing step: vehicles at nodes can change their destination.
        Real time does not advance.
        """
        for agent, action in actions.items():
            if agent in self.vehicle_agents:
                pos = self.agent_positions.get(agent)
                # Only vehicles at nodes can route
                if pos is not None and not isinstance(pos, tuple):
                    if action == 0:
                        # Set destination to None
                        self.vehicle_destinations[agent] = None
                    else:
                        # Set destination to node (action - 1)
                        nodes = list(self.network.nodes())
                        if 1 <= action <= len(nodes):
                            self.vehicle_destinations[agent] = nodes[action - 1]
    
    def _process_unboarding_actions(self, actions):
        """
        Process unboarding step: humans aboard vehicles at nodes can unboard.
        Real time does not advance.
        """
        for agent, action in actions.items():
            if agent in self.human_agents:
                aboard = self.human_aboard.get(agent)
                if aboard is not None:
                    vehicle_pos = self.agent_positions.get(aboard)
                    # Only humans aboard vehicles at nodes can unboard
                    if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
                        if action == 1:  # action 0 is pass, action 1 is unboard
                            self.human_aboard[agent] = None
    
    def _process_boarding_actions(self, actions):
        """
        Process boarding step: humans at nodes can board vehicles at same node.
        Humans are processed in random order. Only board if vehicle not full.
        Real time does not advance.
        """
        # Get humans who want to board and their chosen vehicles
        boarding_requests = []
        for agent, action in actions.items():
            if agent in self.human_agents and action > 0:  # action 0 is pass
                pos = self.agent_positions.get(agent)
                aboard = self.human_aboard.get(agent)
                # Only humans at nodes and not aboard can board
                if pos is not None and not isinstance(pos, tuple) and aboard is None:
                    # Find vehicles at same node
                    vehicles_at_node = [
                        v for v in self.vehicle_agents 
                        if self.agent_positions.get(v) == pos
                    ]
                    # action - 1 gives the index in vehicles_at_node list
                    vehicle_idx = action - 1
                    if 0 <= vehicle_idx < len(vehicles_at_node):
                        chosen_vehicle = vehicles_at_node[vehicle_idx]
                        boarding_requests.append((agent, chosen_vehicle))
        
        # Process boarding requests in random order
        if boarding_requests:
            self.np_random.shuffle(boarding_requests)
            for human, vehicle in boarding_requests:
                # Count humans already aboard this vehicle
                humans_aboard = sum(1 for h in self.human_agents 
                                   if self.human_aboard.get(h) == vehicle)
                capacity = self.agent_attributes[vehicle]['capacity']
                
                # Board if vehicle not full
                if humans_aboard < capacity:
                    self.human_aboard[human] = vehicle
    
    def _get_agent_speed(self, agent, edge_data):
        """
        Get the speed of an agent.
        Vehicles use the edge's speed rating, humans use their own speed attribute.
        """
        if agent in self.vehicle_agents:
            return edge_data['speed']
        elif agent in self.human_agents:
            return self.agent_attributes[agent]['speed']
        else:
            return 1.0  # Fallback
    
    def _process_departing_actions(self, actions):
        """
        Process departing step: vehicles and humans (not aboard) at nodes can depart/walk into edges.
        All agents on edges move. Real time advances by minimum remaining duration on edges.
        """
        # First, process departing actions for agents at nodes
        for agent, action in actions.items():
            if action > 0:  # action 0 is pass
                pos = self.agent_positions.get(agent)
                # Check if agent is at a node
                if pos is not None and not isinstance(pos, tuple):
                    # Get outgoing edges from this node
                    outgoing_edges = list(self.network.out_edges(pos))
                    edge_idx = action - 1
                    
                    if 0 <= edge_idx < len(outgoing_edges):
                        chosen_edge = outgoing_edges[edge_idx]
                        
                        # Vehicles can always depart
                        if agent in self.vehicle_agents:
                            self.agent_positions[agent] = (chosen_edge, 0.0)
                        # Humans can only depart if not aboard
                        elif agent in self.human_agents:
                            aboard = self.human_aboard.get(agent)
                            if aboard is None:
                                self.agent_positions[agent] = (chosen_edge, 0.0)
        
        # Now compute movement for all agents on edges
        # Find minimum remaining duration on edges
        remaining_durations = []
        
        for agent in self.agents:
            pos = self.agent_positions.get(agent)
            if pos is not None and isinstance(pos, tuple):
                edge, coord = pos
                edge_data = self.network[edge[0]][edge[1]]
                edge_length = edge_data['length']
                
                # Validate coordinate is within valid range
                if coord > edge_length + self.FLOAT_EPSILON:
                    # Coordinate exceeds edge length - should not happen
                    # Clamp to edge length
                    coord = edge_length
                    self.agent_positions[agent] = (edge, coord)
                
                remaining_distance = edge_length - coord
                
                # Only compute duration if there's remaining distance
                if remaining_distance > self.FLOAT_EPSILON:
                    speed = self._get_agent_speed(agent, edge_data)
                    
                    if speed > 0:
                        duration = remaining_distance / speed
                        remaining_durations.append(duration)
        
        # If there are agents on edges, advance time and move them
        if remaining_durations:
            delta_t = min(remaining_durations)
            self.real_time += delta_t
            
            # Move all agents on edges
            for agent in self.agents:
                pos = self.agent_positions.get(agent)
                if pos is not None and isinstance(pos, tuple):
                    edge, coord = pos
                    edge_data = self.network[edge[0]][edge[1]]
                    
                    speed = self._get_agent_speed(agent, edge_data)
                    
                    # Update coordinate
                    new_coord = coord + speed * delta_t
                    edge_length = edge_data['length']
                    
                    # Check if agent has reached the end of the edge
                    if abs(new_coord - edge_length) < self.FLOAT_EPSILON:
                        # Agent arrives at target node
                        target_node = edge[1]
                        self.agent_positions[agent] = target_node
                    else:
                        # Agent still on edge
                        self.agent_positions[agent] = (edge, new_coord)
        
        # Update positions of humans aboard vehicles to match vehicle positions
        for human in self.human_agents:
            aboard = self.human_aboard.get(human)
            if aboard is not None:
                self.agent_positions[human] = self.agent_positions[aboard]
