import functools
from typing import Optional, Union, Tuple, Dict, Any
import os

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, Tuple as TupleSpace
from gymnasium.utils import seeding
import networkx as nx
from scipy.spatial import Delaunay

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG

DEFAULT_NUM_HUMANS = DEFAULT_EXPERIMENT_CONFIG.scenario.num_humans
DEFAULT_NUM_VEHICLES = DEFAULT_EXPERIMENT_CONFIG.scenario.num_vehicles
DEFAULT_HUMAN_WALKING_SPEED_KMH = DEFAULT_EXPERIMENT_CONFIG.mobility.human_walking_speed_kmh
DEFAULT_VEHICLE_SPEED_KMH = DEFAULT_EXPERIMENT_CONFIG.mobility.vehicle_speed_kmh


def env(render_mode=None, num_humans=DEFAULT_NUM_HUMANS, num_vehicles=DEFAULT_NUM_VEHICLES, network=None,
        human_speeds=None, vehicle_speeds=None, vehicle_capacities=None, vehicle_fuel_uses=None,
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
        human_speeds=human_speeds,
        vehicle_speeds=vehicle_speeds,
        vehicle_capacities=vehicle_capacities,
        vehicle_fuel_uses=vehicle_fuel_uses,
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


def raw_env(render_mode=None, num_humans=DEFAULT_NUM_HUMANS, num_vehicles=DEFAULT_NUM_VEHICLES, network=None,
            human_speeds=None, vehicle_speeds=None, vehicle_capacities=None, vehicle_fuel_uses=None,
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
        human_speeds=human_speeds,
        vehicle_speeds=vehicle_speeds,
        vehicle_capacities=vehicle_capacities,
        vehicle_fuel_uses=vehicle_fuel_uses,
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
        num_humans=DEFAULT_NUM_HUMANS,
        num_vehicles=DEFAULT_NUM_VEHICLES,
        network=None,
        human_speeds=None,
        vehicle_speeds=None,
        vehicle_capacities=None,
        vehicle_fuel_uses=None,
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
            human_speeds: List of walking speeds in km/h for each human, or
                None to use default (3.0 km/h for all)
            vehicle_speeds: List of maximum speeds in km/h for each vehicle, or
                None to use default (30.0 km/h for all). Vehicle movement uses
                min(vehicle speed, current edge speed).
            vehicle_capacities: List of capacities for each vehicle, or None to use default (4 for all)
            vehicle_fuel_uses: List of fuel uses for each vehicle, or None to use default (1.0 for all)
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
        
        # Agent attributes - use lists if provided, otherwise use defaults
        if human_speeds is None:
            human_speeds = [DEFAULT_HUMAN_WALKING_SPEED_KMH] * num_humans
        elif len(human_speeds) != num_humans:
            raise ValueError(f"human_speeds must have length {num_humans}, got {len(human_speeds)}")
            
        if vehicle_speeds is None:
            vehicle_speeds = [DEFAULT_VEHICLE_SPEED_KMH] * num_vehicles
        elif len(vehicle_speeds) != num_vehicles:
            raise ValueError(f"vehicle_speeds must have length {num_vehicles}, got {len(vehicle_speeds)}")
            
        if vehicle_capacities is None:
            vehicle_capacities = [4] * num_vehicles
        elif len(vehicle_capacities) != num_vehicles:
            raise ValueError(f"vehicle_capacities must have length {num_vehicles}, got {len(vehicle_capacities)}")
            
        if vehicle_fuel_uses is None:
            vehicle_fuel_uses = [1.0] * num_vehicles
        elif len(vehicle_fuel_uses) != num_vehicles:
            raise ValueError(f"vehicle_fuel_uses must have length {num_vehicles}, got {len(vehicle_fuel_uses)}")
        
        self.agent_attributes = {}
        for i, agent in enumerate(human_agents):
            self.agent_attributes[agent] = {
                'speed': float(human_speeds[i])
            }
        for i, agent in enumerate(vehicle_agents):
            self.agent_attributes[agent] = {
                'speed': float(vehicle_speeds[i]),
                'capacity': int(vehicle_capacities[i]),
                'fuel_use': float(vehicle_fuel_uses[i])
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
        # Initialize np_random (will be properly seeded in reset)
        from gymnasium.utils import seeding
        self.np_random, _ = seeding.np_random(None)
        
        # Rendering state
        self.fig = None
        self.ax = None
        self.frames = []  # For video recording
        self._time_per_frame = 0.02  # 0.2/0.5/ Capture frame every 0.02 time units ("clicks")
        self._recording = False
        
        # State for video rendering interpolation
        self._last_event_time = 0.0  # Time of last arrival event (when render was called after step)
        self._positions_at_last_event = {}  # Agent positions at last event
        self._speeds_at_last_event = {}  # Agent speeds at last event (for agents on edges)
        
        # Artist-based rendering: persistent matplotlib objects
        self._artists_initialized = False
        self._node_artists = {}  # node -> Circle artist
        self._edge_artists = []  # List of Line2D artists for edges
        self._vehicle_artists = {}  # vehicle -> Rectangle artist
        self._human_artists = {}  # human -> Circle artist  
        self._passenger_artists = {}  # (vehicle, human) -> Circle artist
        self._destination_artists = {}  # vehicle -> Line2D artist for destination arc
        self._network_pos = None  # Cached node positions
        
        # State components (will be initialized in reset)
        self.real_time = None
        self.agent_positions = None
        self.vehicle_destinations = None
        self.human_destinations = None
        self.human_aboard = None  # For each human: None or vehicle ID
        self.termination_participants = None
        self._step_type = None  # One of: 'routing', 'unboarding', 'boarding', 'departing' (private, use step_type property)

        # Termination modes:
        # "initial_active" : only agents whose destination is NOT None at episode start participate in termination (participants are frozen once)
        # "current_active" : agents with a non-None destination at the current timestep participate dynamically in termination.
        # "all_agents"     : all agents participate in termination (episode ends only when every agent is not on an edge and has reached its destination)
        self.termination_mode = "initial_active"

        # Cached network observation data (constant throughout episode)
        self._cached_network_nodes = None
        self._cached_network_edges = None
    
    @property
    def step_type(self):
        """Read-only property for current step type in the cycle.
        
        The step type automatically cycles through: routing -> unboarding -> boarding -> departing
        when step() is called. To skip a step type, call step() with default actions (all agents pass).
        
        Returns:
            str: One of 'routing', 'unboarding', 'boarding', 'departing'
        """
        return self._step_type
    
    def _set_step_type_for_testing(self, step_type: str):
        """Internal method to set step_type directly for testing purposes only.
        
        This method bypasses the normal step cycle and should ONLY be used in unit tests.
        Regular code should never call this - use step() with pass actions to cycle through steps.
        
        Args:
            step_type: One of 'routing', 'unboarding', 'boarding', 'departing'
        """
        if step_type not in ['routing', 'unboarding', 'boarding', 'departing']:
            raise ValueError(f"Invalid step_type: {step_type}")
        self._step_type = step_type

    def _create_default_network(self):
        """Create a simple default network for testing"""
        G = nx.DiGraph()
        # Add nodes with name attribute
        G.add_node(0, name="A")
        G.add_node(1, name="B")
        G.add_node(2, name="C")

        # Base directed cycle (ensures strong connectivity)
        base_edges = [(0, 1), (1, 2), (2, 0)]
        for u, v in base_edges:
            G.add_edge(u, v, length=10.0, speed=DEFAULT_VEHICLE_SPEED_KMH, capacity=10)

        # Deterministically make selected cycle edges bidirectional to add variety.
        for u, v in base_edges[:2]:
            if not G.has_edge(v, u):
                G.add_edge(v, u, length=G[u][v]['length'], speed=G[u][v]['speed'], capacity=G[u][v]['capacity'])

        return G
    
    def create_random_2d_network(self, num_nodes=10, bidirectional_prob=0.85,
                                 speed_mean=DEFAULT_VEHICLE_SPEED_KMH, capacity_mean=10.0, 
                                 coord_mean=0.0, coord_std=10.0, seed=None):
        """
        Create a 2D random network using Delaunay triangulation.
        
        Args:
            num_nodes: Number of nodes to generate
            bidirectional_prob: Probability that an edge is bidirectional (otherwise random direction)
            speed_mean: Vehicle speed in km/h assigned to generated edges.
            capacity_mean: Mean (scale parameter) for exponential distribution of edge capacities.
                          Exponential distribution models varying infrastructure quality
                          with occasional high-capacity routes.
            coord_mean: Mean for 2D Gaussian distribution of node coordinates
            coord_std: Standard deviation for 2D Gaussian distribution of node coordinates
            seed: Random seed for reproducibility
            
        Returns:
            NetworkX DiGraph with nodes having 'name', 'x', 'y' attributes
            and edges having 'length' in km, 'speed' in km/h, and 'capacity'
            attributes. Time advances in hours because duration = length / speed.
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
            
            # Use a fixed default vehicle speed so time remains interpretable.
            speed = float(speed_mean)
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

            # Ensure strong connectivity: if not strongly connected, connect SCCs in a directed cycle
            if not nx.is_strongly_connected(G):
                sccs = list(nx.strongly_connected_components(G))
                # Choose one representative node from each SCC
                reps = [min(scc) for scc in sccs]

                # Create bridging edges between consecutive SCC representatives
                for i in range(len(reps)):
                    a = reps[i]
                    b = reps[(i + 1) % len(reps)]
                    if not G.has_edge(a, b):
                        # compute length between a and b (fallback to euclidean distance)
                        dx = coords[b, 0] - coords[a, 0]
                        dy = coords[b, 1] - coords[a, 1]
                        length = float(np.sqrt(dx ** 2 + dy ** 2))
                        speed = float(speed_mean)
                        capacity = float(rng.exponential(scale=capacity_mean))
                        speed = max(speed, 0.1)
                        capacity = max(capacity, 1.0)
                        G.add_edge(a, b, length=length, speed=speed, capacity=capacity)
                # After adding a cycle among SCCs the graph should be strongly connected.
                # As a safety, if still not strongly connected (very unlikely), add reverse edges
                if not nx.is_strongly_connected(G):
                    # add reverse edges for any isolated directions until connected
                    for u, v in list(G.edges()):
                        if not G.has_edge(v, u):
                            data = G[u][v]
                            G.add_edge(
                                v, u,
                                length=data.get('length', 1.0),
                                speed=data.get('speed', 1.0),
                                capacity=data.get('capacity', 1.0),
                            )
                        if nx.is_strongly_connected(G):
                            break

        return G
    
    def initialize_random_positions(self, seed=None):
        """
        Initialize agent positions randomly on the network.
        Some agents at nodes, others on edges.
        Also randomly initialize human aboard status.
        
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
        
        # Initialize human aboard status
        # Randomly assign some humans to be aboard vehicles
        for human in self.human_agents:
            # 30% chance of being aboard a vehicle (if there are any vehicles)
            if self.vehicle_agents and rng.random() < 0.3:
                # Choose a random vehicle
                vehicle = list(self.vehicle_agents)[rng.randint(len(self.vehicle_agents))]
                self.human_aboard[human] = vehicle
                # Sync human position with vehicle
                self.agent_positions[human] = self.agent_positions[vehicle]
            else:
                self.human_aboard[human] = None

    
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
                        if not isinstance(self.agent_positions.get(v), tuple) and self.agent_positions.get(v) == pos
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

    def render(self, goal_info=None, value_dict=None, title=None):
        """
        Renders the environment using the node coordinates.
        
        When recording video, automatically generates frames at uniform time intervals
        (every _time_per_frame "clicks" of continuous time) to ensure smooth playback.
        
        - Vehicles shown as blue rectangles with human passengers inside
        - Humans shown as red dots
        - Bidirectional roads shown with two separate lanes
        - Goal node shown with dashed outline and star marker
        - Dashed line connects agent to their goal
        
        Args:
            goal_info: Optional dict with goal visualization info:
                - 'agent': agent name who has the goal
                - 'node': target node for the goal
                - 'type': 'node' or 'cluster'
            value_dict: Optional dict mapping nodes to V-values for coloring
            title: Optional custom title (overrides default)
        
        Returns:
            matplotlib figure (if graphical rendering) or None
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        
        # Store render parameters for later use
        self._last_goal_info = goal_info
        self._last_value_dict = value_dict
        self._last_title = title
        
        # Check if graphical rendering is enabled
        use_graphical = getattr(self, '_use_graphical', False)
        
        # Text-only rendering for backwards compatibility
        if self.render_mode == "human" and not use_graphical:
            self._render_text()
            return
        
        # Graphical rendering with uniform time intervals for video recording
        if hasattr(self, '_recording') and self._recording:
            # Generate multiple frames at uniform time intervals
            self._render_uniform_frames(goal_info=goal_info, value_dict=value_dict, title=title)
        else:
            # Single frame for non-recording
            return self._render_graphical(goal_info=goal_info, value_dict=value_dict, title=title)
    
    def _render_text(self):
        """Original text-based rendering"""
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
                    dest = self.human_destinations.get(agent)
                    print(f"    aboard: {aboard}")
                    print(f"    target: {dest}")
        else:
            print("Environment terminated")
    
    def _render_uniform_frames(self, goal_info=None, value_dict=None, title=None):
        """
        Render frames at uniform time intervals for smooth video playback.
        
        Generates frames at integer multiples of time_per_frame (clicks) that fall after
        the last arrival event.
        
        Key variables:
        - t_lastevent: Time of last arrival event (when render() was called after step())
        - t_currentevent: Current time (this arrival event)
        - At t_lastevent, we saved each agent's position and speed (if on edge)
        - For any click t > t_lastevent: position_at_t = position_at_lastevent + speed * (t - t_lastevent)
        
        Args:
            goal_info: Optional dict with goal visualization info
            value_dict: Optional dict mapping nodes to V-values for coloring
            title: Optional custom title (overrides default)
        """
        t_currentevent = self.real_time  # Current arrival event time
        
        # Get saved state from last event
        if not hasattr(self, '_last_event_time'):
            # First render call - initialize
            self._last_event_time = 0.0
            self._positions_at_last_event = {}
            self._speeds_at_last_event = {}
        
        t_lastevent = self._last_event_time
        positions_at_lastevent = self._positions_at_last_event if self._positions_at_last_event else self.agent_positions.copy()
        speeds_at_lastevent = self._speeds_at_last_event if self._speeds_at_last_event else {}
        
        # Find all clicks (integer multiples of time_per_frame) in the range (t_lastevent, t_currentevent]
        click_interval = self._time_per_frame
        
        # First click index after t_lastevent
        first_click_index = int(t_lastevent / click_interval) + 1
        
        # Last click index at or before t_currentevent
        last_click_index = int(t_currentevent / click_interval)
        
        # Generate frames for all clicks
        if first_click_index <= last_click_index:
            # Save current positions (at t_currentevent)
            saved_positions = self.agent_positions.copy()
            
            for click_index in range(first_click_index, last_click_index + 1):
                t_click = click_index * click_interval
                
                # Compute agent positions at this click time
                for agent in self.agents:
                    pos_at_lastevent = positions_at_lastevent.get(agent)
                    speed = speeds_at_lastevent.get(agent, 0.0)
                    
                    if isinstance(pos_at_lastevent, tuple):
                        # Agent was on edge at last event
                        edge, coord_at_lastevent = pos_at_lastevent
                        
                        # Compute position at click: coord = coord_at_lastevent + speed * (t_click - t_lastevent)
                        elapsed = t_click - t_lastevent
                        coord_at_click = coord_at_lastevent + speed * elapsed
                        
                        # Get edge length
                        edge_data = self.network[edge[0]][edge[1]]
                        edge_length = edge_data['length']
                        
                        # Check if agent would have reached end of edge
                        if coord_at_click >= edge_length - 0.001:
                            # Agent reached end - check if this happened by t_currentevent
                            elapsed_to_current = t_currentevent - t_lastevent
                            coord_at_currentevent = coord_at_lastevent + speed * elapsed_to_current
                            
                            if coord_at_currentevent >= edge_length - 0.001:
                                # Agent arrived at node at or before t_currentevent
                                self.agent_positions[agent] = edge[1]
                            else:
                                # Agent hasn't arrived yet, clamp to edge
                                self.agent_positions[agent] = (edge, min(coord_at_click, edge_length))
                        else:
                            # Agent still on edge
                            self.agent_positions[agent] = (edge, coord_at_click)
                    else:
                        # Agent was at node at last event, stays there
                        self.agent_positions[agent] = pos_at_lastevent
                
                # Synchronize humans aboard vehicles
                for human in self.human_agents:
                    aboard = self.human_aboard.get(human)
                    if aboard is not None:
                        self.agent_positions[human] = self.agent_positions[aboard]
                
                # Count humans aboard vehicles
                humans_aboard = sum(1 for h in self.human_agents if self.human_aboard.get(h) is not None)
                
                # Create title
                frame_title = f"Time: {t_click:.2f}s | Humans aboard: {humans_aboard}"
                
                # Render this frame
                self._render_single_frame(
                    goal_info=goal_info,
                    value_dict=value_dict,
                    title=frame_title,
                    capture_frame=True
                )
            
            # Restore current positions (at t_currentevent)
            self.agent_positions = saved_positions
        
        # Note: We don't save state here anymore - it's saved BEFORE movement
        # in _process_departing_actions() so that we have the correct starting positions

        # ---- FIX: advance render anchor to current event time ----
        self._last_event_time = t_currentevent
        self._positions_at_last_event = self.agent_positions.copy()


        self._speeds_at_last_event = {}
        for agent in self.agents:
            pos = self.agent_positions.get(agent)
            if isinstance(pos, tuple):
                edge, _ = pos
                edge_data = self.network[edge[0]][edge[1]]
                self._speeds_at_last_event[agent] = self._get_agent_speed(agent, edge_data)
            else:
                self._speeds_at_last_event[agent] = 0.0


    def _render_graphical(self, goal_info=None, value_dict=None, title=None):
        """
        Graphical rendering using matplotlib.
        
        Args:
            goal_info: Optional dict with goal visualization info:
                - 'agent': agent name who has the goal
                - 'node': target node for the goal
                - 'type': 'node' or 'cluster'
            value_dict: Optional dict mapping nodes to V-values for coloring
            title: Optional custom title (overrides default)
        """
        return self._render_single_frame(
            goal_info=goal_info,
            value_dict=value_dict,
            title=title,
            capture_frame=False
        )
    
    def _render_single_frame(self, goal_info=None, value_dict=None, title=None, capture_frame=False):
        """
        Internal method to render a single frame using persistent artists.
        Instead of clearing and redrawing, updates existing artist properties.
        
        Args:
            goal_info: Optional dict with goal visualization info
            value_dict: Optional dict mapping nodes to V-values for coloring
            title: Optional custom title (overrides default)
            capture_frame: If True, captures frame for video recording
        """
        # Initialize artists if not done yet
        if not self._artists_initialized:
            self._initialize_artists()
        
        # Import needed modules
        from matplotlib.transforms import Affine2D
        import matplotlib.pyplot as plt
        from accessibility_equity.visualization.video_display import (
            compute_human_accessibility_values,
            reset_dynamic_labels,
            update_human_overlay,
            update_passenger_overlay,
            update_summary_overlay,
            update_vehicle_overlay,
        )
        
        pos = self._network_pos
        accessibility_values = compute_human_accessibility_values(self)
        reset_dynamic_labels(self)
        
        # Update vehicle artists
        for vehicle in self.vehicle_agents:
            vehicle_pos = self.agent_positions.get(vehicle)
            artist = self._vehicle_artists[vehicle]
            
            if vehicle_pos is None:
                artist.set_visible(False)
                continue
            
            # Get vehicle position
            if isinstance(vehicle_pos, tuple):
                # On edge
                edge, coord = vehicle_pos
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                edge_length = self.network[edge[0]][edge[1]]['length']
                
                if edge_length > 0:
                    t = coord / edge_length
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    
                    # Calculate rotation angle to align with road
                    dx = x2 - x1
                    dy = y2 - y1
                    rotation_angle = np.degrees(np.arctan2(dy, dx))
                    
                    # FIX: Vehicle lane positioning
                    # Vehicles should stay on one lane (offset perpendicular to direction)
                    # Check if bidirectional and apply consistent lane offset
                    if self.network.has_edge(edge[1], edge[0]):  # Bidirectional
                        # Calculate perpendicular offset (same as edge rendering)
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            px = -dy / length * 0.15
                            py = dx / length * 0.15
                            # Always use same lane direction (vehicle traveling u→v uses +px,+py lane)
                            x += px
                            y += py
                else:
                    x, y = x1, y1
                    rotation_angle = 0
            else:
                # At node
                x, y = pos[vehicle_pos]
                rotation_angle = 0
            
            # Update vehicle rectangle position and rotation
            capacity = self.agent_attributes.get(vehicle, {}).get('capacity', 4)
            vehicle_width = max(0.8, 0.4 + capacity * 0.25)
            vehicle_height = 0.3
            
            artist.set_xy((x - vehicle_width/2, y - vehicle_height/2))
            artist.set_width(vehicle_width)
            artist.set_height(vehicle_height)
            
            # Apply rotation
            t = Affine2D().rotate_deg_around(x, y, rotation_angle) + self.ax.transData
            artist.set_transform(t)
            artist.set_visible(True)
            update_vehicle_overlay(self, vehicle, (x, y))
            
            # Update passengers inside vehicle
            passengers = [h for h in self.human_agents if self.human_aboard.get(h) == vehicle]
            num_passengers = len(passengers)
            
            for i in range(capacity):
                passenger_artist = self._passenger_artists.get((vehicle, i))
                if passenger_artist:
                    if i < num_passengers:
                        # Position passenger inside vehicle
                        if num_passengers == 1:
                            offset_x = 0
                        else:
                            spacing = vehicle_width * 0.8
                            offset_x = -spacing/2 + (i * spacing / (num_passengers - 1))
                        
                        passenger_artist.set_center((x + offset_x, y))
                        passenger_artist.set_visible(True)
                    else:
                        passenger_artist.set_visible(False)
            update_passenger_overlay(self, vehicle, passengers)
            
            # Update destination arc for vehicle
            dest = self.vehicle_destinations.get(vehicle)
            dest_artist = self._destination_artists[vehicle]
            
            if dest is not None and dest in pos:
                dest_x, dest_y = pos[dest]
                dx = dest_x - x
                dy = dest_y - y
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0.5:
                    # Generate curved arc points
                    base_angle = np.arctan2(dy, dx)
                    ctrl_distance = distance * 0.3
                    start_offset = np.pi / 3
                    end_offset = np.pi / 3
                    
                    ctrl1_x = x + ctrl_distance * np.cos(base_angle + start_offset)
                    ctrl1_y = y + ctrl_distance * np.sin(base_angle + start_offset)
                    ctrl2_x = dest_x + ctrl_distance * np.cos(base_angle + np.pi - end_offset)
                    ctrl2_y = dest_y + ctrl_distance * np.sin(base_angle + np.pi - end_offset)
                    
                    # Cubic Bezier curve
                    t_values = np.linspace(0, 1, 40)
                    curve_x, curve_y = [], []
                    for t in t_values:
                        s = 1 - t
                        bx = s**3 * x + 3*s**2*t * ctrl1_x + 3*s*t**2 * ctrl2_x + t**3 * dest_x
                        by = s**3 * y + 3*s**2*t * ctrl1_y + 3*s*t**2 * ctrl2_y + t**3 * dest_y
                        curve_x.append(bx)
                        curve_y.append(by)
                    
                    dest_artist.set_data(curve_x, curve_y)
                    dest_artist.set_visible(True)
                else:
                    dest_artist.set_visible(False)
            else:
                dest_artist.set_visible(False)
        
        # Update human artists (only those not aboard vehicles)
        for human in self.human_agents:
            aboard = self.human_aboard.get(human)
            artist = self._human_artists[human]
            
            if aboard is not None:
                # Human is aboard a vehicle - hide walking artist (shown as passenger)
                artist.set_visible(False)
                continue
            
            # Human walking - update position
            human_pos = self.agent_positions.get(human)
            if human_pos is None:
                artist.set_visible(False)
                continue
            
            if isinstance(human_pos, tuple):
                # On edge
                edge, coord = human_pos
                x1, y1 = pos[edge[0]]
                x2, y2 = pos[edge[1]]
                edge_length = self.network[edge[0]][edge[1]]['length']
                
                if edge_length > 0:
                    t = coord / edge_length
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    
                    # Apply lane offset for bidirectional roads
                    u, v = edge[0], edge[1]
                    if self.network.has_edge(u, v) and self.network.has_edge(v, u):
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            px = -dy / length * 0.15
                            py = dx / length * 0.15
                            x += px
                            y += py
                else:
                    x, y = x1, y1
            else:
                # At node
                x, y = pos[human_pos]
            
            artist.set_center((x, y))
            artist.set_visible(True)
            update_human_overlay(self, human, (x, y), accessibility_values)
        
        # Update title
        if title is not None:
            self.ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            self.ax.set_title(f'Transport Network - Time: {self.real_time:.2f}, Step: {self.step_type}',
                             fontsize=14, fontweight='bold')
        update_summary_overlay(self, accessibility_values, title=title)
        
        # Draw canvas
        if self.fig is not None:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        
        # Capture frame if requested
        if capture_frame and hasattr(self, '_recording') and self._recording:
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3].copy()
            self.frames.append(frame)
        
        return self.fig
    
    def render_frame(self, goal_info=None, value_dict=None, title=None):
        """
        Render the current state and return as RGB array.
        
        This is useful for creating videos or animations outside of the
        built-in video recording system.
        
        Args:
            goal_info: Optional dict with goal visualization info:
                - 'agent': agent name who has the goal
                - 'node': target node for the goal
                - 'type': 'node' or 'cluster'
            value_dict: Optional dict mapping nodes to V-values for coloring
            title: Optional custom title (overrides default)
        
        Returns:
            RGB numpy array of the rendered frame
        """
        # Temporarily enable graphical rendering
        old_render_mode = self.render_mode
        old_use_graphical = getattr(self, '_use_graphical', False)
        
        self.render_mode = 'human'
        self._use_graphical = True
        
        # Render
        self._render_graphical(goal_info=goal_info, value_dict=value_dict, title=title)
        
        # Convert figure to RGB array
        if self.fig is not None:
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]  # Convert RGBA to RGB
        else:
            frame = None
        
        # Restore original state
        self.render_mode = old_render_mode
        self._use_graphical = old_use_graphical
        
        return frame
    
    def enable_rendering(self, mode='graphical'):
        """Enable graphical or text rendering"""
        if mode == 'graphical':
            self._use_graphical = True
        else:
            self._use_graphical = False
    
    def start_video_recording(self):
        """Start recording frames for video - initialize persistent artists"""
        self._recording = True
        self.frames = []
        self.enable_rendering('graphical')
        
        # Initialize matplotlib figure and persistent artists
        self._initialize_artists()
    
    def _initialize_artists(self):
        """
        Initialize persistent matplotlib artist objects for efficient rendering.
        Called once at start of video recording.
        """
        # Import matplotlib
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
            from matplotlib.lines import Line2D
        except ImportError:
            print("matplotlib is required for graphical rendering")
            return
        
        # Create figure if needed
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Remove all margins and padding - maximize network visibility
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Get node positions
        self._network_pos = {}
        for node in self.network.nodes():
            if 'x' in self.network.nodes[node] and 'y' in self.network.nodes[node]:
                self._network_pos[node] = (
                    self.network.nodes[node]['x'],
                    self.network.nodes[node]['y']
                )
            else:
                # Use spring layout if coordinates not available
                self._network_pos = nx.spring_layout(self.network, seed=42)
                break
        
        # Set axis limits with minimal margins (just enough to see edge nodes)
        if self._network_pos:
            x_vals = [p[0] for p in self._network_pos.values()]
            y_vals = [p[1] for p in self._network_pos.values()]
            # Much smaller margins - just 3% of range plus tiny buffer for edge visibility
            x_margin = (max(x_vals) - min(x_vals)) * 0.03 + 0.5
            y_margin = (max(y_vals) - min(y_vals)) * 0.03 + 0.5
            self.ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
            self.ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)
        
        # Draw static background. If a scenario is attached, use the
        # scenario-aware display layer; otherwise keep the original network
        # fallback so standalone env usage still works.
        from accessibility_equity.visualization.video_display import (
            draw_scenario_background,
            initialize_video_display_artists,
        )

        if not draw_scenario_background(self):
            self._draw_static_network()
        
        # Initialize artists for agents (vehicles and humans)
        self._initialize_agent_artists()
        initialize_video_display_artists(self)
        
        self._artists_initialized = True
    
    def _draw_static_network(self):
        """Draw the static road network (nodes and edges) once."""
        from matplotlib.patches import Circle
        from matplotlib.lines import Line2D
        
        pos = self._network_pos
        
        # Draw edges as simple lines (bidirectional = two parallel lines)
        self._edge_artists = []
        for u, v in self.network.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            
            # Check if bidirectional
            is_bidirectional = self.network.has_edge(v, u)
            
            if is_bidirectional and u < v:
                # Two parallel lines for bidirectional
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    # Perpendicular unit vector
                    px = -dy / length * 0.15
                    py = dx / length * 0.15
                    
                    # First line (offset to one side)
                    line1 = Line2D([x1 + px, x2 + px], [y1 + py, y2 + py],
                                   color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                    self.ax.add_line(line1)
                    self._edge_artists.append(line1)
                    
                    # Second line (offset to other side)
                    line2 = Line2D([x2 - px, x1 - px], [y2 - py, y1 - py],
                                   color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                    self.ax.add_line(line2)
                    self._edge_artists.append(line2)
            elif not is_bidirectional:
                # Single line for unidirectional
                line = Line2D([x1, x2], [y1, y2],
                              color='gray', linewidth=1.5, alpha=0.6, zorder=1)
                self.ax.add_line(line)
                self._edge_artists.append(line)
        
        # Draw nodes
        self._node_artists = {}
        for node in self.network.nodes():
            x, y = pos[node]
            circle = Circle((x, y), radius=1.0, color='lightblue',
                           ec='black', linewidth=2, zorder=2)
            self.ax.add_patch(circle)
            self._node_artists[node] = circle
            
            # Draw node label
            self.ax.text(x, y, str(node), ha='center', va='center',
                        fontsize=10, fontweight='bold', zorder=3)
    
    def _initialize_agent_artists(self):
        """Initialize persistent artists for all agents."""
        from matplotlib.patches import Rectangle, Circle
        from matplotlib.lines import Line2D
        
        # Initialize vehicle artists
        self._vehicle_artists = {}
        for vehicle in self.vehicle_agents:
            capacity = self.agent_attributes.get(vehicle, {}).get('capacity', 4)
            vehicle_width = max(0.8, 0.4 + capacity * 0.25)
            vehicle_height = 0.3
            
            rect = Rectangle((0, 0), vehicle_width, vehicle_height,
                           color='cornflowerblue', ec='darkblue',
                           linewidth=1.5, zorder=4, alpha=0.7)
            rect.set_visible(False)  # Hidden until positioned
            self.ax.add_patch(rect)
            self._vehicle_artists[vehicle] = rect
        
        # Initialize human artists (walking)
        self._human_artists = {}
        for human in self.human_agents:
            circle = Circle((0, 0), radius=0.15, color='red',
                           ec='darkred', linewidth=1.5, zorder=5)
            circle.set_visible(False)  # Hidden until positioned
            self.ax.add_patch(circle)
            self._human_artists[human] = circle
        
        # Initialize passenger artists (humans aboard vehicles)
        self._passenger_artists = {}
        for vehicle in self.vehicle_agents:
            capacity = self.agent_attributes.get(vehicle, {}).get('capacity', 4)
            for i in range(capacity):
                # Create passenger circle (will be positioned inside vehicle)
                circle = Circle((0, 0), radius=0.10, color='red',
                               ec='darkred', linewidth=1, zorder=6)
                circle.set_visible(False)
                self.ax.add_patch(circle)
                self._passenger_artists[(vehicle, i)] = circle
        
        # Initialize destination arc artists for vehicles
        self._destination_artists = {}
        for vehicle in self.vehicle_agents:
            # Create empty line for destination arc
            line = Line2D([], [], color='cornflowerblue',
                         linestyle=':', linewidth=2, alpha=0.6, zorder=2)
            line.set_visible(False)
            self.ax.add_line(line)
            self._destination_artists[vehicle] = line
    
    def save_video(self, filename='transport_video.mp4', fps=20):
        """
        Save recorded frames as video or GIF.
        
        Uses imageio with ffmpeg for fast MP4 conversion, falls back to GIF using PIL if needed.
        
        Args:
            filename: Output filename (can be .mp4 or .gif)
            fps: Frames per second
        """
        if not self.frames:
            print("No frames recorded. Call start_video_recording() first.")
            return
        
        print(f"Saving {len(self.frames)} frames...")
        
        try:
            # Determine output format from filename
            is_gif = filename.lower().endswith('.gif')
            
            if not is_gif:
                # Try MP4 using imageio with ffmpeg (much faster than matplotlib)
                try:
                    import imageio
                    
                    # Save as MP4 using imageio with ffmpeg
                    writer = imageio.get_writer(
                        filename, 
                        fps=fps,
                        codec='libx264',
                        pixelformat='yuv420p',
                        quality=8  # 0-10, higher is better quality
                    )
                    
                    for frame in self.frames:
                        writer.append_data(frame)
                    
                    writer.close()
                    print(f"Video saved to {filename} ({len(self.frames)} frames)")
                    return
                    
                except Exception as e:
                    print(f"Could not save MP4 with imageio ({e}), trying GIF...")
                    # Fall through to GIF saving
                    filename = filename.replace('.mp4', '.gif')
            
            # Save as GIF using PIL
            from PIL import Image
            
            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(frame) for frame in self.frames]
            
            # Save as animated GIF
            duration_ms = int(1000 / fps)
            pil_frames[0].save(
                filename,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0
            )
            print(f"Video saved as GIF to {filename} ({len(self.frames)} frames)")
            
        except Exception as e:
            print(f"Error saving video: {e}")
        finally:
            # Always reset state
            self._recording = False
            self.frames = []
    
    def save_frame(self, filename='frame.png'):
        """Save current frame as PNG image"""
        if self.fig is not None:
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Frame saved to {filename}")

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        if self.fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(self.fig)
            except ImportError:
                pass
            self.fig = None
            self.ax = None

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Returns the observations for each agent.
        Positions and aboard status are initialized randomly unless an
        ``initial_state`` payload is supplied in ``options``.
        """
        if seed is not None:
            self.np_random, self.np_random_seed = seeding.np_random(seed)
        
        self.agents = self.possible_agents[:]
        
        # Initialize state components
        self.real_time = 0.0
        self._last_event_time = 0.0  # Track last arrival event time
        self._positions_at_last_event = {}  # Track positions at last event
        self._speeds_at_last_event = {}  # Track speeds at last event
        
        # Initialize empty dictionaries first (needed by initialize_random_positions)
        self.agent_positions = {}
        self.human_aboard = {}
        
        initial_state = {}
        if options:
            initial_state = options.get("initial_state", options)

        # Initialize agent positions and aboard status randomly, then apply any
        # scenario-provided overrides. This keeps reset usable without a
        # scenario while allowing wrappers to supply a fully prepared state.
        self.initialize_random_positions(seed=seed)
        initial_positions = initial_state.get("agent_positions", {})
        for agent, position in initial_positions.items():
            if agent in self.possible_agents:
                self.agent_positions[agent] = position

        initial_human_aboard = initial_state.get("human_aboard", {})
        for human, vehicle in initial_human_aboard.items():
            if human in self.human_agents:
                self.human_aboard[human] = vehicle
                if vehicle is not None and vehicle in self.agent_positions:
                    self.agent_positions[human] = self.agent_positions[vehicle]
        
        # Initialize vehicle destinations - all start with None
        self.vehicle_destinations = {agent: None for agent in self.vehicle_agents}
        self.vehicle_destinations.update({
            agent: destination
            for agent, destination in initial_state.get("vehicle_destinations", {}).items()
            if agent in self.vehicle_agents
        })

        self.human_destinations = {agent: None for agent in self.human_agents}
        self.human_destinations.update({
            agent: destination
            for agent, destination in initial_state.get("human_destinations", {}).items()
            if agent in self.human_agents
        })
        # clear termination participants for new episode (will be frozen lazily later)
        self.termination_participants = None

        # Initialize step type - start with routing
        self._step_type = 'routing'
        
        # Cache network data for observations (constant throughout episode)
        self._cached_network_nodes = list(self.network.nodes())
        self._cached_network_edges = [(u, v, dict(data)) for u, v, data in self.network.edges(data=True)]
        
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
            'human_destinations': dict(self.human_destinations),
            'agent_attributes': dict(self.agent_attributes),
            'network_nodes': self._cached_network_nodes,
            'network_edges': self._cached_network_edges,
            'action_mapping': self._get_action_mapping(agent),
            'my_position': self.agent_positions[agent],
        }
        return obs
    
    def _generate_local_observation(self, agent):
        """
        Local observation: agent observes only agents at same node or on same edge (ignoring coordinate on edge),
        along with their state components and attributes.
        """
        agent_pos = self.agent_positions[agent]
        
        # Determine location for comparison
        # If on edge, compare just the edge tuple (ignoring coordinate)
        if isinstance(agent_pos, tuple):
            agent_location = agent_pos[0]  # Just the edge tuple
        else:
            agent_location = agent_pos  # Node
        
        # Find agents at same location
        agents_at_location = []
        for other_agent in self.agents:
            other_pos = self.agent_positions[other_agent]
            
            # Determine other agent's location
            if isinstance(other_pos, tuple):
                other_location = other_pos[0]  # Just the edge tuple
            else:
                other_location = other_pos  # Node
            
            # Check if at same location
            if agent_location == other_location:
                agents_at_location.append(other_agent)
        
        # Build observation with info about agents at same location
        obs = {
            'real_time': float(self.real_time),
            'step_type': self.step_type,
            'my_position': agent_pos,
            'agents_here': {},
            'action_mapping': self._get_action_mapping(agent)
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
                agent_info['destination'] = self.human_destinations.get(other_agent)
            obs['agents_here'][other_agent] = agent_info
        
        return obs
    
    def _generate_statistical_observation(self, agent):
        """
        Statistical observation: as local, plus counts of humans and vehicles
        at every node and on every edge.
        """
        # Start with local observation (which includes action_mapping)
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
    
    def _get_action_mapping(self, agent):
        """
        Generate a mapping from action indices to their meanings for the given agent.
        This makes the action space transparent to the agent.
        
        Returns a dict with:
        - 'description': human-readable description of what each action does
        - 'details': specific IDs/objects that each action index refers to
        """
        if self.step_type is None or agent not in self.agents:
            return {'description': {0: 'pass'}, 'details': {0: None}}
        
        mapping = {'description': {}, 'details': {}}
        
        if self.step_type == 'routing':
            if agent in self.vehicle_agents:
                pos = self.agent_positions.get(agent)
                if pos is not None and not isinstance(pos, tuple):
                    # Vehicle at node can set destination
                    #TODO: limit to reachable nodes?
                    mapping['description'][0] = 'set_destination_none'
                    mapping['details'][0] = None
                    nodes = list(self.network.nodes())
                    for i, node in enumerate(nodes):
                        mapping['description'][i + 1] = f'set_destination_node'#_{node}
                        mapping['details'][i + 1] = node
                    return mapping
            # All other agents can only pass
            mapping['description'][0] = 'pass'
            mapping['details'][0] = None
            return mapping
        
        elif self.step_type == 'unboarding':
            if agent in self.human_agents:
                aboard = self.human_aboard.get(agent)

                if aboard is not None:
                    vehicle_pos = self.agent_positions.get(aboard)
                    if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
                        # Human aboard vehicle at node can unboard
                        mapping['description'][0] = 'pass'
                        mapping['details'][0] = None
                        mapping['description'][1] = 'unboard'
                        mapping['details'][1] = aboard  # Include vehicle ID being unboarded from

                        return mapping
            # All other agents can only pass
            mapping['description'][0] = 'pass'
            mapping['details'][0] = None
            return mapping
        
        elif self.step_type == 'boarding':
            if agent in self.human_agents:
                pos = self.agent_positions.get(agent)
                aboard = self.human_aboard.get(agent)
                if pos is not None and not isinstance(pos, tuple) and aboard is None:
                    # Human at node can board vehicles at same node
                    vehicles_at_node = [
                        v for v in self.vehicle_agents 
                        if not isinstance(self.agent_positions.get(v), tuple) and self.agent_positions.get(v) == pos
                    ]
                    mapping['description'][0] = 'pass'
                    mapping['details'][0] = None
                    for i, vehicle_id in enumerate(vehicles_at_node):
                        mapping['description'][i + 1] = 'board_vehicle'
                        mapping['details'][i + 1] = vehicle_id
                    return mapping
            # All other agents can only pass
            mapping['description'][0] = 'pass'
            mapping['details'][0] = None
            return mapping
        
        elif self.step_type == 'departing':
            pos = self.agent_positions.get(agent)
            if pos is not None and not isinstance(pos, tuple):
                if agent in self.vehicle_agents:
                    # Vehicle at node can depart into outgoing edges
                    outgoing_edges = list(self.network.out_edges(pos))
                    mapping['description'][0] = 'pass'
                    mapping['details'][0] = None
                    for i, edge in enumerate(outgoing_edges):
                        mapping['description'][i + 1] = 'depart_edge'
                        mapping['details'][i + 1] = edge
                    return mapping
                elif agent in self.human_agents:
                    aboard = self.human_aboard.get(agent)
                    if aboard is None:
                        # Human at node (not aboard) can walk into outgoing edges
                        outgoing_edges = list(self.network.out_edges(pos))
                        mapping['description'][0] = 'pass'
                        mapping['details'][0] = None
                        for i, edge in enumerate(outgoing_edges):
                            mapping['description'][i + 1] = 'walk_edge'
                            mapping['details'][i + 1] = edge
                        return mapping
            # All other agents can only pass
            mapping['description'][0] = 'pass'
            mapping['details'][0] = None
            return mapping
        
        # Default fallback
        mapping['description'][0] = 'pass'
        mapping['details'][0] = None
        return mapping


    #  Termination logic
    def freeze_termination_participants(self):
        """Freeze termination participants NOW based on current env destinations."""
        participants = set()
        for agent in self.agents:
            if agent in self.vehicle_agents:
                dest = self.vehicle_destinations.get(agent)
            elif agent in self.human_agents:
                dest = self.human_destinations.get(agent)
            else:
                dest = None

            if dest is not None:
                participants.add(agent)

        self.termination_participants = participants

    def termination_status(self):
        """
        Per-agent termination dict: agent -> True/False.
          False : not participating, or participating but not reached yet
          True  : participating and reached
        """
        mode = getattr(self, "termination_mode", "initial_active") # Default to "initial_active" if termination_mode attribute doesn't exist

        if mode == "initial_active":
            if self.termination_participants is None:
                self.freeze_termination_participants()
            participants = self.termination_participants or set()

        elif mode == "current_active":
            #whose dest != None currently
            participants = set()
            for agent in self.agents:
                if agent in self.vehicle_agents:
                    dest = self.vehicle_destinations.get(agent)
                elif agent in self.human_agents:
                    dest = self.human_destinations.get(agent)
                else:
                    dest = None
                if dest is not None:
                    participants.add(agent)

        elif mode == "all_agents":
            participants = set(self.agents)

        else:
            raise ValueError(f"Unknown termination_mode: {mode}")


        status = {}
        for agent in self.agents:
            if agent not in participants:
                status[agent] = False
                continue

            # current dest (may change), but participant still must reach a dest
            if agent in self.vehicle_agents:
                dest = self.vehicle_destinations.get(agent)
            elif agent in self.human_agents:
                dest = self.human_destinations.get(agent)
            else:
                dest = None

            if dest is None:
                status[agent] = False
                continue

            pos = self.agent_positions.get(agent)
            if hasattr(pos, "item") and not isinstance(pos, tuple):
                pos = pos.item()

            # on edge => not reached
            if isinstance(pos, tuple):
                status[agent] = False
                continue

            # normalize dest to set
            if isinstance(dest, set):
                dest_set = dest
            elif isinstance(dest, (list, tuple)):
                dest_set = set(dest)
            else:
                if hasattr(dest, "item"):
                    dest = dest.item()
                dest_set = {dest}

            status[agent] = (pos in dest_set)

        return status

    def _termination_participation(self):
        """Return whether each active agent participates in termination checks."""
        mode = getattr(self, "termination_mode", "initial_active")

        if mode == "initial_active":
            if self.termination_participants is None:
                self.freeze_termination_participants()
            participants = self.termination_participants or set()
        elif mode == "current_active":
            participants = set()
            for agent in self.agents:
                if agent in self.vehicle_agents:
                    dest = self.vehicle_destinations.get(agent)
                elif agent in self.human_agents:
                    dest = self.human_destinations.get(agent)
                else:
                    dest = None
                if dest is not None:
                    participants.add(agent)
        elif mode == "all_agents":
            participants = set(self.agents)
        else:
            raise ValueError(f"Unknown termination_mode: {mode}")

        return {agent: agent in participants for agent in self.agents}



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


        # Automatically cycle to next step type AFTER processing current step
        step_cycle = ['routing', 'unboarding', 'boarding', 'departing']
        current_idx = step_cycle.index(self._step_type)
        self._step_type = step_cycle[(current_idx + 1) % len(step_cycle)]
        
        step_agents = list(self.agents)

        # All rewards are constantly zero
        rewards = {agent: 0.0 for agent in step_agents}

        # Track destination reach status for diagnostics. Fixed-horizon
        # episodes do not terminate individual agents at the env layer.
        destination_reached = self.termination_status()
        participation = self._termination_participation()

        terminations = {agent: False for agent in step_agents}
        truncations = {agent: False for agent in step_agents}
        infos = {
            agent: {
                "termination_participant": participation.get(agent, False),
                "destination_reached": destination_reached.get(agent, False),
            }
            for agent in step_agents
        }

        # Keep agents active after reaching destinations. Higher-level wrappers
        # use max_steps to decide episode boundaries.
        observations = self._generate_observations()

        # Auto-render only if render_mode is human and not currently recording
        # When recording, we want explicit control over when to capture frames
        if self.render_mode == "human" and not getattr(self, '_recording', False):
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
                    vehicles_at_node = []
                    for v in self.vehicle_agents:
                        v_pos = self.agent_positions.get(v)
                        # Compare positions carefully (both should be nodes)
                        if v_pos is not None and not isinstance(v_pos, tuple) and v_pos == pos:
                            vehicles_at_node.append(v)
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
        Length is measured in km and speed in km/h, so movement durations are
        measured in hours. Humans use their own walking speed attribute.
        Vehicles use the slower value between their maximum speed and the
        current edge speed.
        """
        if agent in self.vehicle_agents:
            vehicle_speed = float(self.agent_attributes[agent]['speed'])
            edge_speed = float(edge_data['speed'])
            return min(vehicle_speed, edge_speed)
        elif agent in self.human_agents:
            return self.agent_attributes[agent]['speed']
        else:
            return 1.0  # Fallback
    
    def _process_departing_actions(self, actions):
        """
        Process departing step: vehicles and humans (not aboard) at nodes can depart/walk into edges.
        All agents on edges move. Real time advances by minimum remaining duration on edges.
        """
        # Process departing actions for agents at nodes
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
            
            # IMPORTANT: Save state BEFORE movement for video rendering
            # When render() is called after this step, it needs to know where agents
            # were at the START of this time period to interpolate correctly
            if getattr(self, '_recording', False):
                self._positions_at_last_event = self.agent_positions.copy()
                self._speeds_at_last_event = {}
                self._last_event_time = self.real_time
                
                # Compute speeds for agents on edges
                for agent in self.agents:
                    pos = self.agent_positions.get(agent)
                    if isinstance(pos, tuple):
                        edge, coord = pos
                        edge_data = self.network[edge[0]][edge[1]]
                        self._speeds_at_last_event[agent] = self._get_agent_speed(agent, edge_data)
                    else:
                        self._speeds_at_last_event[agent] = 0.0
            
            # Move all agents on edges
            for agent in self.agents:
                pos = self.agent_positions.get(agent)
                
                if pos is not None and isinstance(pos, tuple):
                    edge, coord = pos
                    edge_data = self.network[edge[0]][edge[1]]
                    speed = self._get_agent_speed(agent, edge_data)
                    new_coord = coord + speed * delta_t
                    edge_length = edge_data['length']
                    
                    if abs(new_coord - edge_length) < self.FLOAT_EPSILON:
                        # Reached target node
                        self.agent_positions[agent] = edge[1]
                    else:
                        # Still on edge
                        self.agent_positions[agent] = (edge, new_coord)
            
            # Update humans aboard to match vehicle positions
            for human in self.human_agents:
                aboard = self.human_aboard.get(human)
                if aboard is not None:
                    self.agent_positions[human] = self.agent_positions[aboard]
            
            # Advance real time
            self.real_time += delta_t
        else:
            # No movement - just update humans aboard
            for human in self.human_agents:
                aboard = self.human_aboard.get(human)
                if aboard is not None:
                    self.agent_positions[human] = self.agent_positions[aboard]
