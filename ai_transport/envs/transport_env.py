import functools
from typing import Optional, Union, Tuple, Dict, Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, Tuple as TupleSpace
from gymnasium.utils import seeding
import networkx as nx

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None, num_humans=2, num_vehicles=1, network=None,
        human_speed=1.0, vehicle_speed=2.0, vehicle_capacity=4, vehicle_fuel_use=1.0):
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
        vehicle_fuel_use=vehicle_fuel_use
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
            human_speed=1.0, vehicle_speed=2.0, vehicle_capacity=4, vehicle_fuel_use=1.0):
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
        vehicle_fuel_use=vehicle_fuel_use
    )
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "transport_v0"}

    def __init__(
        self,
        render_mode=None,
        num_humans=2,
        num_vehicles=1,
        network=None,
        human_speed=1.0,
        vehicle_speed=2.0,
        vehicle_capacity=4,
        vehicle_fuel_use=1.0
    ):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        
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
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # For now, returning a placeholder space - will be refined based on actual observation structure
        return Box(low=0, high=1, shape=(10,), dtype=np.float32)

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
        
        # Create observations (placeholder for now)
        observations = {agent: np.zeros(10, dtype=np.float32) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        
        NOTE: Step logic is not yet implemented as per requirements.
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Placeholder step logic - to be implemented later
        observations = {agent: np.zeros(10, dtype=np.float32) for agent in self.agents}
        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if self.render_mode == "human":
            self.render()
            
        return observations, rewards, terminations, truncations, infos
