import functools
from typing import Optional, Union, Tuple, Dict, Any

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Dict as DictSpace, Discrete, Tuple as TupleSpace
from gymnasium.utils import seeding
import networkx as nx

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


def env(render_mode=None, num_humans=2, num_vehicles=1, network=None):
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
        network=network
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


def raw_env(render_mode=None, num_humans=2, num_vehicles=1, network=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(
        render_mode=render_mode,
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        network=network
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
        
        # State components (will be initialized in reset)
        self.real_time = None
        self.agent_positions = None
        self.vehicle_destinations = None

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
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Placeholder action space - will be refined based on actual action structure
        return Discrete(5, seed=self.np_random_seed)

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
            print(f"Current state: real_time={self.real_time:.2f}")
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
