"""AI Transport Environment - PettingZoo Parallel API Implementation."""

from typing import Dict, Any, Optional
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


class AITransportParallelEnv(ParallelEnv):
    """
    Parallel environment for AI-governed public transport simulation.
    
    This environment simulates a public transport network where multiple AI agents
    control different vehicles (buses, trains, etc.) to efficiently transport passengers.
    
    Agents: Multiple transport vehicles (e.g., "bus_0", "bus_1", "train_0")
    Actions: Discrete actions for each vehicle (move to next stop, wait, skip stop)
    Observations: Current location, passenger count, waiting passengers at stops
    Rewards: Based on passenger satisfaction, efficiency, and coordination
    """
    
    metadata = {
        "name": "ai_transport_v0",
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
    }
    
    def __init__(
        self,
        num_buses: int = 3,
        num_stops: int = 10,
        max_passengers_per_vehicle: int = 50,
        max_passengers_per_stop: int = 20,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the AI Transport environment.
        
        Args:
            num_buses: Number of bus agents in the environment
            num_stops: Number of stops in the transport network
            max_passengers_per_vehicle: Maximum passenger capacity per vehicle
            max_passengers_per_stop: Maximum passengers that can wait at a stop
            render_mode: Mode for rendering ("human" or "rgb_array")
        """
        super().__init__()
        
        self.num_buses = num_buses
        self.num_stops = num_stops
        self.max_passengers_per_vehicle = max_passengers_per_vehicle
        self.max_passengers_per_stop = max_passengers_per_stop
        self.render_mode = render_mode
        
        # Define possible agents
        self.possible_agents = [f"bus_{i}" for i in range(num_buses)]
        
        # Initialize agent state
        self.agents = self.possible_agents[:]
        
        # Define action and observation spaces
        self._action_spaces = {
            agent: spaces.Discrete(3)  # 0: move to next stop, 1: wait, 2: skip stop
            for agent in self.possible_agents
        }
        
        # Observation: [current_stop, passengers_on_vehicle, passengers_at_current_stop, 
        #               passengers_at_next_stop, time_step]
        self._observation_spaces = {
            agent: spaces.Box(
                low=0,
                high=max(num_stops, max_passengers_per_vehicle, max_passengers_per_stop, 1000),
                shape=(5,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        
        # Initialize state variables
        self.timestep = 0
        self.max_timesteps = 200
        
    @property
    def observation_spaces(self) -> Dict[str, spaces.Space]:
        """Return observation spaces for all agents."""
        return self._observation_spaces
    
    @property
    def action_spaces(self) -> Dict[str, spaces.Space]:
        """Return action spaces for all agents."""
        return self._action_spaces
    
    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for a specific agent."""
        return self._observation_spaces[agent]
    
    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for a specific agent."""
        return self._action_spaces[agent]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observations: Dictionary of observations for each agent
            infos: Dictionary of info for each agent
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.timestep = 0
        
        # Initialize bus positions (evenly distributed across stops)
        self.bus_positions = {
            agent: i * (self.num_stops // self.num_buses) % self.num_stops
            for i, agent in enumerate(self.agents)
        }
        
        # Initialize passengers on each bus
        self.bus_passengers = {agent: 0 for agent in self.agents}
        
        # Initialize passengers waiting at each stop
        self.stop_passengers = np.random.randint(0, 5, size=self.num_stops)
        
        # Generate initial observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        
        return observations, infos
    
    def step(self, actions: Dict[str, int]):
        """
        Execute one step of the environment.
        
        Args:
            actions: Dictionary mapping agent names to their actions
            
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            terminations: Dictionary of termination flags for each agent
            truncations: Dictionary of truncation flags for each agent
            infos: Dictionary of info for each agent
        """
        self.timestep += 1
        
        rewards = {agent: 0.0 for agent in self.agents}
        
        # Process each agent's action
        for agent in self.agents:
            if agent not in actions:
                continue
                
            action = actions[agent]
            current_pos = self.bus_positions[agent]
            
            if action == 0:  # Move to next stop
                # Pick up passengers at current stop
                passengers_to_pick = min(
                    self.stop_passengers[current_pos],
                    self.max_passengers_per_vehicle - self.bus_passengers[agent]
                )
                self.bus_passengers[agent] += passengers_to_pick
                self.stop_passengers[current_pos] -= passengers_to_pick
                rewards[agent] += passengers_to_pick * 0.5  # Reward for picking up passengers
                
                # Drop off some passengers (simplified: random portion)
                if self.bus_passengers[agent] > 0:
                    passengers_to_drop = np.random.randint(0, self.bus_passengers[agent] + 1)
                    self.bus_passengers[agent] -= passengers_to_drop
                    rewards[agent] += passengers_to_drop * 1.0  # Higher reward for delivering passengers
                
                # Move to next stop
                self.bus_positions[agent] = (current_pos + 1) % self.num_stops
                
            elif action == 1:  # Wait at current stop
                # Pick up more passengers while waiting
                passengers_to_pick = min(
                    self.stop_passengers[current_pos] // 2,  # Pick up half
                    self.max_passengers_per_vehicle - self.bus_passengers[agent]
                )
                self.bus_passengers[agent] += passengers_to_pick
                self.stop_passengers[current_pos] -= passengers_to_pick
                rewards[agent] += passengers_to_pick * 0.3  # Lower reward for waiting
                rewards[agent] -= 0.1  # Small penalty for not moving
                
            elif action == 2:  # Skip stop (express route)
                # Move two stops ahead
                self.bus_positions[agent] = (current_pos + 2) % self.num_stops
                rewards[agent] -= 0.2  # Penalty for skipping stops
        
        # Add new passengers to stops
        for i in range(self.num_stops):
            new_passengers = np.random.poisson(0.5)  # Average 0.5 passengers per stop per timestep
            self.stop_passengers[i] = min(
                self.stop_passengers[i] + new_passengers,
                self.max_passengers_per_stop
            )
        
        # Penalty for overcrowded stops
        for agent in self.agents:
            crowded_stops = np.sum(self.stop_passengers > 15)
            rewards[agent] -= crowded_stops * 0.1
        
        # Get new observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        
        # Check for termination
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_timesteps for agent in self.agents}
        
        infos = {agent: {} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos
    
    def _get_observation(self, agent: str) -> np.ndarray:
        """
        Get observation for a specific agent.
        
        Args:
            agent: Name of the agent
            
        Returns:
            Observation array for the agent
        """
        current_pos = self.bus_positions[agent]
        next_pos = (current_pos + 1) % self.num_stops
        
        return np.array([
            current_pos,
            self.bus_passengers[agent],
            self.stop_passengers[current_pos],
            self.stop_passengers[next_pos],
            self.timestep,
        ], dtype=np.float32)
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"\n=== Timestep {self.timestep} ===")
            for agent in self.agents:
                print(f"{agent}: Position {self.bus_positions[agent]}, "
                      f"Passengers {self.bus_passengers[agent]}")
            print(f"Stop passengers: {self.stop_passengers}")
    
    def close(self):
        """Close the environment."""
        pass


def parallel_env(**kwargs) -> AITransportParallelEnv:
    """
    Create a parallel environment instance.
    
    Args:
        **kwargs: Arguments to pass to AITransportParallelEnv
        
    Returns:
        AITransportParallelEnv instance
    """
    return AITransportParallelEnv(**kwargs)


def raw_env(**kwargs) -> AITransportParallelEnv:
    """
    Create a raw parallel environment instance.
    
    Args:
        **kwargs: Arguments to pass to AITransportParallelEnv
        
    Returns:
        AITransportParallelEnv instance
    """
    return AITransportParallelEnv(**kwargs)


def env(**kwargs):
    """
    Create an AEC environment from the parallel environment.
    
    Args:
        **kwargs: Arguments to pass to AITransportParallelEnv
        
    Returns:
        AEC wrapped environment
    """
    parallel_environment = parallel_env(**kwargs)
    aec_env = parallel_to_aec(parallel_environment)
    return aec_env
