"""
Gym-Style Wrapper for AI Transport Environment.

This module provides a wrapper around the ai_transport PettingZoo environment
that makes it compatible with single-agent RL training frameworks like stable-baselines3.

Key Features:
1. Single-agent interface: treats all vehicles as controlled by one policy
2. Human agents follow a fixed policy (simulated internally)
3. Custom reward function support
4. Compatible with gym/gymnasium interface
5. Action masking for handling invalid actions across step types

Usage:
    >>> from ai_transport.wrappers import TransportGymWrapper
    >>> from ai_transport.policies import HeuristicRoutingHumanPolicy
    >>> 
    >>> # Create wrapper with heuristic human policy
    >>> env = TransportGymWrapper(
    ...     num_humans=4,
    ...     num_vehicles=2,
    ...     human_policy_class=HeuristicRoutingHumanPolicy,
    ...     human_policy_kwargs={'p_wait': 0.5}
    ... )
    >>> obs, info = env.reset(seed=42)
    >>> action = env.action_space.sample()
    >>> obs, reward, done, truncated, info = env.step(action)
"""

from typing import List, Dict, Tuple, Any, Optional, Callable, Type
import numpy as np
import networkx as nx
import gymnasium as gym
from gymnasium import spaces

from ai_transport import parallel_env as TransportParallelEnv
from ai_transport.policies.human_policies import HumanPolicy


class TransportGymWrapper(gym.Env):
    """
    Single-agent gym wrapper for transport environment.
    
    This wrapper treats all vehicles as controlled by a single learned policy,
    while human agents follow a fixed policy that is simulated internally.
    
    The action space is a MultiDiscrete space where each vehicle can take
    one of several actions depending on the current step type.
    
    Attributes:
        env: The underlying ai_transport parallel environment
        num_vehicles: Number of vehicle agents
        num_humans: Number of human agents
        human_policy_class: Class for human agent policies
        human_policies: Dictionary of human policy instances
        reward_function: Custom reward function
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(
        self,
        num_humans: int = 4,
        num_vehicles: int = 2,
        network: Optional[nx.DiGraph] = None,
        human_speeds: Optional[List[float]] = None,
        vehicle_speeds: Optional[List[float]] = None,
        vehicle_capacities: Optional[List[int]] = None,
        vehicle_fuel_uses: Optional[List[float]] = None,
        observation_scenario: str = 'full',
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        human_policy_class: Optional[Type[HumanPolicy]] = None,
        human_policy_kwargs: Optional[Dict] = None,
        reward_function: Optional[Callable] = None,
        human_goal_nodes: Optional[List[int]] = None,
    ):
        """
        Initialize the transport gym wrapper.
        
        Args:
            num_humans: Number of human (passenger) agents
            num_vehicles: Number of vehicle agents
            network: NetworkX DiGraph for the road network (generated if None)
            human_speeds: Walking speed for each human
            vehicle_speeds: Speed for each vehicle
            vehicle_capacities: Capacity for each vehicle
            vehicle_fuel_uses: Fuel consumption for each vehicle
            observation_scenario: One of 'full', 'local', or 'statistical'
            render_mode: 'human' for rendering, None for no rendering
            max_steps: Maximum steps per episode
            human_policy_class: Policy class for human agents (if None, uses random)
            human_policy_kwargs: Keyword arguments for human policy initialization
            reward_function: Custom reward function taking (wrapper, obs_dict, actions_dict) -> Dict[str, float]
            human_goal_nodes: List of goal nodes for each human (cycling through on goal achievement)
        """
        super().__init__()
        
        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.human_goal_nodes = human_goal_nodes or []
        
        # Create underlying environment
        self.env = TransportParallelEnv(
            num_humans=num_humans,
            num_vehicles=num_vehicles,
            network=network,
            human_speeds=human_speeds,
            vehicle_speeds=vehicle_speeds,
            vehicle_capacities=vehicle_capacities,
            vehicle_fuel_uses=vehicle_fuel_uses,
            observation_scenario=observation_scenario,
            render_mode=render_mode,
        )
        
        # Store vehicle and human agent IDs
        self.vehicle_agents = list(self.env.vehicle_agents)
        self.human_agents = list(self.env.human_agents)
        
        # Initialize human policies
        self.human_policy_class = human_policy_class
        self.human_policy_kwargs = human_policy_kwargs or {}
        self.human_policies: Dict[str, Optional[HumanPolicy]] = {}
        self.human_current_goals: Dict[str, Optional[int]] = {}
        
        # Set custom reward function (defaults to zero rewards)
        self.reward_function = reward_function or self._default_reward_function
        
        # Step counter
        self.step_count = 0
        
        # Define action and observation spaces
        # Action space: One action per vehicle (simplified fixed action space)
        # 0 = pass, 1-20 = routing destinations, 21-30 = depart edges, 31 = unboard (N/A for vehicles)
        self.action_space = spaces.MultiDiscrete([42] * num_vehicles)
        
        # Observation space: Dict with various information
        # This is a placeholder - in practice, you'd flatten this for neural networks
        self.observation_space = spaces.Dict({
            'step_type': spaces.Discrete(4),  # routing, unboarding, boarding, departing
            'real_time': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'vehicle_positions': spaces.Box(low=-np.inf, high=np.inf, shape=(num_vehicles, 2), dtype=np.float32),
            'human_positions': spaces.Box(low=-np.inf, high=np.inf, shape=(num_humans, 2), dtype=np.float32),
        })
    
    def _init_human_policies(self):
        """Initialize human policies after reset (when network is available)."""
        if self.human_policy_class is None:
            # No policy specified, humans will take random actions
            for agent in self.human_agents:
                self.human_policies[agent] = None
                self.human_current_goals[agent] = None
        else:
            # Initialize policy for each human
            for i, agent in enumerate(self.human_agents):
                # Set initial goal for this human
                if self.human_goal_nodes and i < len(self.human_goal_nodes):
                    initial_goal = {self.human_goal_nodes[i]}
                else:
                    # Pick a random node as goal
                    nodes = list(self.env.network.nodes())
                    initial_goal = {self.np_random.choice(nodes)} if nodes else {0}
                
                self.human_current_goals[agent] = list(initial_goal)[0]
                
                policy_kwargs = dict(self.human_policy_kwargs)
                policy_kwargs['agent_id'] = agent
                policy_kwargs['network'] = self.env.network
                policy_kwargs['target_nodes'] = initial_goal
                
                if 'seed' not in policy_kwargs:
                    policy_kwargs['seed'] = self.np_random.integers(0, 2**31-1)
                
                self.human_policies[agent] = self.human_policy_class(**policy_kwargs)
    
    def _update_human_goals(self, obs_dict: Dict):
        """Check if humans reached their goals and update to next goal."""
        if not self.human_goal_nodes:
            return
        
        for i, agent in enumerate(self.human_agents):
            if i >= len(self.human_goal_nodes):
                continue
            
            pos = self.env.agent_positions.get(agent)
            current_goal = self.human_current_goals.get(agent)
            
            # Check if at goal node
            if pos is not None and not isinstance(pos, tuple) and pos == current_goal:
                # Reached goal! Pick next goal
                nodes = list(self.env.network.nodes())
                if nodes:
                    # Pick a different random node
                    available_nodes = [n for n in nodes if n != pos]
                    if available_nodes:
                        new_goal = self.np_random.choice(available_nodes)
                    else:
                        new_goal = self.np_random.choice(nodes)
                    
                    self.human_current_goals[agent] = new_goal
                    
                    # Update policy with new goal
                    if self.human_policies[agent] is not None:
                        self.human_policies[agent].reset()
                        # If policy supports target_nodes, update it
                        if hasattr(self.human_policies[agent], 'target_nodes'):
                            self.human_policies[agent].target_nodes = {new_goal}
    
    def _get_human_actions(self, obs_dict: Dict) -> Dict[str, int]:
        """Get actions for all human agents using their policies."""
        human_actions = {}
        
        for agent in self.human_agents:
            if agent not in obs_dict:
                human_actions[agent] = 0
                continue
            
            policy = self.human_policies.get(agent)
            if policy is None:
                # No policy, take random action
                action_space = self.env.action_space(agent)
                human_actions[agent] = action_space.sample()
            else:
                # Use policy to get action
                obs = obs_dict[agent]
                human_actions[agent] = policy.get_action(obs, self.env)
        
        return human_actions
    
    def _default_reward_function(self, obs_dict: Dict, actions_dict: Dict) -> Dict[str, float]:
        """Default reward function (all zeros)."""
        return {agent: 0.0 for agent in self.vehicle_agents}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        self.step_count = 0
        obs_dict, info_dict = self.env.reset(seed=seed, options=options)
        
        # Initialize human policies now that network is available
        self._init_human_policies()
        
        # Convert to single-agent observation
        obs = self._get_observation(obs_dict)
        info = {'step_type': self.env.step_type, 'real_time': self.env.real_time}
        
        return obs, info
    
    def step(self, vehicle_actions: np.ndarray):
        """
        Take a step with actions for vehicles only.
        
        Args:
            vehicle_actions: Array of actions for each vehicle
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        
        # Get current observations
        obs_dict = self.env._generate_observations()
        
        # Update human goals if they reached their targets
        self._update_human_goals(obs_dict)
        
        # Get human actions from policies
        human_actions = self._get_human_actions(obs_dict)
        
        # Combine vehicle and human actions
        actions_dict = dict(human_actions)
        for i, agent in enumerate(self.vehicle_agents):
            if i < len(vehicle_actions):
                actions_dict[agent] = int(vehicle_actions[i])
            else:
                actions_dict[agent] = 0
        
        # Step the environment
        obs_dict, rewards_dict, terms_dict, truncs_dict, info_dict = self.env.step(actions_dict)
        
        # Compute reward using custom reward function
        vehicle_rewards = self.reward_function(obs_dict, actions_dict)
        
        # Aggregate reward (sum over vehicles)
        reward = sum(vehicle_rewards.get(agent, 0.0) for agent in self.vehicle_agents)
        
        # Episode termination
        terminated = any(terms_dict.values()) or self.step_count >= self.max_steps
        truncated = any(truncs_dict.values())
        
        # Build observation and info
        obs = self._get_observation(obs_dict)
        info = {
            'step_type': self.env.step_type,
            'real_time': self.env.real_time,
            'step_count': self.step_count,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self, obs_dict: Dict) -> Dict:
        """Convert multi-agent observations to single-agent observation.
        
        Returns observation matching the defined observation_space with keys:
        - step_type: int (0-3 for routing, unboarding, boarding, departing)
        - real_time: array with current simulation time
        - vehicle_positions: array of vehicle positions
        - human_positions: array of human positions
        """
        # Map step type to index
        step_types = {'routing': 0, 'unboarding': 1, 'boarding': 2, 'departing': 3}
        step_type_idx = step_types.get(self.env.step_type, 0)
        
        # Extract positions for vehicles and humans
        vehicle_positions = []
        for vehicle in self.vehicle_agents:
            pos = self.env.agent_positions.get(vehicle)
            if pos is None:
                vehicle_positions.append([0.0, 0.0])
            elif isinstance(pos, tuple):
                # On edge - use approximate position
                edge, coord = pos
                # For simplicity, just use node coordinates
                if edge and len(edge) >= 2:
                    node = edge[0]
                    node_data = self.env.network.nodes.get(node, {})
                    x = float(node_data.get('x', 0.0))
                    y = float(node_data.get('y', 0.0))
                    vehicle_positions.append([x, y])
                else:
                    vehicle_positions.append([0.0, 0.0])
            else:
                # At node
                node_data = self.env.network.nodes.get(pos, {})
                x = float(node_data.get('x', 0.0))
                y = float(node_data.get('y', 0.0))
                vehicle_positions.append([x, y])
        
        human_positions = []
        for human in self.human_agents:
            pos = self.env.agent_positions.get(human)
            if pos is None:
                human_positions.append([0.0, 0.0])
            elif isinstance(pos, tuple):
                # On edge
                edge, coord = pos
                if edge and len(edge) >= 2:
                    node = edge[0]
                    node_data = self.env.network.nodes.get(node, {})
                    x = float(node_data.get('x', 0.0))
                    y = float(node_data.get('y', 0.0))
                    human_positions.append([x, y])
                else:
                    human_positions.append([0.0, 0.0])
            else:
                # At node
                node_data = self.env.network.nodes.get(pos, {})
                x = float(node_data.get('x', 0.0))
                y = float(node_data.get('y', 0.0))
                human_positions.append([x, y])
        
        # Pad or truncate to match observation space dimensions
        while len(vehicle_positions) < self.num_vehicles:
            vehicle_positions.append([0.0, 0.0])
        vehicle_positions = vehicle_positions[:self.num_vehicles]
        
        while len(human_positions) < self.num_humans:
            human_positions.append([0.0, 0.0])
        human_positions = human_positions[:self.num_humans]
        
        obs = {
            'step_type': step_type_idx,
            'real_time': np.array([self.env.real_time], dtype=np.float32),
            'vehicle_positions': np.array(vehicle_positions, dtype=np.float32),
            'human_positions': np.array(human_positions, dtype=np.float32),
        }
        
        return obs
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()


def create_transport_env(
    num_humans: int = 4,
    num_vehicles: int = 2,
    num_nodes: int = 12,
    seed: Optional[int] = None,
    **kwargs
) -> TransportGymWrapper:
    """
    Convenience function to create a transport environment with random network.
    
    Args:
        num_humans: Number of human agents
        num_vehicles: Number of vehicle agents
        num_nodes: Number of nodes in random network
        seed: Random seed
        **kwargs: Additional arguments for TransportGymWrapper
        
    Returns:
        TransportGymWrapper with random network
    """
    # Create temporary env to generate network
    temp_env = TransportParallelEnv(num_humans=1, num_vehicles=1)
    network = temp_env.create_random_2d_network(num_nodes=num_nodes, seed=seed)
    temp_env.close()
    
    # Create wrapper with the network
    wrapper = TransportGymWrapper(
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        network=network,
        **kwargs
    )
    
    return wrapper
