"""
Gym-Style Wrapper for AI Transport Accessibility Equity.

This implementation stays intentionally close to
`ai_transport.wrappers.gym_wrapper`. The project-specific version keeps the same
single-agent idea: one RL agent controls all vehicles, while humans are treated
as part of the environment.

At this stage the main project-specific extension is the default reward:
it now uses the route-based accessibility and equity utility defined in the
local reward modules.
"""

import inspect
from typing import List, Dict, Any, Optional, Callable, Type, Sequence

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from accessibility_equity.config import (
    DEFAULT_EXPERIMENT_CONFIG,
    MobilityConfig,
    RewardConfig,
)
from accessibility_equity.envs.transport_env import parallel_env as TransportParallelEnv
from accessibility_equity.policies.human_policies import HumanPolicy
from accessibility_equity.scenarios import TransportScenario, build_transport_scenario
from accessibility_equity.rewards.equity_reward import (
    compute_reward,
    compute_scenario_utility_bounds,
)


_DEFAULT_CONFIG = DEFAULT_EXPERIMENT_CONFIG


class TransportGymWrapper(gym.Env):
    """
    Single-agent gym wrapper for the accessibility-equity transport environment.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        num_humans: int = _DEFAULT_CONFIG.scenario.num_humans,
        num_vehicles: int = _DEFAULT_CONFIG.scenario.num_vehicles,
        network: Optional[nx.DiGraph] = None,
        scenario: Optional[TransportScenario] = None,
        num_nodes: int = _DEFAULT_CONFIG.scenario.num_nodes,
        scenario_seed: Optional[int] = None,
        human_speeds: Optional[List[float]] = None,
        vehicle_speeds: Optional[List[float]] = None,
        vehicle_capacities: Optional[List[int]] = None,
        vehicle_fuel_uses: Optional[List[float]] = None,
        observation_scenario: str = "full",
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        human_policy_class: Optional[Type[HumanPolicy]] = None,
        human_policy_kwargs: Optional[Dict] = None,
        reward_function: Optional[Callable] = None,
        human_goal_nodes: Optional[List[int]] = None,
        randomize_initial_state: bool = False,
        network_kwargs: Optional[Dict[str, Any]] = None,
        poi_kwargs: Optional[Dict[str, Any]] = None,
        human_distribution_kwargs: Optional[Dict[str, Any]] = None,
        vehicle_distribution_kwargs: Optional[Dict[str, Any]] = None,
        reward_config: Optional[RewardConfig] = None,
        mobility_config: Optional[MobilityConfig] = None,
    ):
        super().__init__()

        self.reward_config = reward_config or _DEFAULT_CONFIG.reward
        self.mobility_config = mobility_config or _DEFAULT_CONFIG.mobility
        if human_speeds is None:
            human_speeds = [self.mobility_config.human_walking_speed_kmh] * num_humans
        if vehicle_speeds is None:
            vehicle_speeds = [self.mobility_config.vehicle_speed_kmh] * num_vehicles
        network_options = dict(network_kwargs or {})
        network_options.setdefault("speed_mean", self.mobility_config.edge_speed_kmh)
        poi_options = dict(poi_kwargs or {})
        poi_options.setdefault("central_count", _DEFAULT_CONFIG.scenario.central_count)

        self.num_humans = num_humans
        self.num_vehicles = num_vehicles
        self.max_steps = max_steps
        self.render_mode = render_mode
        effective_seed = _DEFAULT_CONFIG.scenario.seed if scenario_seed is None else scenario_seed

        if scenario is None:
            scenario = build_transport_scenario(
                num_humans=num_humans,
                num_vehicles=num_vehicles,
                num_nodes=num_nodes,
                seed=effective_seed,
                network=network,
                network_kwargs=network_options,
                poi_kwargs=poi_options,
                human_distribution_kwargs=human_distribution_kwargs,
                vehicle_distribution_kwargs=vehicle_distribution_kwargs,
                human_goal_nodes=human_goal_nodes,
            )
        elif human_goal_nodes is not None:
            scenario.human_goal_nodes = list(human_goal_nodes)

        self.scenario = scenario
        self.human_goal_nodes = list(scenario.human_goal_nodes)

        # Kwargs used to rebuild the scenario on each reset when randomization is
        # enabled. The network topology is reused (passed explicitly), so only the
        # agent positions and human destinations are re-drawn with a fresh seed.
        self.randomize_initial_state = randomize_initial_state
        self._scenario_build_kwargs = dict(
            num_humans=num_humans,
            num_vehicles=num_vehicles,
            network=scenario.network,
            network_kwargs=network_options,
            poi_kwargs=poi_options,
            human_distribution_kwargs=human_distribution_kwargs,
            vehicle_distribution_kwargs=vehicle_distribution_kwargs,
            human_goal_nodes=human_goal_nodes,
        )
        self.utility_bounds = None
        if self.reward_config.normalize_utility:
            self.utility_bounds = compute_scenario_utility_bounds(
                graph=scenario.network,
                population_size=num_humans,
                reward_config=self.reward_config,
            )

        self.env = TransportParallelEnv(
            num_humans=num_humans,
            num_vehicles=num_vehicles,
            network=scenario.network,
            human_speeds=human_speeds,
            vehicle_speeds=vehicle_speeds,
            vehicle_capacities=vehicle_capacities,
            vehicle_fuel_uses=vehicle_fuel_uses,
            observation_scenario=observation_scenario,
            render_mode=render_mode,
        )
        self.env.video_scenario = scenario
        self.env.reward_config = self.reward_config

        self.vehicle_agents = list(self.env.vehicle_agents)
        self.human_agents = list(self.env.human_agents)

        self.human_policy_class = human_policy_class
        self.human_policy_kwargs = human_policy_kwargs or {}
        self.human_policies: Dict[str, Optional[HumanPolicy]] = {}
        self.human_current_goals: Dict[str, Optional[int]] = {}

        self.reward_function = reward_function or self._default_reward_function
        self._current_reward_state: Optional[Dict[str, Dict[str, Any]]] = None
        self._next_reward_state: Optional[Dict[str, Dict[str, Any]]] = None
        self._current_reward_time: float = 0.0
        self._next_reward_time: float = 0.0
        self._last_reward_result: Optional[Dict[str, Any]] = None
        self._last_human_action_reasons: Dict[str, str] = {}
        self.last_boarding_debug_records: List[Dict[str, Any]] = []
        self.step_count = 0

        self.action_space = spaces.MultiDiscrete([num_nodes + 1] * num_vehicles)
        self.observation_space = spaces.Dict({
            "step_type": spaces.Discrete(4),
            "real_time": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "vehicle_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(num_vehicles, 2), dtype=np.float32),
            "human_positions": spaces.Box(low=-np.inf, high=np.inf, shape=(num_humans, 2), dtype=np.float32),
        })

    def _init_human_policies(self):
        """Initialize human policies after reset (when network is available)."""
        if self.human_policy_class is None:
            for agent in self.human_agents:
                self.human_policies[agent] = None
                self.human_current_goals[agent] = None
        else:
            for i, agent in enumerate(self.human_agents):
                if self.human_goal_nodes and i < len(self.human_goal_nodes):
                    initial_goal = {self.human_goal_nodes[i]}
                else:
                    nodes = list(self.env.network.nodes())
                    if nodes:
                        rng = np.random.RandomState()
                        initial_goal = {rng.choice(nodes)}
                    else:
                        initial_goal = {0}

                self.human_current_goals[agent] = list(initial_goal)[0]

                policy_kwargs = dict(self.human_policy_kwargs)
                candidate_kwargs = {
                    "agent_id": agent,
                    "network": self.env.network,
                    "target_nodes": initial_goal,
                }

                if "seed" not in policy_kwargs:
                    rng = np.random.RandomState()
                    candidate_kwargs["seed"] = rng.randint(0, 2**31 - 1)

                signature = inspect.signature(self.human_policy_class.__init__)
                accepts_kwargs = any(
                    param.kind == inspect.Parameter.VAR_KEYWORD
                    for param in signature.parameters.values()
                )
                accepted_names = set(signature.parameters)
                for key, value in candidate_kwargs.items():
                    if accepts_kwargs or key in accepted_names:
                        policy_kwargs.setdefault(key, value)

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
            if pos is not None and not isinstance(pos, tuple) and pos == current_goal:
                nodes = list(self.env.network.nodes())
                if nodes:
                    available_nodes = [n for n in nodes if n != pos]
                    rng = np.random.RandomState()
                    if available_nodes:
                        new_goal = rng.choice(available_nodes)
                    else:
                        new_goal = rng.choice(nodes)

                    self.human_current_goals[agent] = new_goal

                    if self.human_policies[agent] is not None:
                        self.human_policies[agent].reset()
                        if hasattr(self.human_policies[agent], "target_nodes"):
                            self.human_policies[agent].target_nodes = {new_goal}

    def _get_human_actions(self, obs_dict: Dict) -> Dict[str, int]:
        """Get actions for all human agents using their policies."""
        human_actions = {}
        self._last_human_action_reasons = {}

        for agent in self.human_agents:
            if agent not in obs_dict:
                human_actions[agent] = 0
                self._last_human_action_reasons[agent] = "Passing (no observation)"
                continue

            policy = self.human_policies.get(agent)
            if policy is None:
                action_space = self.env.action_space(agent)
                human_actions[agent] = action_space.sample()
                self._last_human_action_reasons[agent] = "Sampled random human action"
            else:
                obs = obs_dict[agent]
                action_space = self.env.action_space(agent)
                action_space_size = action_space.n
                action, reason = policy.get_action(obs, action_space_size)
                human_actions[agent] = action
                self._last_human_action_reasons[agent] = str(reason)

        return human_actions

    def _normalize_debug_value(self, value: Any) -> Any:
        """Convert numpy scalar containers into plain log-friendly values."""
        if isinstance(value, dict):
            return {
                self._normalize_debug_value(key): self._normalize_debug_value(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._normalize_debug_value(item) for item in value)
        if isinstance(value, list):
            return [self._normalize_debug_value(item) for item in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except (TypeError, ValueError):
                return value
        return value

    def _build_boarding_debug_records(
        self,
        human_actions: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Capture human/vehicle state for each boarding decision."""
        records = []
        vehicle_positions = {
            vehicle: self.env.agent_positions.get(vehicle)
            for vehicle in self.vehicle_agents
        }
        vehicle_destinations = dict(self.env.vehicle_destinations)

        for human in self.human_agents:
            human_pos = self.env.agent_positions.get(human)
            human_node = None if isinstance(human_pos, tuple) else human_pos
            same_node_vehicles = [
                vehicle
                for vehicle, vehicle_pos in vehicle_positions.items()
                if (
                    human_node is not None
                    and vehicle_pos is not None
                    and not isinstance(vehicle_pos, tuple)
                    and vehicle_pos == human_node
                )
            ]

            action_space_n = self.env.action_space(human).n
            record = {
                "step_count": self.step_count,
                "real_time": self.env.real_time,
                "human": human,
                "human_node": human_node,
                "human_position": human_pos,
                "vehicle_nodes": {
                    vehicle: None if isinstance(pos, tuple) else pos
                    for vehicle, pos in vehicle_positions.items()
                },
                "vehicle_positions": vehicle_positions,
                "vehicle_destinations": vehicle_destinations,
                "same_node_vehicles": same_node_vehicles,
                "human_aboard_before": self.env.human_aboard.get(human),
                "human_boarding_action": int(human_actions.get(human, 0)),
                "human_action_reason": self._last_human_action_reasons.get(human, ""),
                "human_action_space_n": int(action_space_n),
            }
            records.append(
                self._normalize_debug_value(record)
            )
        return records

    def _default_reward_function(self, obs_dict: Dict, next_obs_dict: Dict, actions_dict: Dict, current_time=None, next_time=None) -> Dict[str, float]:
        """
        Default reward function based on route-induced accessibility and equity utility.

        The wrapper stores current and next reward states before calling this
        function, so the default implementation can use the transition duration
        to compute a time-integrated utility reward:

            r_t = U(s_t) * delta_t

        where ``delta_t`` is measured in hours.
        """
        _ = obs_dict, actions_dict, current_time, next_time

        if self._current_reward_state is None or self._next_reward_state is None:
            return {agent: 0.0 for agent in self.vehicle_agents}

        reward_result = compute_reward(
            graph=self.env.network,
            current_human_to_node=self._current_reward_state["human_to_node"],
            current_human_to_route=self._current_reward_state["human_to_route"],
            next_human_to_node=self._next_reward_state["human_to_node"],
            next_human_to_route=self._next_reward_state["human_to_route"],
            reward_config=self.reward_config,
        )

        delta_t = max(
            0.0,
            float(self._next_reward_time) - float(self._current_reward_time),
        )
        current_utility = float(reward_result["current_utility"])
        raw_time_integrated_utility = current_utility * delta_t
        reward_result["delta_t"] = delta_t
        reward_result["raw_time_integrated_utility"] = raw_time_integrated_utility

        reward_utility = current_utility
        normalized_utility = None
        utility_out_of_bounds = False
        if self.reward_config.normalize_utility:
            lower = float(self.utility_bounds["utility_lower_bound"])
            upper = float(self.utility_bounds["utility_upper_bound"])
            denominator = upper - lower
            if denominator > 0.0:
                normalized_utility = (current_utility - lower) / denominator
            else:
                normalized_utility = 0.0
            utility_out_of_bounds = normalized_utility < 0.0 or normalized_utility > 1.0
            if self.reward_config.clip_normalized_utility:
                normalized_utility = min(1.0, max(0.0, normalized_utility))
            reward_utility = (
                normalized_utility * self.reward_config.normalized_reward_scale
            )
            reward_result["utility_lower_bound"] = lower
            reward_result["utility_upper_bound"] = upper

        reward_result["current_normalized_utility"] = normalized_utility
        reward_result["utility_out_of_bounds"] = utility_out_of_bounds
        reward_result["reward"] = float(reward_utility) * delta_t
        self._last_reward_result = reward_result

        total_reward = float(reward_result["reward"])
        if not self.vehicle_agents:
            return {}

        reward_per_vehicle = total_reward / float(len(self.vehicle_agents))
        return {agent: reward_per_vehicle for agent in self.vehicle_agents}

    def _position_to_node(self, pos: Any) -> Optional[Any]:
        """Map an environment position to a representative node."""
        if pos is None:
            return None
        if isinstance(pos, tuple):
            edge, coord = pos
            _ = coord
            if edge and len(edge) >= 2:
                return edge[0]
            return None
        return pos

    def _get_vehicle_route_nodes(self, vehicle: str) -> List[Any]:
        """
        Build a simple route-node representation for a vehicle from its current
        position and destination.
        """
        vehicle_pos = self.env.agent_positions.get(vehicle)
        current_node = self._position_to_node(vehicle_pos)
        if current_node is None:
            return []

        destination = self.env.vehicle_destinations.get(vehicle)
        if destination is None or destination == current_node:
            return [current_node]

        try:
            route_nodes = nx.shortest_path(
                self.env.network,
                source=current_node,
                target=destination,
                weight="length",
            )
            return list(route_nodes)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [current_node]

    def _extract_reward_state(self) -> Dict[str, Dict[str, Any]]:
        """
        Extract the reward-relevant human state from the underlying environment.

        Each human is represented by:
        - a current node
        - a route, which is either the route of the boarded vehicle or the
          singleton route at the human's current node
        """
        human_to_node: Dict[str, Any] = {}
        human_to_route: Dict[str, Sequence[Any]] = {}

        for human in self.human_agents:
            human_pos = self.env.agent_positions.get(human)
            human_node = self._position_to_node(human_pos)
            if human_node is None:
                continue

            human_to_node[human] = human_node

            aboard = self.env.human_aboard.get(human)
            if False and aboard is not None:
                route_nodes = self._get_vehicle_route_nodes(aboard)
            else:
                route_nodes = [human_node]

            human_to_route[human] = route_nodes

        return {
            "human_to_node": human_to_node,
            "human_to_route": human_to_route,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.step_count = 0
        if self.randomize_initial_state:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))
            self.scenario = build_transport_scenario(
                seed=episode_seed, **self._scenario_build_kwargs
            )
            self.human_goal_nodes = list(self.scenario.human_goal_nodes)
        initial_state = dict(self.scenario.initial_state)
        reset_options = {}
        if options:
            option_initial_state = options.get("initial_state")
            if option_initial_state:
                initial_state.update(option_initial_state)
            reset_options.update(options)
        reset_options["initial_state"] = initial_state

        obs_dict, info_dict = self.env.reset(seed=seed, options=reset_options)
        _ = info_dict

        self._init_human_policies()
        self._current_reward_state = self._extract_reward_state()
        self._next_reward_state = None
        self._current_reward_time = float(self.env.real_time)
        self._next_reward_time = float(self.env.real_time)
        self._last_reward_result = None

        obs = self._get_observation(obs_dict)
        info = {"step_type": self.env.step_type, "real_time": self.env.real_time}
        return obs, info

    def step(self, vehicle_actions: np.ndarray, compute_reward: bool = True):
        """Take a step with actions for vehicles only."""
        self.step_count += 1
        self.last_boarding_debug_records = []

        self._current_reward_state = self._extract_reward_state()
        self._current_reward_time = float(self.env.real_time)

        current_obs_dict = self.env._generate_observations()
        self._update_human_goals(current_obs_dict)
        human_actions = self._get_human_actions(current_obs_dict)
        if self.env.step_type == "boarding":
            self.last_boarding_debug_records = self._build_boarding_debug_records(human_actions)

        actions_dict = dict(human_actions)
        for i, agent in enumerate(self.vehicle_agents):
            if i < len(vehicle_actions):
                actions_dict[agent] = int(vehicle_actions[i])
            else:
                actions_dict[agent] = 0

        obs_dict, rewards_dict, terms_dict, truncs_dict, info_dict = self.env.step(actions_dict)
        _ = rewards_dict, info_dict
        for record in self.last_boarding_debug_records:
            record["human_aboard_after"] = self._normalize_debug_value(
                self.env.human_aboard.get(record["human"])
            )

        reward = 0
        if compute_reward:
            self._next_reward_state = self._extract_reward_state()
            self._next_reward_time = float(self.env.real_time)
            next_obs_dict = self.env._generate_observations()

            vehicle_rewards = self.reward_function(
                current_obs_dict,
                next_obs_dict,
                actions_dict,
                self._current_reward_time,
                self._next_reward_time,
            )
            reward = sum(vehicle_rewards.get(agent, 0.0) for agent in self.vehicle_agents)
        terminated = False
        truncated = self.env.real_time >= self.max_steps

        obs = self._get_observation(obs_dict)
        info = {
            "step_type": self.env.step_type,
            "real_time": self.env.real_time,
            "step_count": self.step_count,
            "underlying_terminations": terms_dict,
            "underlying_truncations": truncs_dict,
        }
        if self._last_reward_result is not None:
            info["current_utility"] = self._last_reward_result["current_utility"]
            info["next_utility"] = self._last_reward_result["next_utility"]
            info["current_total_accessibility"] = self._last_reward_result["current_total_accessibility"]
            info["next_total_accessibility"] = self._last_reward_result["next_total_accessibility"]
            info["delta_t"] = self._last_reward_result["delta_t"]
            info["time_integrated_utility"] = self._last_reward_result["reward"]
            info["raw_time_integrated_utility"] = self._last_reward_result["raw_time_integrated_utility"]
            info["current_normalized_utility"] = self._last_reward_result["current_normalized_utility"]
            info["utility_out_of_bounds"] = self._last_reward_result["utility_out_of_bounds"]
            if "utility_lower_bound" in self._last_reward_result:
                info["utility_lower_bound"] = self._last_reward_result["utility_lower_bound"]
                info["utility_upper_bound"] = self._last_reward_result["utility_upper_bound"]
        return obs, reward, terminated, truncated, info

    def _get_observation(self, obs_dict: Dict) -> Dict:
        """Convert multi-agent observations to a single-agent observation."""
        _ = obs_dict
        step_types = {"routing": 0, "unboarding": 1, "boarding": 2, "departing": 3}
        step_type_idx = step_types.get(self.env.step_type, 0)

        vehicle_positions = []
        for vehicle in self.vehicle_agents:
            pos = self.env.agent_positions.get(vehicle)
            if pos is None:
                vehicle_positions.append([0.0, 0.0])
            elif isinstance(pos, tuple):
                edge, coord = pos
                _ = coord
                if edge and len(edge) >= 2:
                    node = edge[0]
                    node_data = self.env.network.nodes.get(node, {})
                    x = float(node_data.get("x", 0.0))
                    y = float(node_data.get("y", 0.0))
                    vehicle_positions.append([x, y])
                else:
                    vehicle_positions.append([0.0, 0.0])
            else:
                node_data = self.env.network.nodes.get(pos, {})
                x = float(node_data.get("x", 0.0))
                y = float(node_data.get("y", 0.0))
                vehicle_positions.append([x, y])

        human_positions = []
        for human in self.human_agents:
            pos = self.env.agent_positions.get(human)
            if pos is None:
                human_positions.append([0.0, 0.0])
            elif isinstance(pos, tuple):
                edge, coord = pos
                _ = coord
                if edge and len(edge) >= 2:
                    node = edge[0]
                    node_data = self.env.network.nodes.get(node, {})
                    x = float(node_data.get("x", 0.0))
                    y = float(node_data.get("y", 0.0))
                    human_positions.append([x, y])
                else:
                    human_positions.append([0.0, 0.0])
            else:
                node_data = self.env.network.nodes.get(pos, {})
                x = float(node_data.get("x", 0.0))
                y = float(node_data.get("y", 0.0))
                human_positions.append([x, y])

        while len(vehicle_positions) < self.num_vehicles:
            vehicle_positions.append([0.0, 0.0])
        vehicle_positions = vehicle_positions[:self.num_vehicles]

        while len(human_positions) < self.num_humans:
            human_positions.append([0.0, 0.0])
        human_positions = human_positions[:self.num_humans]

        return {
            "step_type": step_type_idx,
            "real_time": np.array([self.env.real_time], dtype=np.float32),
            "vehicle_positions": np.array(vehicle_positions, dtype=np.float32),
            "human_positions": np.array(human_positions, dtype=np.float32),
        }

    def render(self):
        """Render the environment."""
        return self.env.render()

    def close(self):
        """Close the environment."""
        return self.env.close()


def create_transport_env(
    num_humans: int = _DEFAULT_CONFIG.scenario.num_humans,
    num_vehicles: int = _DEFAULT_CONFIG.scenario.num_vehicles,
    num_nodes: int = _DEFAULT_CONFIG.scenario.num_nodes,
    seed: Optional[int] = None,
    **kwargs,
) -> TransportGymWrapper:
    """
    Convenience function to create a transport environment with a scenario-built
    random network.
    """
    mobility_config = kwargs.get("mobility_config") or _DEFAULT_CONFIG.mobility
    network_kwargs = dict(kwargs.pop("network_kwargs", {}) or {})
    network_kwargs.setdefault("speed_mean", mobility_config.edge_speed_kmh)
    poi_kwargs = dict(kwargs.pop("poi_kwargs", {}) or {})
    poi_kwargs.setdefault("central_count", _DEFAULT_CONFIG.scenario.central_count)
    effective_seed = _DEFAULT_CONFIG.scenario.seed if seed is None else seed
    scenario = build_transport_scenario(
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        num_nodes=num_nodes,
        seed=effective_seed,
        network_kwargs=network_kwargs,
        poi_kwargs=poi_kwargs,
    )
    wrapper = TransportGymWrapper(
        num_humans=num_humans,
        num_vehicles=num_vehicles,
        scenario=scenario,
        network_kwargs=network_kwargs,
        poi_kwargs=poi_kwargs,
        **kwargs,
    )
    return wrapper
