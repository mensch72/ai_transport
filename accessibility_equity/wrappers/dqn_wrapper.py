"""
DQN-oriented wrapper for the accessibility-equity transport task.

Standard DQN expects a single ``Discrete`` action space and a flat vector
observation. This wrapper provides that first, minimal DQN-compatible view by
restricting control to one vehicle while reusing the project-local scenario,
human policy, simulator, and reward logic from ``TransportGymWrapper``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces

from accessibility_equity.wrappers.gym_wrapper import TransportGymWrapper


REWARD_SCALE = 1.0


class DQNTransportWrapper(gym.Env):
    """
    Single-vehicle DQN-compatible wrapper.

    The underlying environment is still ``TransportGymWrapper``. This class
    adapts the task into a decision process where each DQN action is meaningful:

    - fixed destination action: ``0`` clears the destination, ``1..N`` choose a
      network node as the vehicle destination during routing
    - ``decision_mode='routing'`` returns only vehicle routing decisions and
      auto-departs along the current shortest path between routing decisions
    - ``decision_mode='routing_departing'`` returns both routing decisions and
      departing edge choices; unboarding, boarding, and edge-travel phases are
      still advanced internally
    - dict observation -> flat float vector with compact control-state features

    Note: rewards are multiplied by ``REWARD_SCALE`` for DQN training signal
    strength. The unscaled reward is reported in ``info["raw_reward"]``.
    """

    metadata = TransportGymWrapper.metadata

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.use_action_masking = bool(kwargs.pop("use_action_masking", False))
        self.decision_mode = str(kwargs.pop("decision_mode", "routing"))
        valid_decision_modes = {"routing", "routing_departing"}
        if self.decision_mode not in valid_decision_modes:
            raise ValueError(
                "decision_mode must be one of "
                f"{sorted(valid_decision_modes)}, got {self.decision_mode!r}."
            )
        if "seed" in kwargs and "scenario_seed" not in kwargs:
            kwargs["scenario_seed"] = kwargs.pop("seed")
        kwargs["num_vehicles"] = 1
        self.base_env = TransportGymWrapper(*args, **kwargs)
        self.num_humans = self.base_env.num_humans
        self.num_vehicles = 1
        self.vehicle_agent = self.base_env.vehicle_agents[0]
        self.node_order = list(self.base_env.env.network.nodes())
        self.node_to_index = {node: index for index, node in enumerate(self.node_order)}
        self.auto_advance_callback: Optional[Callable[["DQNTransportWrapper", Dict[str, Any]], None]] = None
        self.action_space = spaces.Discrete(len(self.node_order) + 1)
        flat_observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(5 + 2 * self.num_vehicles + 3 * self.num_humans,),
            dtype=np.float32,
        )
        if self.use_action_masking:
            self.observation_space = spaces.Dict(
                {
                    "features": flat_observation_space,
                    "action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.action_space.n,),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = flat_observation_space

    def action_masks(self) -> np.ndarray:
        """
        Return the valid mask for the unified vehicle-node action space.

        Actions use one stable index space:
        - 0: pass/stay/no target
        - 1..N: node index in ``self.node_order``

        In routing, node actions set a vehicle destination. When
        ``decision_mode='routing_departing'``, departing node actions mean
        depart to that neighbor node, so only outgoing neighbors are valid.
        Boarding and unboarding are environment-controlled human phases, so the
        learning vehicle can only pass and the wrapper normally skips them.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True

        env = self.base_env.env
        if not self._vehicle_is_at_node():
            return mask

        current_node = self._position_to_node(env.agent_positions.get(self.vehicle_agent))
        if current_node is None:
            return mask

        if env.step_type == "routing":
            mask[1:] = True
        elif env.step_type == "departing" and self.decision_mode == "routing_departing":
            for _source, target in env.network.out_edges(current_node):
                node_index = self.node_to_index.get(target)
                if node_index is not None:
                    mask[node_index + 1] = True

        return mask

    def _masked_node_action_to_underlying_action(self, action: int) -> int:
        """
        Translate a unified node action into the current underlying vehicle action.
        """
        env = self.base_env.env
        action = int(action)
        if action < 0 or action >= self.action_space.n:
            return 0

        if not self.action_masks()[action]:
            return 0

        if env.step_type == "routing":
            return action

        if env.step_type != "departing" or action == 0:
            return 0

        current_node = self._position_to_node(env.agent_positions.get(self.vehicle_agent))
        target_node = self.node_order[action - 1]
        outgoing_edges = list(env.network.out_edges(current_node))
        for index, edge in enumerate(outgoing_edges, start=1):
            if edge[1] == target_node:
                return index
        return 0

    def _add_action_mask_info(self, info: Dict[str, Any]) -> Dict[str, Any]:
        info = dict(info)
        info["action_mask"] = self.action_masks()
        return info

    def _format_observation(self, obs: Dict[str, Any]) -> np.ndarray | Dict[str, np.ndarray]:
        flat_obs = self._flatten_observation(obs)
        if not self.use_action_masking:
            return flat_obs
        return {
            "features": flat_obs,
            "action_mask": self.action_masks().astype(np.float32),
        }

    def _position_to_node(self, pos: Any) -> Optional[Any]:
        if pos is None:
            return None
        if isinstance(pos, tuple):
            edge, _coord = pos
            if edge and len(edge) >= 2:
                return edge[0]
            return None
        return pos

    def _node_feature(self, node: Any) -> float:
        if node is None:
            return -1.0
        return float(self.node_to_index.get(node, -1))

    def _vehicle_is_at_node(self) -> bool:
        pos = self.base_env.env.agent_positions.get(self.vehicle_agent)
        return pos is not None and not isinstance(pos, tuple)

    def _make_vehicle_action_for_current_phase(
        self,
        routing_action: Optional[int],
        hold_destination_action: Optional[int] = None,
        auto_depart: bool = True,
    ) -> int:
        """
        Return the vehicle action for the current underlying phase.

        DQN only controls routing destinations. In intermediate routing phases,
        ``hold_destination_action`` repeats the active target so action 0 does
        not accidentally clear the destination. In departing phases, the wrapper
        chooses the outgoing edge on the shortest path to the current
        destination. Other phases pass for the vehicle.
        """
        env = self.base_env.env
        step_type = env.step_type

        if step_type == "routing":
            if not self._vehicle_is_at_node():
                return 0
            if routing_action is not None:
                return int(routing_action)
            if hold_destination_action is not None:
                return int(hold_destination_action)
            return 0

        if step_type != "departing" or not self._vehicle_is_at_node() or not auto_depart:
            return 0

        current_node = self._position_to_node(env.agent_positions.get(self.vehicle_agent))
        destination = env.vehicle_destinations.get(self.vehicle_agent)
        if current_node is None or destination is None or current_node == destination:
            return 0

        try:
            path = nx.shortest_path(
                env.network,
                source=current_node,
                target=destination,
                weight="length",
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0

        if len(path) < 2:
            return 0

        next_edge = (path[0], path[1])
        outgoing_edges = list(env.network.out_edges(current_node))
        for index, edge in enumerate(outgoing_edges, start=1):
            if edge == next_edge:
                return index
        return 0

    def _at_dqn_decision_point(self) -> bool:
        if not self._vehicle_is_at_node():
            return False
        step_type = self.base_env.env.step_type
        if step_type == "routing":
            return True
        return step_type == "departing" and self.decision_mode == "routing_departing"

    def _auto_vehicle_action_between_decisions(self) -> int:
        return self._make_vehicle_action_for_current_phase(
            routing_action=None,
            auto_depart=self.decision_mode == "routing",
        )

    def _action_to_destination(self, action: int) -> Optional[Any]:
        if int(action) <= 0:
            return None
        index = int(action) - 1
        if 0 <= index < len(self.node_order):
            return self.node_order[index]
        return None

    def _destination_reached(self, target_destination: Optional[Any]) -> bool:
        if target_destination is None:
            return True
        current_node = self._position_to_node(
            self.base_env.env.agent_positions.get(self.vehicle_agent)
        )
        return current_node == target_destination

    def _render_auto_advance_frame(self) -> None:
        """
        Capture frames during wrapper-internal auto-advance when recording.

        Training environments use ``render_mode=None``, so this only affects
        explicit evaluation/video runs.
        """
        if self.auto_advance_callback is not None:
            self.auto_advance_callback(self, {})
        if self.base_env.render_mode != "human":
            return
        if not getattr(self.base_env.env, "_recording", False):
            return
        self.base_env.env.render()

    def _flatten_observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Flatten the project dict observation into a fixed DQN vector.
        """
        env = self.base_env.env
        step_type = np.asarray([float(obs.get("step_type", 0))], dtype=np.float32)
        real_time = np.asarray(obs.get("real_time", [0.0]), dtype=np.float32).reshape(-1)
        vehicle_positions = np.asarray(
            obs.get("vehicle_positions", np.zeros((self.num_vehicles, 2))),
            dtype=np.float32,
        ).reshape(-1)
        human_positions = np.asarray(
            obs.get("human_positions", np.zeros((self.num_humans, 2))),
            dtype=np.float32,
        ).reshape(-1)
        current_node = self._position_to_node(env.agent_positions.get(self.vehicle_agent))
        destination = env.vehicle_destinations.get(self.vehicle_agent)
        control_state = np.asarray(
            [
                self._node_feature(current_node),
                self._node_feature(destination),
                float(self.action_space.n),
            ],
            dtype=np.float32,
        )
        human_aboard = np.asarray(
            [
                1.0 if env.human_aboard.get(human) == self.vehicle_agent else 0.0
                for human in self.base_env.human_agents[: self.num_humans]
            ],
            dtype=np.float32,
        )

        flat_obs = np.concatenate(
            [
                step_type,
                real_time[:1],
                vehicle_positions,
                human_positions,
                control_state,
                human_aboard,
            ],
            dtype=np.float32,
        )
        return flat_obs.astype(np.float32, copy=False)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        obs, info = self.base_env.reset(seed=seed, options=options)
        # Present DQN with an actual routing decision point after reset.
        guard = 0
        while not self._at_dqn_decision_point() and guard < 100:
            obs, _reward, terminated, truncated, info = self.base_env.step(
                np.asarray([self._auto_vehicle_action_between_decisions()], dtype=np.int64)
            )
            guard += 1
            if terminated or truncated:
                break
        return self._format_observation(obs), self._add_action_mask_info(info)

    def _decision_step_reward(self, decision_obs: Dict[str, Any], decision_time: float) -> float:
        """
        Compute reward between decision steps.
        """
        next_time = float(self.base_env.env.real_time)
        vehicle_rewards = self.base_env.reward_function(
            decision_obs, {}, next_time, decision_time
        )
        if not isinstance(vehicle_rewards, dict):
            return float(vehicle_rewards)
        return sum(
            float(vehicle_rewards.get(agent, 0.0))
            for agent in self.base_env.vehicle_agents
        )

    def step(self, action: int):
        decision_obs = self.base_env.env._generate_observations()
        decision_time = float(self.base_env.env.real_time)

        if self.use_action_masking:
            action = int(action)
            pre_step_mask = self.action_masks()
            action_was_valid = 0 <= action < self.action_space.n and bool(pre_step_mask[action])
            underlying_action = self._masked_node_action_to_underlying_action(int(action))
            boarding_debug_records = []
            obs, _reward, terminated, truncated, info = self.base_env.step(
                np.asarray([underlying_action], dtype=np.int64)
            )
            if self.base_env.last_boarding_debug_records:
                boarding_debug_records.extend(self.base_env.last_boarding_debug_records)
            self._render_auto_advance_frame()

            guard = 1
            while not (terminated or truncated) and not self._at_dqn_decision_point() and guard < 1000:
                obs, _reward, terminated, truncated, info = self.base_env.step(
                    np.asarray([self._auto_vehicle_action_between_decisions()], dtype=np.int64)
                )
                if self.base_env.last_boarding_debug_records:
                    for record in self.base_env.last_boarding_debug_records:
                        enriched_record = dict(record)
                        enriched_record["dqn_underlying_step"] = guard + 1
                        boarding_debug_records.append(enriched_record)
                self._render_auto_advance_frame()
                guard += 1

            if guard >= 1000:
                truncated = True

            raw_reward = self._decision_step_reward(decision_obs, decision_time)
            scaled_reward = float(raw_reward) * REWARD_SCALE
            info = self._add_action_mask_info(info)
            info["raw_reward"] = float(raw_reward)
            info["reward_scale"] = REWARD_SCALE
            info["scaled_reward"] = scaled_reward
            info["dqn_decision_mode"] = self.decision_mode
            info["dqn_node_action"] = int(action)
            info["dqn_underlying_vehicle_action"] = int(underlying_action)
            info["dqn_underlying_steps"] = guard
            info["dqn_auto_advance_guard_hit"] = guard >= 1000
            info["dqn_action_was_valid"] = action_was_valid
            info["dqn_boarding_debug_records"] = boarding_debug_records
            return (
                self._format_observation(obs),
                scaled_reward,
                bool(terminated),
                bool(truncated),
                info,
            )

        routing_action: Optional[int] = int(action)
        target_destination = self._action_to_destination(routing_action)
        raw_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}
        obs: Dict[str, Any] = {}
        boarding_debug_records = []

        # Apply the DQN destination decision once, then automatically advance
        # until the chosen destination is reached at a routing decision point.
        # For action 0 (clear destination / pass), advance one full control
        # cycle and return at the next routing decision point.
        applied_dqn_routing_action = False
        guard = 0
        while guard < 1000:
            hold_destination_action = None
            if (
                applied_dqn_routing_action
                and target_destination is not None
                and not self._destination_reached(target_destination)
            ):
                hold_destination_action = int(action)

            vehicle_action = np.asarray(
                [
                    self._make_vehicle_action_for_current_phase(
                        routing_action,
                        hold_destination_action=hold_destination_action,
                    )
                ],
                dtype=np.int64,
            )
            obs, _reward, terminated, truncated, info = self.base_env.step(vehicle_action)
            if self.base_env.last_boarding_debug_records:
                for record in self.base_env.last_boarding_debug_records:
                    enriched_record = dict(record)
                    enriched_record["dqn_underlying_step"] = guard + 1
                    enriched_record["dqn_target_destination"] = target_destination
                    boarding_debug_records.append(enriched_record)
            self._render_auto_advance_frame()
            guard += 1
            if routing_action is not None:
                applied_dqn_routing_action = True
            routing_action = None

            if terminated or truncated:
                break

            if self._at_dqn_decision_point() and applied_dqn_routing_action:
                if target_destination is None or self._destination_reached(target_destination):
                    break

        if guard >= 1000:
            truncated = True

        raw_reward = self._decision_step_reward(decision_obs, decision_time)
        scaled_reward = raw_reward * REWARD_SCALE
        info = dict(info)
        info["raw_reward"] = raw_reward
        info["reward_scale"] = REWARD_SCALE
        info["scaled_reward"] = scaled_reward
        info["dqn_target_destination"] = target_destination
        info["dqn_underlying_steps"] = guard
        info["dqn_auto_advance_guard_hit"] = guard >= 1000
        info["dqn_boarding_debug_records"] = boarding_debug_records
        return (
            self._format_observation(obs),
            float(scaled_reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def render(self):
        return self.base_env.render()

    def close(self) -> None:
        self.base_env.close()

    @property
    def unwrapped(self):
        return self.base_env.unwrapped


def create_dqn_env(*args: Any, **kwargs: Any) -> DQNTransportWrapper:
    """
    Create a single-vehicle DQN-compatible environment.
    """
    return DQNTransportWrapper(*args, **kwargs)
