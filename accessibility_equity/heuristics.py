from typing import Dict
from typing import Union
import networkx as nx
import numpy as np
from networkx.algorithms.approximation import traveling_salesman_problem as tsp
from networkx.algorithms.approximation import simulated_annealing_tsp
from scipy.sparse.csgraph import floyd_warshall
from stable_baselines3 import DQN
from accessibility_equity.wrappers.gym_wrapper import TransportGymWrapper
from accessibility_equity.wrappers.dqn_wrapper import DQNTransportWrapper


class VehicleAgent:
    def __init__(self, env: TransportGymWrapper | DQNTransportWrapper):
        if type(env) is TransportGymWrapper:
            self.env = env
            self.env_type = "Gym"
        elif type(env) is DQNTransportWrapper:
            self.env = env.base_env
            self.env_type = "DQN"
        else:
            raise TypeError(
                f"TSPVehicleAgent needs environment of type TransportGymWrapper or DQNTransportWrapper, not {type(env)}."
            )

    def _get_step_type(self, obs):
        if self.env_type == "Gym":
            return obs["step_type"]
        if self.env_type == "DQN":
            if type(obs) is dict:
                return obs["features"][0]
            return obs[0]

    def _current_node(self):
        vehicle_agent = list(self.env.env.vehicle_agents)[0]
        pos = self.env.env.agent_positions[vehicle_agent]
        if type(pos) is tuple:
            return None
        return pos

    def _depart(self, current_node, next_node):
        out_edges = list(self.env.env.network.out_edges(current_node))
        for i, e in enumerate(out_edges):
            if e[1] == next_node:
                return i + 1
        return 0


class TSPVehicleAgent(VehicleAgent):
    def __init__(self, env: TransportGymWrapper | DQNTransportWrapper):
        super().__init__(env)
        self.route = tsp(
            self.env.env.network, method=simulated_annealing_tsp, init_cycle="greedy"
        )[:-1]
        self.route_dict = {
            self.route[i]: self.route[(i + 1) % len(self.route)]
            for i in range(len(self.route))
        }

    def _current_node(self):
        vehicle_agent = list(self.env.env.vehicle_agents)[0]
        pos = self.env.env.agent_positions[vehicle_agent]
        if type(pos) is tuple:
            return None
        return pos

    def next_node(self):
        current_node = self._current_node()
        if current_node == None:
            return 0
        return self.route_dict[current_node]

    def get_action(self, obs):
        step_type = self._get_step_type(obs)
        action = 0
        if step_type == 0:
            action = self.next_node() + 1
        elif step_type == 3:
            action = self._depart(self._current_node, self.next_node())
        if self.env_type == "Gym":
            return [action]
        elif self.env_type == "DQN":
            return action


class GoToHumanVehicleAgent(VehicleAgent):
    def __init__(self, env: TransportGymWrapper | DQNTransportWrapper):
        super().__init__(env)
        self.graph = self.env.env.network
        length_matrix = nx.to_scipy_sparse_array(self.graph, weight="length")
        self.dist_matrix, pred_matrix = floyd_warshall(
            length_matrix, return_predecessors=True
        )
        self.next_matrix = self._compute_next_matrix(pred_matrix)

    def _compute_next_matrix(self, predecessors):
        NULL = -9999
        n = predecessors.shape[0]
        H = predecessors.copy()
        i_idx = np.arange(n)[:, None]
        for _ in range(n):
            safe = np.where(H == NULL, 0, H)
            prev = predecessors[i_idx, safe]
            step = (H != NULL) & (prev != i_idx)
            if not step.any():
                break
            H = np.where(step, prev, H)
        return H

    def _pos_to_node(self, pos):
        if not isinstance(pos, tuple):
            return pos
        edge, coord = pos
        edge_len = self.graph[edge[0]][edge[1]]["length"]
        return edge[0] if coord < edge_len / 2 else edge[1]

    def _get_closest_human_node(self):
        closest_node = None
        human_positions = [
            self.env.env.agent_positions.get(human) for human in self.env.human_agents
        ]
        for pos in human_positions:
            pos_node = self._pos_to_node(pos)
            current_node = self._current_node()
            if current_node == pos_node:
                return None
            if closest_node is None:
                closest_node = pos_node
                continue
            closest_dist = self.dist_matrix[current_node, closest_node]
            pos_dist = self.dist_matrix[current_node, pos_node]
            if pos_dist < closest_dist:
                closest_node = pos_node
        return closest_node

    def get_action(self, obs):
        action = self._get_closest_human_node()
        if action is None:
            action = self.env.scenario.poi_distribution["voronoi_area_info"][
                "central_nodes"
            ][0]
        step_type = self._get_step_type(obs)
        if action == self._current_node():
            action = 0
        elif step_type == 0:
            action = action + 1
        elif step_type == 3:
            current_node = self._current_node()
            next_node = self.next_matrix[current_node][action]
            action = self._depart(current_node, next_node)
        else:
            action = 0
        if self.env_type == "Gym":
            return [action]
        if self.env_type == "DQN":
            return action
