from typing import Dict, List
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall


class EquityReward:
    def __init__(self, beta, alpha, xi, eta, epsilon=1e-6):
        self.dist_matrix = None
        self.vehicle_agents = None
        self.human_agents = None
        self.beta = beta
        self.alpha = alpha
        self.xi = xi
        self.eta = eta
        self.epsilon = epsilon

    def _compute_poi_counts(self, nodes: List, poi_records: Dict) -> Dict:
        node_poi_count = {}
        for poi in poi_records:
            if poi["poi_type"] not in node_poi_count:
                node_poi_count.update({poi["poi_type"]: {n: 0 for n in nodes}})
                node_poi_count[poi["poi_type"]].update({"weight": poi["poi_weight"]})
            node_poi_count[poi["poi_type"]][poi["node"]] += 1
        return node_poi_count

    def _compute_node_values(self, s_nodes: List, d_nodes: List, dist_matrix):
        node_values = {n: 0 for n in s_nodes}
        for s_node in s_nodes:
            for poi_type in self.node_poi_count:
                res = 0
                for d_node in d_nodes:
                    res += self.node_poi_count[poi_type][d_node] * np.exp(
                        -self.beta * dist_matrix[s_node][d_node]
                    )
                res = self.node_poi_count[poi_type]["weight"] * (res**self.alpha)
                node_values[s_node] += res
        return node_values
    
    def _compute_vehicle_values(self):


    def initialize(self, graph: nx.DiGraph, vehicle_agents, human_agents):
        length_matrix = nx.to_scipy_sparse_array(graph, weight='length')
        speed_matrix = nx.to_scipy_sparse_array(graph, weight='speed')
        cost_matrix = length_matrix.copy()
        with np.errstate(divide='ignore', invalid='ignore'):
            cost_matrix.data = np.where(
                speed_matrix.data != 0, 
                length_matrix.data / speed_matrix.data, 
                0.0
            )
        cost_matrix.eliminate_zeros()
        self.dist_matrix = floyd_warshall(cost_matrix)
        self.vehicle_agents = vehicle_agents
        self.human_agents = human_agents
        self.nodes = list(graph.nodes)
        self.node_poi_count = self._compute_poi_counts(self.nodes, graph.graph["poi_records"])
        self.node_values = self._compute_node_values(self.nodes, self.nodes, self.dist_matrix)

    def _position_to_node(self, pos):
        """Map an environment position to a node."""
        if pos is None:
            return None
        if isinstance(pos, tuple):
            edge, coord = pos
            _ = coord
            if edge and len(edge) >= 2:
                return edge[0]
            return None
        return pos

    def compute_utility(self, obs: Dict):
        utitlity = 0
        for human in self.human_agents:
            pos = obs[human]["my_position"]
            node = self._position_to_node(pos)
            node_value = max(self.node_values[node], self.epsilon)
            utitlity += node_value ** (-self.xi)
        utitlity = -(utitlity**self.eta)
        return utitlity
    

    def reward(
            self, 
            obs_dict: Dict, 
            actions_dict: Dict, 
            next_reward_time = None, 
            current_reward_time = None) -> Dict[str, float]:
        _ = obs_dict, actions_dict

        total_reward = self.compute_utility(obs_dict)

        if next_reward_time is not None and current_reward_time is not None:
            delta_t = max(
                0.0,
                float(next_reward_time) - float(current_reward_time),
            )
            total_reward = total_reward * delta_t

        reward_per_vehicle = total_reward / float(len(self.vehicle_agents))
        return {agent: reward_per_vehicle for agent in self.vehicle_agents}
