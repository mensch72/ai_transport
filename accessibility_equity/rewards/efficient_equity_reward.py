from typing import Dict
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

    def _compute_node_values(self, graph: nx.DiGraph):
        node_poi_count = {}
        nodes = list(graph.nodes)
        for poi in graph.graph["poi_records"]:
            if poi["poi_type"] not in node_poi_count:
                node_poi_count.update({poi["poi_type"]: {n: 0 for n in nodes}})
                node_poi_count[poi["poi_type"]].update({"weight": poi["poi_weight"]})
            node_poi_count[poi["poi_type"]][poi["node"]] += 1
        node_values = {n: 0 for n in nodes}
        for node_i in nodes:
            for poi_type in node_poi_count:
                res = 0
                for node_j in nodes:
                    res += node_poi_count[poi_type][node_j] * np.exp(
                        -self.beta * self.dist_matrix[node_i][node_j]
                    )
                res = node_poi_count[poi_type]["weight"] * (res**self.alpha)
                node_values[node_i] += res
        return node_values

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
        self.node_values = self._compute_node_values(graph)

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

    def reward(self, obs: Dict, actions: Dict):
        reward = 0
        for human in self.human_agents:
            pos = obs[human]["my_position"]
            node = self._position_to_node(pos)
            node_value = max(self.node_values[node], self.epsilon)
            reward += node_value ** (-self.xi)
        reward = -(reward**self.eta)
        print(reward)

        return {vehicle: reward for vehicle in self.vehicle_agents}
