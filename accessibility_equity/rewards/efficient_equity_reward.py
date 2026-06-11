from typing import Dict, List, Union
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import floyd_warshall


class EquityReward:
    def __init__(self, beta, alpha, xi, eta, human_speed, vehicle_speed, epsilon=1e-6):
        self.beta = beta
        self.alpha = alpha
        self.xi = xi
        self.eta = eta
        self.human_speed = human_speed
        self.vehicle_speed = vehicle_speed
        self.epsilon = epsilon

    def _compute_dist_and_pred_matrix(self, graph: nx.DiGraph):
        """
        Compute the distance matrix, which gives pairwise distances between all nodes
        and the predicessor matrix, which can be used to reconstruct all shortes paths.

        Input:
        graph: Graph used to compute the matrices.

        Output:
        dist_matrix: Matrix where dist_matrix[i][j] is the distance from node i to node j.
        pred_matrix: Matrix where pred_matrix[i][j] is the index of the last node before 
                     node j on the shortest path from node i to node j.
        """
        length_matrix = nx.to_scipy_sparse_array(graph, weight='length')
        dist_matrix, pred_matrix = floyd_warshall(length_matrix, return_predecessors=True)
        return dist_matrix, pred_matrix

    # TODO: poi type counts are already in the node attribute dict of the graph,
    # so this function is redundant and should be replaced
    def _compute_poi_counts(self, nodes: List, poi_records: Dict) -> Dict:
        node_poi_count = {}
        for poi in poi_records:
            if poi["poi_type"] not in node_poi_count:
                node_poi_count.update({poi["poi_type"]: {n: 0 for n in nodes}})
                node_poi_count[poi["poi_type"]].update({"weight": poi["poi_weight"]})
            node_poi_count[poi["poi_type"]][poi["node"]] += 1
        return node_poi_count

    def _compute_node_values(self, s_nodes: List, d_nodes: List, dist_matrix) -> Dict:
        """
        Computes locational value of a set of source nodes by their distance 
        to a given set of destination nodes.

        Parameters:
        s_nodes: List of nodes for which to compute the values.
        d_nodes: List of nodes for which the distance to the source nodes is considered
                 when computing the value.
        dist_matrix: Matrix where dist_matrix[s_node][d_node] is the distance from s_node
                     to d_node.

        Returns:
        node_values: Dictionary where node_values[n] is the value of node n.       
        """
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
    
    def _compute_vehicle_route(self, vehicle_pos: Union[int, tuple], dest_node: int) -> List[int]:
        """
        Computes the route of a vehicle given its position and destination. If the vehicle is
        at a node, the first node in the route is that node, otherwise the first node in the 
        route is the next node that the vehicle will reach.

        Parameters:
        vehicle_pos: Vehicle position given as a node if the vehicle is at a node or as an edge
                     and distance along the edge otherwise.
        dest_node: 
        """
        is_edge_pos = isinstance(vehicle_pos, tuple)
        vehicle_node = vehicle_pos[0][0] if is_edge_pos else vehicle_pos
        route = [dest_node]
        current_node = dest_node
        while current_node != vehicle_node:
            current_node = self.pred_matrix[vehicle_node][current_node]
            route.append(current_node)
        route.reverse()
        if is_edge_pos and vehicle_pos[0][1] == route[1]:
            route = route[1:]
        return route

    def _compute_route_distances(
            self, 
            vehicle_pos: Union[int, tuple], 
            route: List[int]) -> List[float]:
        """
        Computes time to reach each node along a given route.

        Parameters:
        vehicle_pos: Current position of the vehicle, given as either a node or coordinate along an edge.
        route: List of nodes along the route.

        Returns:
        distances: List of times it will take to reach each node in the route.
        """
        cumulative_time = 0.0

        # If the vehicle is currently on an edge (not at a node), account for the
        # time it takes to reach the first node in the route.
        if isinstance(vehicle_pos, tuple):
            edge, coord = vehicle_pos
            source, target = edge[0], edge[1]
            edge_data = self.graph[source][target]
            speed = min(self.vehicle_speed, edge_data['speed'])
            if route[0] == target:
                # Vehicle is heading towards the target; only the remaining
                # distance to the target node is left to travel.
                remaining_distance = max(edge_data['length'] - coord, 0.0)
            else:
                # Route starts at the source node; the vehicle has to travel
                # back the distance it already covered on the edge.
                remaining_distance = coord
            cumulative_time += remaining_distance / speed

        distances = [cumulative_time]

        # Accumulate the travel time of every subsequent edge along the route.
        for u, v in zip(route[:-1], route[1:]):
            edge_data = self.graph[u][v]
            speed = min(self.vehicle_speed, edge_data['speed'])
            cumulative_time += edge_data['length'] / speed
            distances.append(cumulative_time)

        return distances


    def _compute_vehicle_values(self, obs: Dict) -> Dict:
        """
        Computes the locational value for each vehicle.

        Parameters:
        obs: Observation of the current environment state.

        Returns:
        vehicle_values: Dictionary where vehicle_values[v] is the value of vehicle v.
        """
        nodes = list(self.graph.nodes)
        vehicle_dist_matrix = np.empty((len(self.vehicle_agents), len(nodes)))
        for i, vehicle in enumerate(self.vehicle_agents):
            dest = obs[vehicle]['vehicle_destinations'][vehicle]
            pos = obs[vehicle]['my_position']
            if dest is None:
                node = self._position_to_node(pos)
                vehicle_dist_matrix[i] = self.walk_dist_matrix[node]
                continue
            route = self._compute_vehicle_route(pos, dest)
            distances = self._compute_route_distances(pos, route)
            vehicle_dist_matrix[i] = np.min(np.array(distances)[:, np.newaxis] + self.walk_dist_matrix[route], axis=0)
        vehicle_nodes = [i for i in range(len(self.vehicle_agents))]
        vehicle_values = self._compute_node_values(vehicle_nodes, nodes, vehicle_dist_matrix)
        vehicle_values = {
            self.vehicle_agents[i]: value for i, value in vehicle_values.items()
        }
        return vehicle_values

    def _position_to_node(self, pos):
        """Map an environment position to a node."""
        if isinstance(pos, tuple):
            edge, coord = pos
            _ = coord
            if edge and len(edge) >= 2:
                return edge[0]
            return None
        return pos

    def _compute_utility(self, obs: Dict, vehicle_values: Dict):
        utitlity = 0
        for human in self.human_agents:
            vehicle = obs[human]['human_aboard'][human]
            if vehicle is not None and vehicle in vehicle_values:
                value = vehicle_values[vehicle]
            else:
                pos = obs[human]["my_position"]
                node = self._position_to_node(pos)
                value = self.node_values[node]
            value = max(value, self.epsilon)
            utitlity += value ** (-self.xi)
        utitlity = -(utitlity**self.eta)
        return utitlity

    def initialize(self, graph: nx.DiGraph, vehicle_agents: List, human_agents: List):
        self.vehicle_agents = vehicle_agents
        self.human_agents = human_agents
        self.graph = graph
        self.dist_matrix, self.pred_matrix = self._compute_dist_and_pred_matrix(graph)
        self.dist_matrix = self.dist_matrix / 10
        self.walk_dist_matrix = self.dist_matrix / self.human_speed
        nodes = list(graph.nodes)
        self.node_poi_count = self._compute_poi_counts(nodes, graph.graph["poi_records"])
        self.node_values = self._compute_node_values(nodes, nodes, self.walk_dist_matrix)


    def reward(
            self, 
            obs_dict: Dict, 
            actions_dict: Dict, 
            next_reward_time = None, 
            current_reward_time = None) -> Dict[str, float]:
        _ = obs_dict, actions_dict

        vehicle_values = self._compute_vehicle_values(obs_dict)

        total_reward = self._compute_utility(obs_dict, vehicle_values)

        if next_reward_time is not None and current_reward_time is not None:
            delta_t = max(
                0.0,
                float(next_reward_time) - float(current_reward_time),
            )
            total_reward = total_reward * delta_t

        reward_per_vehicle = total_reward / float(len(self.vehicle_agents))
        return {agent: reward_per_vehicle for agent in self.vehicle_agents}
