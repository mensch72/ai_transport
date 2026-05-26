from typing import Dict
from networkx.algorithms.approximation import traveling_salesman_problem as tsp
from accessibility_equity.wrappers.gym_wrapper import TransportGymWrapper


class TSPVehicleAgent:
    def __init__(self, env: TransportGymWrapper):
        self.env = env
        self.route = tsp(env.env.network)[:-1]
        print(self.route)
        self.route_dict = {
            self.route[i]: self.route[(i + 1) % len(self.route)]
            for i in range(len(self.route))
        }

    def current_node(self):
        vehicle_agent = list(self.env.env.vehicle_agents)[0]
        pos = self.env.env.agent_positions[vehicle_agent]
        if type(pos) is tuple:
            return None
        return pos

    def next_node(self):
        current_node = self.current_node()
        if current_node == None:
            return 0
        return self.route_dict[current_node]
    
    def depart(self, next_node):
        current_node = self.current_node()
        out_edges = list(self.env.env.network.out_edges(current_node))
        for i, e in enumerate(out_edges):
            if e[1] == next_node:
                return i + 1
        return 0

    def get_action(self, obs: Dict):
        if obs["step_type"] == 0:
            return [self.next_node()]
        if obs["step_type"] == 3:
            return [self.depart(self.next_node())]
        return [0]
