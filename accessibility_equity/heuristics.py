from typing import Dict
from typing import Union
from networkx.algorithms.approximation import traveling_salesman_problem as tsp
from networkx.algorithms.approximation import simulated_annealing_tsp
from accessibility_equity.wrappers.gym_wrapper import TransportGymWrapper
from accessibility_equity.wrappers.dqn_wrapper import DQNTransportWrapper


class TSPVehicleAgent:
    def __init__(self, env: TransportGymWrapper | DQNTransportWrapper):
        if type(env) is TransportGymWrapper:
            self.env = env
            self.env_type = 'Gym'
        elif type(env) is DQNTransportWrapper:
            self.env = env.base_env
            self.env_type = 'DQN'
        else:
            raise TypeError(f"TSPVehicleAgent needs environment of type TransportGymWrapper or DQNTransportWrapper, not {type(env)}.")
        self.route = tsp(self.env.env.network, method=simulated_annealing_tsp, init_cycle='greedy')[:-1]
        self.route_dict = {
            self.route[i]: self.route[(i + 1) % len(self.route)]
            for i in range(len(self.route))
        }

    def get_step_type(self, obs):
        if self.env_type == 'Gym':
            return obs["step_type"]
        if self.env_type == 'DQN':
            if type(obs) is dict:
                return obs['features'][0]
            return obs[0]

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

    def get_action(self, obs):
        step_type = self.get_step_type(obs)
        action = 0
        if step_type == 0:
            action = self.next_node() + 1
        elif step_type == 3:
            action = self.depart(self.next_node())
        
        if self.env_type == 'Gym':
            return [action]
        elif self.env_type == 'DQN':
            return action 
