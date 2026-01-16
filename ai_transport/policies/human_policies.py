"""
Human agent policies for AI Transport environment.

Provides abstract base class and concrete implementations for human decision-making.
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Set, Tuple, List

# python
def apply_wrong_vehicle_boarding(
    available_vehicles: Dict[int, str],
    best_vehicle_action: int,
    best_vehicle_id: str,
    wrong_vehicle_prob: float,
    rng: np.random.RandomState
) -> Tuple[int, Optional[str]]:
    """
    Apply probability of boarding wrong vehicle by mistake.

    With probability wrong_vehicle_prob, human boards a random vehicle
    instead of the best one.

    Args:
        available_vehicles: Dict mapping action_idx to vehicle_id
        best_vehicle_action: Action index of best vehicle
        best_vehicle_id: ID of best vehicle to board
        wrong_vehicle_prob: Probability of boarding wrong vehicle (0.0 to 1.0)
        rng: Random number generator instance

    Returns:
        Tuple of (action_idx, message)
        - action_idx: Action to take (best or wrong vehicle)
        - message: Explanation, or None if using best vehicle
    """
    if wrong_vehicle_prob <= 0 or rng.random() >= wrong_vehicle_prob:
        # Board correct vehicle
        return best_vehicle_action, None

    # Choose random vehicle from available options
    vehicle_actions = list(available_vehicles.keys())
    if not vehicle_actions:
        return best_vehicle_action, None

    wrong_action = rng.choice(vehicle_actions)
    wrong_vehicle_id = available_vehicles[wrong_action]

    message = f"Mistakenly boarding {wrong_vehicle_id} instead of {best_vehicle_id}"
    return wrong_action, message


def apply_wrong_station_unboarding(
    current_node: int,
    planned_exit_node: Optional[int],
    target_nodes: Set[int],
    action_mapping: Dict[str, Any],
    action_space_size: int,
    wrong_station_prob: float,
    rng: np.random.RandomState
) -> Tuple[Optional[int], Optional[str]]:
    """
    Apply probability of unboarding at wrong station by mistake.

    With probability wrong_station_prob, human unboards at a random node
    instead of the planned exit node or target.

    Args:
        current_node: Current node ID
        planned_exit_node: Node human planned to exit at (or None)
        target_nodes: Set of target node IDs
        action_mapping: Action mapping dict
        action_space_size: Size of action space
        wrong_station_prob: Probability of unboarding at wrong station (0.0 to 1.0)
        rng: Random number generator instance

    Returns:
        Tuple of (action_idx or None, message or None)
        - action_idx: 1 (unboard) if making mistake, None otherwise
        - message: Explanation if making mistake, None otherwise
    """
    if wrong_station_prob <= 0 or rng.random() >= wrong_station_prob:
        # Don't make mistake
        return None, None

    # Check if should unboard (at target or planned exit)
    should_unboard = (current_node in target_nodes or
                      (planned_exit_node is not None and current_node == planned_exit_node))

    if should_unboard:
        # Would normally unboard anyway, no mistake possible
        return None, None

    # Make mistake: unboard at wrong station
    message = f"Mistakenly unboarding at node {current_node} (not planned exit)"
    return 1, message


def apply_suboptimal_walking(
    current_node: int,
    target_nodes: Set[int],
    action_mapping: Dict[str, Any],
    action_space_size: int,
    walking_edges: Dict[int, Tuple[int, int]],
    suboptimal_walk_prob: float,
    rng: np.random.RandomState
) -> Tuple[Optional[int], Optional[str]]:
    """
    Apply probability of walking in suboptimal direction by mistake.

    With probability suboptimal_walk_prob, human chooses a random outgoing edge
    instead of the best edge toward target.

    Args:
        current_node: Current node ID
        target_nodes: Set of target node IDs
        action_mapping: Action mapping dict
        action_space_size: Size of action space
        walking_edges: Dict mapping action_idx to (source, target) edge tuple
        suboptimal_walk_prob: Probability of suboptimal walking (0.0 to 1.0)
        rng: Random number generator instance

    Returns:
        Tuple of (action_idx or None, message or None)
        - action_idx: Random action if making mistake, None otherwise
        - message: Explanation if making mistake, None otherwise
    """
    if suboptimal_walk_prob <= 0 or rng.random() >= suboptimal_walk_prob:
        # Don't make mistake
        return None, None

    # Get all outgoing edges from current node
    outgoing_actions = []
    outgoing_edges = []

    for action_idx in range(1, action_space_size):
        edge = walking_edges.get(action_idx)
        if edge and isinstance(edge, tuple) and edge[0] == current_node:
            outgoing_actions.append(action_idx)
            outgoing_edges.append(edge)

    if not outgoing_actions:
        return None, None

    # Choose random edge instead of best one
    random_idx = rng.randint(0, len(outgoing_actions))
    random_action = outgoing_actions[random_idx]
    random_edge = outgoing_edges[random_idx]

    message = f"Walking in suboptimal direction via edge {random_edge}"
    return random_action, message



class HumanPolicy(ABC):
    """Abstract base class for human agent policies."""
    
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        """
        Initialize human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            seed: Random seed for reproducibility
        """
        self.agent_id = agent_id
        self.rng = np.random.RandomState(seed)
    
    @abstractmethod
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action for the current observation.
        
        Args:
            observation: Current observation for the agent
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset policy state (e.g., target destination)."""
        pass


class RandomHumanPolicy(HumanPolicy):
    """
    Completely random policy with configurable passing probabilities by step type.
    
    The agent passes with a certain probability depending on the current step type,
    and otherwise takes a random action from the available options.
    """
    
    def __init__(
        self,
        agent_id: str,
        pass_prob_routing: float = 1.0,
        pass_prob_unboarding: float = 0.8,
        pass_prob_boarding: float = 0.7,
        pass_prob_departing: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize random human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            pass_prob_routing: Probability of passing in routing step (default: 1.0, humans can't act)
            pass_prob_unboarding: Probability of passing in unboarding step (default: 0.8)
            pass_prob_boarding: Probability of passing in boarding step (default: 0.7)
            pass_prob_departing: Probability of passing in departing step (default: 0.5)
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.pass_probs = {
            'routing': pass_prob_routing,
            'unboarding': pass_prob_unboarding,
            'boarding': pass_prob_boarding,
            'departing': pass_prob_departing
        }
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get random action with step-type-specific passing probability.
        
        Args:
            observation: Current observation for the agent (must contain 'step_type')
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        pass_prob = self.pass_probs.get(step_type, 0.5)
        
        # Decide whether to pass
        if self.rng.random() < pass_prob:
            return 0, "Passing (random choice)"
        
        # Otherwise, take random action from non-pass options
        if action_space_size <= 1:
            return 0, "Passing (only option)"
        
        action = self.rng.randint(1, action_space_size)
        
        # Get action description from action_mapping if available
        action_mapping = observation.get('action_mapping', {})
        desc = action_mapping.get('description', {}).get(action, 'unknown')
        detail = action_mapping.get('details', {}).get(action, '')
        
        if desc == 'unboard':
            justification = "Unboarding (random choice)"
        elif desc == 'board_vehicle':
            justification = f"Boarding {detail} (random choice)"
        elif desc == 'depart_edge':
            justification = f"Walking to edge {detail} (random choice)"
        else:
            justification = f"Action {action} (random choice)"
            
        return action, justification
    
    def reset(self):
        """Reset policy state (no state to reset for random policy)."""
        pass


class TargetDestinationHumanPolicy(HumanPolicy):
    """
    Policy where human has a target destination and boards vehicles heading that direction.
    
    The human:
    - Has a target destination (node) that changes with some probability rate per real time
    - At boarding steps, boards the vehicle whose destination is closest in Euclidean distance
      to the human's target destination
    - At departing steps, walks toward the target if no suitable vehicle is available
    """
    
    def __init__(
        self,
        agent_id: str,
        network,
        target_change_rate: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize target destination human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            network: NetworkX graph with node coordinates (x, y attributes)
            target_change_rate: Probability rate per second of changing target destination
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.network = network
        self.target_change_rate = target_change_rate
        self.target = None
        self.last_real_time = 0.0
        self.nodes = list(network.nodes())
        
        # Pre-compute node coordinates for distance calculations
        self.node_coords = {}
        for node in self.nodes:
            self.node_coords[node] = (
                network.nodes[node].get('x', 0.0),
                network.nodes[node].get('y', 0.0)
            )

        # Set initial target destination
        self._update_target(0.0)

    
    def _euclidean_distance(self, node1, node2) -> float:
        """Compute Euclidean distance between two nodes."""
        x1, y1 = self.node_coords.get(node1, (0, 0))
        x2, y2 = self.node_coords.get(node2, (0, 0))
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def _update_target(self, real_time: float):
        """Update target destination based on time elapsed and change rate."""
        if self.target is None:
            # Initialize target
            self.target = self.rng.choice(self.nodes)
            self.last_real_time = real_time
            return
        
        # Check if target should change based on time elapsed
        time_elapsed = real_time - self.last_real_time
        change_prob =1.0 - np.exp(-self.target_change_rate * time_elapsed)

        if (self.rng.random() < change_prob):

            # Change to new random target
            self.target = self.rng.choice(self.nodes)
            self.last_real_time = real_time
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action based on target destination.
        
        For boarding: choose vehicle whose destination is closest to target
        For departing: choose edge leading toward target
        Otherwise: pass
        
        Args:
            observation: Current observation containing step_type, real_time, action_mapping
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        real_time = observation.get('real_time', 0.0)
        action_mapping = observation.get('action_mapping', {})
        
        # Update target destination
        self._update_target(real_time)

        if step_type == 'unboarding':
            return self._get_unboarding_action(observation, action_mapping, action_space_size)
        elif step_type == 'boarding':
            return self._get_boarding_action(observation, action_mapping, action_space_size)
        elif step_type == 'departing':
            return self._get_departing_action(observation, action_mapping, action_space_size)
        else:
            # Routing or on edge - pass
            if step_type == 'routing':
                return 0, "Passing (no action in routing step)"
            else:
                return 0, "Passing"

    def _get_unboarding_action(self, observation: Dict, action_mapping: Dict, action_space_size: int):

        if action_space_size <= 1:
            return 0, "Passing (no unboard option)"

        pos = observation.get("agent_positions", {}).get(self.agent_id)
        if isinstance(pos, tuple):
            return 0, "Passing (on edge)"

        current_node = pos if pos is not None else None
        if current_node is None:
            return 0, "Passing (position unknown)"

        # 检查是否在车上
        human_aboard = observation.get('human_aboard', {})
        if not human_aboard.get(self.agent_id):
            return 0, "Passing (not aboard)"

        # 到达目标就下车（self.target 是这个 policy 的目标）
        print('TargetDestinationHumanPolicy unboarding check: agent_id=', self.agent_id, ' current_node=', current_node,
              ', target=', self.target,)
        if self.target is not None and current_node == self.target:
            return 1, f"Unboarding (arrived at target {self.target})"

        return 0, f"Staying aboard (not at target {self.target})"

    def _get_boarding_action(self, observation: Dict, action_mapping: Dict, action_space_size: int):
        """Choose vehicle whose destination is closest to target."""
        if action_space_size <= 1 or self.target is None:
            return 0, "Passing (no target or no vehicles available)"

        agent_positions = observation.get('agent_positions', {})
        my_pos = agent_positions.get(self.agent_id)
        if not isinstance(my_pos, tuple) and self.target is not None and my_pos == self.target:
            return 0, f"Passing (already at target {self.target})"
        
        # Get vehicle destinations from observation
        vehicle_destinations = observation.get('vehicle_destinations', {})
        details = action_mapping.get('details', {})
        
        best_action = 0
        best_distance = float('inf')
        best_vehicle = None
        
        for action_idx in range(1, action_space_size):
            vehicle_id = details.get(action_idx)
            if vehicle_id:
                vehicle_dest = vehicle_destinations.get(vehicle_id)
                if vehicle_dest is not None:
                    distance = self._euclidean_distance(vehicle_dest, self.target)
                    if distance < best_distance:
                        best_distance = distance
                        best_action = action_idx
                        best_vehicle = vehicle_id
        
        if best_action > 0:
            return best_action, f"Boarding {best_vehicle} (heading toward target {self.target}, distance {best_distance:.1f})"
        else:
            return 0, f"Passing (no vehicles heading toward target {self.target})"

    def _get_departing_action(self, observation: Dict, action_mapping: Dict, action_space_size: int):
        """Choose edge leading toward target destination."""
        if action_space_size <= 1 or self.target is None:
            return 0, "Passing (no target or no edges available)"
        
        my_position = observation.get("agent_positions", {}).get(self.agent_id)#observation.get('my_position')
        if isinstance(my_position, tuple):
            return 0, "Passing (already on edge)"
        
        # Get current node
        current_node = my_position
        print('TargetDestinationHumanPolicy unboarding check: agent_id=', self.agent_id, ' current_node=', current_node,
              ', target=', self.target,'my_position=',my_position)

        if current_node == self.target:
            return 0, f"Passing (already at target {self.target})"
        
        # Find edge that leads toward target
        details = action_mapping.get('details', {})
        best_action = 0
        best_distance = float('inf')
        best_edge = None
        
        for action_idx in range(1, action_space_size):
            edge = details.get(action_idx)
            if edge and isinstance(edge, tuple):
                # Edge is (source, target)
                target_node = edge[1]
                distance = self._euclidean_distance(target_node, self.target)
                if distance < best_distance:
                    best_distance = distance
                    best_action = action_idx
                    best_edge = edge
        
        if best_action > 0:
            return best_action, f"Walking toward target {self.target} via edge {best_edge}"
        else:
            return 0, f"Passing (no edge toward target {self.target})"

    def reset(self):
        """Reset policy state (target destination and time)."""
        self.target = None
        self.last_real_time = 0.0


class HeuristicRoutingHumanPolicy(HumanPolicy):
    """
    Heuristic policy where human uses vehicles to get closer to target nodes.
    
    The human:
    - Has a set of target nodes they want to reach (can be a single node or multiple)
    - At current node x, considers all vehicles v at x with announced destinations d_v
    - Computes shortest duration path (considering road length and speed) from x to d_v
    - Considers all nodes z on any of these paths
    - Finds z closest to target (in terms of walking time from z to target)
    - If z is closer to target than x, boards vehicle that reaches z fastest
    - If z is not closer, either waits (with probability p_wait) or walks toward target
    """
    
    def __init__(
        self,
        agent_id: str,
        network,
        target_nodes: Set[int],
        p_wait: float = 0.5,
        wrong_vehicle_prob: float = 0.0,
        wrong_station_prob: float = 0.0,
        suboptimal_walk_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize heuristic routing human policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            network: NetworkX graph with edge 'length' and 'speed' attributes
            target_nodes: Set of target node IDs the human wants to reach
            p_wait: Probability of waiting at node when no good vehicle available (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.network = network
        self.target_nodes = target_nodes
        self.p_wait = p_wait
        self.wrong_vehicle_prob = wrong_vehicle_prob
        self.wrong_station_prob = wrong_station_prob
        self.suboptimal_walk_prob = suboptimal_walk_prob
        self.nodes = list(network.nodes())
        
        # Track previous node for intelligent unboarding
        self.previous_node = None
        # Track the planned exit node (the best_z chosen when boarding)
        self.planned_exit_node = None
        
        # Create duration graph for computing shortest duration paths (for vehicles)
        # Duration = length / speed
        self.duration_graph = nx.DiGraph()
        for u, v, data in network.edges(data=True):
            length = data.get('length', 1.0)
            speed = data.get('speed', 1.0)
            duration = length / speed if speed > 0 else float('inf')
            self.duration_graph.add_edge(u, v, weight=duration)
    
    def _normalize_node_id(self, node):
        """
        Normalize a node ID to ensure it's hashable and compatible with NetworkX.
        Converts numpy types to Python types.
        
        Args:
            node: Node ID (may be numpy type or Python type)
            
        Returns:
            Normalized node ID
        """
        if hasattr(node, 'item'):  # numpy scalar
            return node.item()
        return node
    
    def _compute_shortest_duration_path(self, source: int, target: int) -> Optional[List[int]]:
        """
        Compute shortest duration path from source to target.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of nodes in path, or None if no path exists
        """
        if source == target:
            return [source]
        
        try:
            path = nx.shortest_path(self.duration_graph, source, target, weight='weight')
            return path
        except nx.NetworkXNoPath:
            return None
    
    def _compute_path_duration(self, path: List[int]) -> float:
        """
        Compute total duration of a path.
        
        Args:
            path: List of nodes in path
            
        Returns:
            Total duration in time units
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_duration = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if self.duration_graph.has_edge(u, v):
                total_duration += self.duration_graph[u][v]['weight']
            else:
                return float('inf')
        
        return total_duration
    
    def _compute_walking_distance_to_target(self, node, walking_speed: float) -> float:
        """
        Compute shortest walking time from node to any target node.
        
        Uses actual road network with walking speed (not vehicle speeds).
        
        Args:
            node: Node ID
            walking_speed: Human's walking speed
            
        Returns:
            Minimum walking time to reach any target node
        """
        # Normalize node ID
        node = self._normalize_node_id(node)
        
        if node in self.target_nodes:
            return 0.0
        
        if walking_speed <= 0:
            return float('inf')
        
        min_time = float('inf')
        
        for target in self.target_nodes:
            # Normalize target ID
            target = self._normalize_node_id(target)
            
            # Compute shortest path based on edge lengths (not vehicle travel times)
            try:
                path = nx.shortest_path(self.network, node, target, weight='length')
                
                # Validate path is a list
                if not isinstance(path, list):
                    continue
                if len(path) < 2:
                    continue
                
                # Compute walking time: sum of (edge_length / walking_speed)
                walking_time = 0.0
                for i in range(len(path) - 1):
                    try:
                        u = path[i]
                        v = path[i + 1]
                    except (IndexError, KeyError, TypeError) as e:
                        # Skip if we can't access path elements
                        break
                    
                    # Normalize node IDs - handle various types
                    if isinstance(u, (list, tuple, dict)):
                        # Skip if u is a complex type (shouldn't happen but be defensive)
                        continue
                    if isinstance(v, (list, tuple, dict)):
                        # Skip if v is a complex type (shouldn't happen but be defensive)
                        continue
                    
                    u = self._normalize_node_id(u)
                    v = self._normalize_node_id(v)
                    
                    if self.network.has_edge(u, v):
                        edge_length = self.network[u][v].get('length', 1.0)
                        walking_time += edge_length / walking_speed
                
                min_time = min(min_time, walking_time)
            except nx.NetworkXNoPath:
                continue
            except (TypeError, ValueError, KeyError, IndexError) as e:
                # Handle any type conversion or access errors
                continue
        
        return min_time
    
    def _get_nodes_on_path(self, path: Optional[List[int]]) -> Set[int]:
        """
        Get all nodes on a path.
        
        Args:
            path: List of nodes in path, or None
            
        Returns:
            Set of node IDs on the path
        """
        if path is None:
            return set()
        return set(path)
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action based on heuristic routing strategy.
        
        For boarding: board vehicle that gets human closest to target fastest
        For departing: walk toward target if no good vehicle, or wait
        Otherwise: pass
        
        Args:
            observation: Current observation containing step_type, agent_positions, etc.
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        
        if step_type == 'boarding':
            return self._get_boarding_action(observation, action_space_size)
        elif step_type == 'departing':
            return self._get_departing_action(observation, action_space_size)
        elif step_type == 'unboarding':
            return self._get_unboarding_action(observation, action_space_size)
        else:
            # Routing or on edge - pass
            if step_type == 'routing':
                return 0, "Passing (no action in routing step)"
            else:
                return 0, "Passing"
    
    def _get_boarding_action(self, observation: Dict, action_space_size: int) -> Tuple[int, str]:
        """
        Decide which vehicle to board (if any) based on heuristic.
        
        The human:
        1. Finds all vehicles at current node with announced destinations
        2. Computes shortest duration paths from current node to vehicle destinations
        3. Collects all nodes Z on any of these paths
        4. Finds Z closest to target (in walking time)
        5. If Z is closer than current position, boards vehicle that reaches Z fastest
        6. Otherwise passes
        """
        if action_space_size <= 1:
            return 0, "Passing (no vehicles available)"
        
        # Get current position (must be a node for boarding step)
        # Handle both 'full' and 'local' observation scenarios
        my_position = observation.get('my_position')
        if my_position is None:
            # Full observation scenario - extract from agent_positions
            agent_positions = observation.get('agent_positions', {})
            my_position = agent_positions.get(self.agent_id)
        
        if isinstance(my_position, tuple):
            return 0, "Passing (not at node)"
        
        if my_position is None:
            return 0, "Passing (position unknown)"
        
        current_node = my_position
        
        # Get agent attributes to find walking speed
        agent_attributes = observation.get('agent_attributes', {})
        my_attributes = agent_attributes.get(self.agent_id, {})
        walking_speed = my_attributes.get('speed', 1.0)
        
        # Get vehicle destinations
        vehicle_destinations = observation.get('vehicle_destinations', {})
        action_mapping = observation.get('action_mapping', {})
        details = action_mapping.get('details', {})
        
        # Compute walking distance from current node to target
        current_to_target_time = self._compute_walking_distance_to_target(current_node, walking_speed)
        
        # Find all vehicles at current node and collect nodes on their paths
        vehicle_info = []  # List of (vehicle_id, destination, path, nodes_on_path)
        all_candidate_nodes = set()
        available_vehicles = {}  # 新增：记录所有可用车辆
        
        for action_idx in range(1, action_space_size):
            vehicle_id = details.get(action_idx)
            if vehicle_id:
                destination = vehicle_destinations.get(vehicle_id)
                if destination is not None:
                    # Compute shortest duration path for this vehicle
                    available_vehicles[action_idx] = vehicle_id  # 新增
                    path = self._compute_shortest_duration_path(current_node, destination)
                    if path and isinstance(path, list):  # Validate path is a list
                        nodes_on_path = self._get_nodes_on_path(path)
                        vehicle_info.append((vehicle_id, destination, path, nodes_on_path, action_idx))
                        all_candidate_nodes.update(nodes_on_path)
        
        if not all_candidate_nodes:
            return 0, "Passing (no vehicles with destinations)"
        
        # Find node Z in all_candidate_nodes that is closest to target (in walking time)
        best_z = None
        best_z_to_target_time = float('inf')
        
        for z in all_candidate_nodes:
            z_to_target_time = self._compute_walking_distance_to_target(z, walking_speed)
            if z_to_target_time < best_z_to_target_time:
                best_z_to_target_time = z_to_target_time
                best_z = z
        
        # Check if best_z is closer to target than current node
        if best_z is None or best_z_to_target_time >= current_to_target_time:
            return 0, f"Passing (no node on vehicle paths closer to target than current position)"
        
        # Find vehicle that reaches best_z fastest
        best_vehicle_id = None
        best_vehicle_action = None
        best_time_to_z = float('inf')
        
        for vehicle_id, destination, path, nodes_on_path, action_idx in vehicle_info:
            if best_z in nodes_on_path:
                # Find time for vehicle to reach best_z
                # Get partial path from current_node to best_z
                try:
                    # Validate path is a list before using .index()
                    if not isinstance(path, list):
                        continue
                        
                    z_index = path.index(best_z)
                    partial_path = path[:z_index + 1]
                    time_to_z = self._compute_path_duration(partial_path)
                    
                    if time_to_z < best_time_to_z:
                        best_time_to_z = time_to_z
                        best_vehicle_id = vehicle_id
                        best_vehicle_action = action_idx
                except (ValueError, IndexError, AttributeError):
                    continue

        if best_vehicle_action is None:
            return 0, "Passing (no suitable vehicle found)"
            # 应用坐错车误差
        final_action, wrong_msg = apply_wrong_vehicle_boarding(
                available_vehicles=available_vehicles,
                best_vehicle_action=best_vehicle_action,
                best_vehicle_id=best_vehicle_id,
                wrong_vehicle_prob=self.wrong_vehicle_prob,
                rng=self.rng
            )

        self.planned_exit_node = best_z# 仍然记录原计划的下车站
        if wrong_msg:
            return final_action, wrong_msg
        return best_vehicle_action, f"Boarding {best_vehicle_id} to reach node {best_z} (closer to target, ETA {best_time_to_z:.1f})"

    def _get_departing_action(self, observation: Dict, action_space_size: int) -> Tuple[int, str]:
        """
        Decide whether to walk toward target or wait for vehicles.
        
        If no good vehicles were available at boarding step, the human either:
        - Waits at current node (with probability p_wait)
        - Walks toward target on shortest walking path (with probability 1 - p_wait)
        """
        # Get current position - handle both observation scenarios
        my_position = observation.get('my_position')
        if my_position is None:
            # Full observation scenario - extract from agent_positions
            agent_positions = observation.get('agent_positions', {})
            my_position = agent_positions.get(self.agent_id)
        
        if isinstance(my_position, tuple):
            return 0, "Passing (already on edge)"
        
        if my_position is None:
            return 0, "Passing (position unknown)"
        
        current_node = my_position
        
        # Check if already at target
        if current_node in self.target_nodes:
            return 0, f"Passing (already at target node {current_node})"
        
        if action_space_size <= 1:
            return 0, "Passing (no edges available)"
        
        # Decide whether to wait or walk
        if self.rng.random() < self.p_wait:
            return 0, f"Waiting at node {current_node} for vehicles (p_wait={self.p_wait})"
        
        # Walk toward target - find next node on shortest path to any target
        action_mapping = observation.get('action_mapping', {})
        details = action_mapping.get('details', {})
        
        # Get agent attributes to find walking speed
        agent_attributes = observation.get('agent_attributes', {})
        my_attributes = agent_attributes.get(self.agent_id, {})
        walking_speed = my_attributes.get('speed', 1.0)
        
        # Find best next node (closest to any target in walking time)
        best_action = 0
        best_edge = None
        best_walking_time = float('inf')
        
        for action_idx in range(1, action_space_size):
            edge = details.get(action_idx)
            if edge and isinstance(edge, tuple):
                next_node = edge[1]
                walking_time = self._compute_walking_distance_to_target(next_node, walking_speed)
                if walking_time < best_walking_time:
                    best_walking_time = walking_time
                    best_action = action_idx
                    best_edge = edge
        

        if best_action == 0:
            return 0, "Passing (no edge toward target)"

        # 应用次优行走误差
        subopt_action, subopt_msg = apply_suboptimal_walking(
            current_node=current_node,
            target_nodes=self.target_nodes,
            action_mapping=action_mapping,
            action_space_size=action_space_size,
            walking_edges=details,
            suboptimal_walk_prob=self.suboptimal_walk_prob,
            rng=self.rng
        )

        if subopt_msg:
            return subopt_action, subopt_msg

        return best_action, f"Walking toward target via edge {best_edge}"


        return 0, "Passing (no edge toward target)"
    
    def _get_unboarding_action(self, observation: Dict, action_space_size: int) -> Tuple[int, str]:
        """
        Decide whether to unboard from vehicle.
        
        Intelligent unboarding happens when:
        1. Arrived at target node
        2. Arrived at the planned exit node (the one chosen when boarding)
        3. Vehicle goes wrong direction (distance to target increases)
        4. A better vehicle is available at current node
        
        Args:
            observation: Current observation
            action_space_size: Size of action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        if action_space_size <= 1:
            return 0, "Passing (not aboard vehicle)"
        
        # Get current position - handle both observation scenarios
        my_position = observation.get('my_position')
        if my_position is None:
            # Full observation scenario - extract from agent_positions
            agent_positions = observation.get('agent_positions', {})
            my_position = agent_positions.get(self.agent_id)
        
        # Only unboard at nodes, not on edges
        if isinstance(my_position, tuple):
            return 0, "Passing (on edge)"
        
        if my_position is None:
            return 0, "Passing (position unknown)"
        
        current_node = my_position
        
        # Check if aboard a vehicle
        human_aboard = observation.get('human_aboard', {})
        aboard_vehicle = human_aboard.get(self.agent_id)
        
        if not aboard_vehicle:
            return 0, "Passing (not aboard vehicle)"
        
        # Get agent attributes to find walking speed
        agent_attributes = observation.get('agent_attributes', {})
        my_attributes = agent_attributes.get(self.agent_id, {})
        walking_speed = my_attributes.get('speed', 1.0)
        
        # Compute walking distance from current node to target
        current_to_target = self._compute_walking_distance_to_target(current_node, walking_speed)
        
        # CASE 1: If we're at the target, definitely unboard
        if current_node in self.target_nodes:
            self.previous_node = current_node
            self.planned_exit_node = None
            return 1, f"Unboarding (arrived at target node {current_node})"
        
        # CASE 2: If we're at the planned exit node, unboard
        if self.planned_exit_node is not None and current_node == self.planned_exit_node:
            # 应用下错站误差
            wrong_station_action, wrong_station_msg = apply_wrong_station_unboarding(
                current_node=current_node,
                planned_exit_node=self.planned_exit_node,
                target_nodes=self.target_nodes,
                action_mapping=observation.get('action_mapping', {}),
                action_space_size=action_space_size,
                wrong_station_prob=self.wrong_station_prob,
                rng=self.rng
            )

            if wrong_station_msg:
                self.previous_node = current_node
                return wrong_station_action, wrong_station_msg

            self.previous_node = current_node
            self.planned_exit_node = None
            return 1, f"Unboarding (arrived at planned exit node {current_node})"
        
        # CASE 3: Check if there's a better vehicle available at current node
        # This should be checked BEFORE wrong direction, so humans can switch vehicles
        vehicle_destinations = observation.get('vehicle_destinations', {})
        action_mapping = observation.get('action_mapping', {})
        details = action_mapping.get('details', {})
        
        # Get current vehicle's destination
        current_vehicle_dest = vehicle_destinations.get(aboard_vehicle)
        
        # Find all vehicles at current node (excluding the one we're on)
        for action_idx in range(1, action_space_size):
            vehicle_id = details.get(action_idx)
            if vehicle_id and vehicle_id != aboard_vehicle:
                destination = vehicle_destinations.get(vehicle_id)
                if destination is not None:
                    # Compute path for this alternative vehicle
                    path = self._compute_shortest_duration_path(current_node, destination)
                    if path and isinstance(path, list):
                        nodes_on_path = self._get_nodes_on_path(path)
                        
                        # Find the best node on this vehicle's path
                        best_z_on_alt = None
                        best_z_distance = float('inf')
                        
                        for z in nodes_on_path:
                            z_to_target = self._compute_walking_distance_to_target(z, walking_speed)
                            if z_to_target < best_z_distance:
                                best_z_distance = z_to_target
                                best_z_on_alt = z
                        
                        # Compare with our current situation
                        # If the alternative vehicle gets us closer to target
                        if best_z_on_alt is not None and best_z_distance < current_to_target:
                            # This vehicle is better! Unboard to switch
                            self.previous_node = current_node
                            self.planned_exit_node = None
                            return 1, f"Unboarding (better vehicle {vehicle_id} available, can reach node {best_z_on_alt} closer to target)"
        
        # CASE 4: Check if vehicle is going wrong direction
        if self.previous_node is not None:
            # Compute walking distance from previous node to target
            previous_to_target = self._compute_walking_distance_to_target(self.previous_node, walking_speed)
            
            # If current distance is larger than previous, vehicle is going wrong direction
            if current_to_target > previous_to_target:
                self.previous_node = current_node
                self.planned_exit_node = None
                return 1, f"Unboarding (vehicle going wrong direction: distance increased from {previous_to_target:.1f} to {current_to_target:.1f})"
        
        # Update previous node and stay on vehicle
        self.previous_node = current_node
        return 0, f"Staying aboard (moving toward target, distance: {current_to_target:.1f})"
    
    def reset(self):
        """Reset policy state."""
        self.previous_node = None
        self.planned_exit_node = None
