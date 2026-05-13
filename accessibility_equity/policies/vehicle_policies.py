"""
Vehicle agent policies for AI Transport environment.

Provides abstract base class and concrete implementations for vehicle decision-making.
"""

from abc import ABC, abstractmethod
import numpy as np
import networkx as nx
from typing import Dict, Any, Optional, Tuple


def apply_empty_vehicle_waiting(
    is_empty: bool,
    depart_count: int,
    wait_cycles: int,
    action_space_size: int
) -> Tuple[Optional[int], int, Optional[str]]:
    """
    Apply waiting mechanism for empty vehicles.
    When vehicle is empty and hasn't waited enough cycles,return wait action. Otherwise allow normal action selection.

    Args:
        is_empty: Whether vehicle is currently empty
        depart_count: Current waiting cycle counter
        wait_cycles: Total cycles to wait (0 = no waiting)
        action_space_size: Size of action space (for validation)

    Returns:
        Tuple of (wait_action, updated_depart_count, wait_message)
        - wait_action: 0 (pass) if should wait, None if should proceed
        - updated_depart_count: New counter value
        - wait_message: Justification string, or None if not waiting
    """

    if wait_cycles <= 0:
        return None, depart_count, None

    if not is_empty:
        return None, 0, None

    if depart_count < wait_cycles:
        new_count = depart_count + 1
        message = f"Waiting at node (count {new_count}/{wait_cycles})"
        return 0, new_count, message

    return None, wait_cycles, None

# python
def apply_random_edge(
    current_node: int,
    next_node_on_shortest_path: Optional[int],
    action_mapping: Dict[str, Any],
    action_space_size: int,
    random_edge_prob: float,
    rng: np.random.RandomState
) -> Tuple[Optional[int], Optional[str]]:
    """
    Apply random edge selection noise to departing action.
    With probability random_edge_prob, choose a random outgoing edge instead of the shortest path edge. Otherwise use shortest path.

    Args:
        current_node: Current node ID
        next_node_on_shortest_path: Next node on shortest path (None = at destination)
        action_mapping: Action mapping dict with 'details' key
        action_space_size: Size of action space
        random_edge_prob: Probability of choosing random edge (0.0 to 1.0)
        rng: Random number generator instance

    Returns:
        Tuple of (action_idx, justification_message)
        - action_idx: Action to take, or None if should use shortest path
        - justification_message: Explanation string, or None
    """
    # No noise, use shortest path
    if random_edge_prob <= 0:
        return None, None

    # Use shortest path (within probability)
    if rng.random() >= random_edge_prob:
        return None, None

    # Select random outgoing edge, excluding the shortest-path edge when possible.
    details = action_mapping.get('details', {})
    outgoing_edges = []
    edge_actions = []

    for action_idx in range(1, action_space_size):
        edge = details.get(action_idx)
        if edge and isinstance(edge, tuple) and edge[0] == current_node:
            if next_node_on_shortest_path is not None and edge[1] == next_node_on_shortest_path:
                continue
            outgoing_edges.append(edge)
            edge_actions.append(action_idx)

    # No outgoing edges available
    if not outgoing_edges:
        return None, None

    # Choose random outgoing edge
    random_idx = rng.randint(0, len(outgoing_edges))
    chosen_edge = outgoing_edges[random_idx]
    chosen_action = edge_actions[random_idx]

    message = f"Taking random outgoing edge {chosen_edge} instead of shortest path"
    return chosen_action, message



class VehiclePolicy(ABC):
    """Abstract base class for vehicle agent policies."""
    
    def __init__(self, agent_id: str, seed: Optional[int] = None):
        """
        Initialize vehicle policy.
        
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
        """Reset policy state (e.g., current destination)."""
        pass


class RandomVehiclePolicy(VehiclePolicy):
    """
    Completely random policy for vehicles.
    
    The vehicle takes random actions with configurable passing probabilities.
    """
    
    def __init__(
        self,
        agent_id: str,
        pass_prob_routing: float = 0.3,
        pass_prob_departing: float = 0.2,
        wait_cycles: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize random vehicle policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            pass_prob_routing: Probability of passing (no destination) in routing step
            pass_prob_departing: Probability of passing (staying at node) in departing step
            wait_cycles: Number of departing-step passes an empty vehicle should make
                before taking a non-pass action.
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.pass_prob_routing = pass_prob_routing
        self.pass_prob_departing = pass_prob_departing
        self.wait_cycles = wait_cycles
        self.depart_count = 0

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

        # Determine pass probability based on step type
        if step_type == 'routing':
            pass_prob = self.pass_prob_routing
        elif step_type == 'departing':
            pass_prob = self.pass_prob_departing
            human_aboard = observation.get('human_aboard', {})
            is_empty = all(v != self.agent_id for v in human_aboard.values())
            wait_action, self.depart_count, wait_message = apply_empty_vehicle_waiting(
                is_empty=is_empty,
                depart_count=self.depart_count,
                wait_cycles=self.wait_cycles,
                action_space_size=action_space_size
            )
            if wait_action is not None:
                return wait_action, wait_message
        else:
            pass_prob = 1.0  # Always pass in unboarding/boarding
        
        # Decide whether to pass
        if self.rng.random() < pass_prob:
            if step_type == 'routing':
                return 0, "Passing (keeping current destination, random choice)"
            elif step_type == 'departing':
                return 0, "Passing (staying at node, random choice)"
            else:
                return 0, "Passing (no action in this step)"
        
        # Otherwise, take random action from non-pass options
        if action_space_size <= 1:
            return 0, "Passing (only option)"
        
        action = self.rng.randint(1, action_space_size)
        
        # Get action description from action_mapping if available
        action_mapping = observation.get('action_mapping', {})
        desc = action_mapping.get('description', {}).get(action, 'unknown')
        detail = action_mapping.get('details', {}).get(action, '')
        
        if desc == 'set_destination_none':
            justification = "Clearing destination (random choice)"
        elif desc == 'set_destination_node':
            justification = f"Setting destination to node {detail} (random choice)"
        elif desc == 'depart_edge':
            justification = f"Departing to edge {detail} (random choice)"
        else:
            justification = f"Action {action} (random choice)"
            
        return action, justification
    
    def reset(self):
        """Reset policy state (no state to reset for random policy)."""
        self.depart_count = 0


class ShortestPathVehiclePolicy(VehiclePolicy):
    """
    Policy where vehicle takes fastest path to destination.
    
    The vehicle:
    - Takes the fastest path to its current destination
    - When reaching destination, chooses new destination with probability proportional to Euclidean distance from current node
    - Uses edge speed to determine fastest path
    """
    
    def __init__(
        self,
        agent_id: str,
        network,
        wait_cycles: int = 0,
        random_edge_prob: float = 0.0,
        seed: Optional[int] = None
    ):
        """
        Initialize shortest path vehicle policy.
        
        Args:
            agent_id: The ID of the agent this policy controls
            network: NetworkX graph with edge 'speed' attribute
            seed: Random seed for reproducibility
        """
        super().__init__(agent_id, seed)
        self.network = network
        self.wait_cycles = wait_cycles
        self.random_edge_prob = random_edge_prob
        self.depart_count = 0
        self.current_destination = None
        self.nodes = list(network.nodes())
        
        # Pre-compute node coordinates for distance calculations
        self.node_coords = {}
        for node in self.nodes:
            self.node_coords[node] = (
                network.nodes[node].get('x', 0.0),
                network.nodes[node].get('y', 0.0)
            )
        
        # Create travel time graph (weight = length / speed)
        self.time_graph = nx.DiGraph()
        for u, v, data in network.edges(data=True):
            length = data.get('length', 1.0)
            speed = data.get('speed', 1.0)
            travel_time = length / speed if speed > 0 else float('inf')
            self.time_graph.add_edge(u, v, weight=travel_time)
    
    def _euclidean_distance(self, node1, node2) -> float:
        """Compute Euclidean distance between two nodes."""
        x1, y1 = self.node_coords.get(node1, (0, 0))
        x2, y2 = self.node_coords.get(node2, (0, 0))
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def _travel_time(self, src, dst):
        if src == dst:
            return 0.0
        try:
            return nx.shortest_path_length(self.time_graph, src, dst, weight='weight')
        except nx.NetworkXNoPath:
            return float('inf')

    def _choose_service_destination(self, current_node, aboard_humans, human_destinations):
        best_dest = None
        best_time = -1.0
        for h in aboard_humans:
            d = human_destinations.get(h)
            if d is None:
                continue
            #choose a best destination if multiple are given
            if isinstance(d, (set, list, tuple)):
                best_d_for_h = None
                best_dist_for_h = -1.0
                for cand in d:
                    dist_cand = self._euclidean_distance(current_node, cand)
                    # choose the farthest one
                    if np.isfinite(dist_cand) and dist_cand > best_dist_for_h:
                        best_dist_for_h = dist_cand
                        best_d_for_h = cand
                d = best_d_for_h
                dist = best_dist_for_h
            else:
                dist = self._euclidean_distance(current_node, d)
            if np.isfinite(dist) and dist > best_time:
                best_time = dist
                best_dest = d
        return best_dest

    def _choose_cruise_destination(self, current_node) -> Optional[int]:
        """
        Choose new destination with probability proportional to distance.
        Args:
            current_node: Current node ID
        Returns:
            New destination node ID
        """
        # Get all other nodes
        other_nodes = [n for n in self.nodes if n != current_node]
        if not other_nodes:
            return None

        # Compute distances
        distances = np.array([self._euclidean_distance(current_node, n) for n in other_nodes])
        
        # Avoid division by zero
        distances = distances + 1e-6
        
        # Compute probabilities proportional to distance
        probabilities = distances / distances.sum()
        
        # Sample new destination
        return self.rng.choice(other_nodes, p=probabilities)
    
    def _get_next_node_on_path(self, current_node, destination) -> Optional[int]:
        """
        Get next node on shortest (fastest) path to destination.
        
        Args:
            current_node: Current node ID
            destination: Destination node ID
            
        Returns:
            Next node on path, or None if no path exists
        """
        if current_node == destination:
            return None
        
        try:
            path = nx.shortest_path(self.time_graph, current_node, destination, weight='weight')
            if len(path) >= 2:
                return path[1]  # Next node after current
        except nx.NetworkXNoPath:
            pass
        
        return None
    
    def get_action(self, observation: Dict[str, Any], action_space_size: int):
        """
        Get action based on shortest path to destination.
        
        For routing: set or update destination
        For departing: choose edge on shortest path to destination
        Otherwise: pass
        
        Args:
            observation: Current observation containing step_type, my_position, action_mapping
            action_space_size: Size of the action space
            
        Returns:
            Tuple of (action_index, justification_string)
        """
        step_type = observation.get('step_type', 'departing')
        
        if step_type == 'routing':
            return self._get_routing_action(observation, action_space_size)
        elif step_type == 'departing':
            return self._get_departing_action(observation, action_space_size)
        else:
            # Unboarding, boarding, or on edge - pass
            if step_type == 'unboarding':
                return 0, "Passing (no action in unboarding step)"
            elif step_type == 'boarding':
                return 0, "Passing (no action in boarding step)"
            else:
                return 0, "Passing"
    
    def _get_routing_action(self, observation: Dict, action_space_size: int):
        """Set or update destination."""
        my_position = observation.get('my_position')
        if my_position is None:
            my_position = observation.get('agent_positions', {}).get(self.agent_id)
        action_mapping = observation.get('action_mapping', {})
        
        # If on edge, can't act
        if isinstance(my_position, tuple):
            return 0, "Passing (on edge)"
        if my_position is None:
            return 0, "Passing (position unavailable)"
        
        current_node = my_position

        human_aboard = observation.get("human_aboard", {})
        human_destinations = observation.get("human_destinations", {})
        aboard_humans = [h for h, v in human_aboard.items() if v == self.agent_id]


        # Check if we've reached current destination
        if self.current_destination is None or current_node == self.current_destination:
            # Choose new destination
            if aboard_humans:
                self.current_destination = self._choose_service_destination(
                    current_node=current_node,
                    aboard_humans=aboard_humans,
                    human_destinations=human_destinations
                )
                if self.current_destination is None:
                    # If human destinations are unknown, fallback to staying or cruising
                    return 0, "Passing (passengers aboard but destinations unknown)"
            else:
                self.current_destination = self._choose_cruise_destination(current_node)
        
        # Find action that sets destination to our target
        if self.current_destination is not None:
            details = action_mapping.get('details', {})
            for action_idx in range(1, action_space_size):
                if details.get(action_idx) == self.current_destination:
                    distance = self._euclidean_distance(current_node, self.current_destination)
                    return action_idx, f"Setting destination to node {self.current_destination} (distance-weighted choice, {distance:.1f} units)"
        
        # Default: pass (or set to None)
        return 0, "Passing (no destination or already set)"
    
    def _get_departing_action(self, observation: Dict, action_space_size: int):
        """Choose edge on shortest path to destination."""
        agent_positions = observation.get("agent_positions")
        action_mapping = observation.get('action_mapping', {})
        human_aboard = observation.get('human_aboard', {})

        # If on edge, must pass
        my_position = observation.get('my_position')
        if my_position is None and agent_positions is not None:
            my_position = agent_positions.get(self.agent_id)
        if isinstance(my_position, tuple):
            return 0, "Passing (already on edge)"
        if my_position is None:
            return 0, "Passing (position unavailable)"
        current_node = my_position

        # If there are humans waiting at this node, do not depart yet
        # (give the boarding step a chance to pick them up)
        is_empty = all(v != self.agent_id for v in human_aboard.values()) #if vehicle is empty
        if is_empty and agent_positions is not None:
            waiting_humans = [h for h, v in human_aboard.items() if v is None]
            humans_here = []
            for h in waiting_humans:
                pos = agent_positions.get(h)
                # If the human is on an edge, pos may look like ((u, v), progress) or (u, v); in either case, they are not considered "waiting at the station".
                if isinstance(pos, tuple):
                    continue
                # numpy -> python int
                if hasattr(pos, "item"):
                    pos = pos.item()
                if pos == current_node:
                    humans_here.append(h)

            if humans_here:
                self.depart_count = 0
                return 0, f"Waiting for boarding at node {current_node}: {humans_here}"
        # Apply empty-vehicle waiting policy (fixed number of cycles)
        wait_action, self.depart_count, wait_message = apply_empty_vehicle_waiting(
            is_empty=is_empty,
            depart_count=self.depart_count,
            wait_cycles=self.wait_cycles,
            action_space_size=action_space_size
        )
        if wait_action is not None:
            return wait_action, wait_message

        # If no waiting is required, continue with the original logic
        if self.current_destination is None:
            self.depart_count = 0
            return 0, "Passing (no destination set)"
        
        # Get next node on shortest path
        next_node = self._get_next_node_on_path(current_node, self.current_destination)
        if next_node is None:
            self.depart_count = 0
            if current_node == self.current_destination:
                return 0, f"Passing (already at destination {self.current_destination})"
            else:
                return 0, f"Passing (no path to destination {self.current_destination})"


        # apply random outgoing edge noise
        noise_action, noise_message = apply_random_edge(
            current_node=current_node,
            next_node_on_shortest_path=next_node,
            action_mapping=action_mapping,
            action_space_size=action_space_size,
            random_edge_prob=self.random_edge_prob,
            rng=self.rng
        )
        if noise_action is not None:
            return noise_action, noise_message

        # Find action corresponding to edge (current_node, next_node)
        details = action_mapping.get('details', {})
        for action_idx in range(1, action_space_size):
            edge = details.get(action_idx)
            if edge and isinstance(edge, tuple) and edge == (current_node, next_node):
                return action_idx, f"Taking shortest path to destination {self.current_destination} via edge {edge}"
        
        # Edge not found, pass
        return 0, f"Passing (edge to next node {next_node} not available)"
    
    def reset(self):
        """Reset policy state (current destination)."""
        self.current_destination = None
        self.depart_count = 0
