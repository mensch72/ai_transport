"""
Vehicle distribution utilities.

This module decides where vehicles start in a prepared transport network. It is
kept separate from the high-level scenario builder so real fleet-depot data or
service-area rules can replace the simple random sampler later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from .human_distribution import get_residential_nodes


def distribute_vehicles_to_nodes(
    graph: nx.DiGraph,
    num_vehicles: int,
    candidate_nodes: Optional[List[Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sample initial vehicle positions from network nodes.
    """
    if candidate_nodes is None:
        candidate_nodes = get_residential_nodes(graph)
        if not candidate_nodes:
            candidate_nodes = list(graph.nodes())

    if not candidate_nodes:
        raise ValueError("No nodes are available for vehicle placement.")

    rng = np.random.RandomState(seed)
    sampled_nodes = rng.choice(candidate_nodes, size=num_vehicles, replace=True)
    initial_vehicle_positions = [
        node.item() if hasattr(node, "item") else node for node in sampled_nodes
    ]
    vehicle_to_initial_node = {
        f"vehicle_{i}": initial_vehicle_positions[i]
        for i in range(num_vehicles)
    }
    vehicle_records = {
        vehicle_id: {"initial_node": initial_node}
        for vehicle_id, initial_node in vehicle_to_initial_node.items()
    }

    return {
        "candidate_nodes": list(candidate_nodes),
        "initial_vehicle_positions": initial_vehicle_positions,
        "vehicle_to_initial_node": vehicle_to_initial_node,
        "vehicle_records": vehicle_records,
    }


def build_vehicle_distribution(
    graph: nx.DiGraph,
    num_vehicles: int,
    seed: int = 42,
    candidate_nodes: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for initial vehicle placement.
    """
    return distribute_vehicles_to_nodes(
        graph=graph,
        num_vehicles=num_vehicles,
        candidate_nodes=candidate_nodes,
        seed=seed,
    )
