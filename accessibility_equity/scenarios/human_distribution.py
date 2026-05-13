"""
Human distribution utilities.

- identify residential nodes from the POI / area assignment result
- place humans at residential areas at the initial time step
- treat the sampled start node as the human's home node
- randomly choose non-residential POIs as human target / demand nodes
- write the sampled homes and targets into a simple scenario dictionary
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np


def get_residential_nodes(
    graph: nx.DiGraph,
    area_type_attr: str = "area_type",
    residential_type: str = "residential",
) -> List[Any]:
    """
    Return the nodes that are marked as residential areas.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get(area_type_attr) == residential_type
    ]


def get_poi_nodes(
    graph: nx.DiGraph,
    poi_type_attr: str = "poi_type",
) -> List[Any]:
    """
    Return nodes that contain a non-residential POI.
    """
    return [
        node
        for node, data in graph.nodes(data=True)
        if data.get(poi_type_attr) is not None
    ]


def get_poi_records(graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Return POI records stored on the graph, if available.
    """
    return list(graph.graph.get("poi_records", []))


def distribute_humans_to_residential_areas(
    graph: nx.DiGraph,
    num_humans: int,
    residential_nodes: Optional[List[Any]] = None,
    target_nodes: Optional[List[Any]] = None,
    target_poi_records: Optional[List[Dict[str, Any]]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sample initial human positions from residential nodes.

    Args:
        graph: Scenario network with node attributes already assigned.
        num_humans: Number of humans to place initially.
        residential_nodes: Optional precomputed residential node list.
        seed: Random seed.

    Returns:
        A dictionary containing:
        - `residential_nodes`
        - `initial_human_positions`
        - `human_to_home_node`
        - `human_to_initial_node`
        - `human_to_target_node`
        - `human_records`
    """
    if residential_nodes is None:
        residential_nodes = get_residential_nodes(graph)

    if not residential_nodes:
        raise ValueError(
            "No residential nodes are available. Build the POI distribution "
            "before placing humans."
        )

    rng = np.random.RandomState(seed)
    sampled_nodes = rng.choice(residential_nodes, size=num_humans, replace=True)
    initial_human_positions = [
        node.item() if hasattr(node, "item") else node for node in sampled_nodes
    ]
    human_to_home_node = {
        f"human_{i}": initial_human_positions[i] for i in range(num_humans)
    }
    human_to_initial_node = {
        f"human_{i}": initial_human_positions[i] for i in range(num_humans)
    }

    if target_poi_records is None:
        target_poi_records = get_poi_records(graph)

    human_target_poi_records: Dict[str, Optional[Dict[str, Any]]] = {}
    if target_poi_records:
        sampled_indices = rng.choice(
            len(target_poi_records),
            size=num_humans,
            replace=True,
        )
        sampled_records = [target_poi_records[int(index)] for index in sampled_indices]
        human_goal_nodes = [
            record.get("nearest_node", record.get("node"))
            for record in sampled_records
        ]
        human_target_poi_records = {
            f"human_{i}": sampled_records[i] for i in range(num_humans)
        }
        target_nodes = [
            record.get("nearest_node", record.get("node"))
            for record in target_poi_records
        ]
    else:
        if target_nodes is None:
            target_nodes = get_poi_nodes(graph)
        if not target_nodes:
            target_nodes = list(graph.nodes())
        if not target_nodes:
            raise ValueError("No nodes are available for human target generation.")

        sampled_targets = rng.choice(target_nodes, size=num_humans, replace=True)
        human_goal_nodes = [
            node.item() if hasattr(node, "item") else node for node in sampled_targets
        ]
        human_target_poi_records = {
            f"human_{i}": None for i in range(num_humans)
        }

    human_to_target_node = {
        f"human_{i}": human_goal_nodes[i] for i in range(num_humans)
    }

    human_records = {
        human_id: {
            "home_node": home_node,
            "initial_node": human_to_initial_node[human_id],
            "target_node": human_to_target_node[human_id],
            "target_poi": human_target_poi_records.get(human_id),
        }
        for human_id, home_node in human_to_home_node.items()
    }

    return {
        "residential_nodes": list(residential_nodes),
        "initial_human_positions": initial_human_positions,
        "human_to_home_node": human_to_home_node,
        "human_to_initial_node": human_to_initial_node,
        "target_nodes": list(target_nodes),
        "human_goal_nodes": human_goal_nodes,
        "human_to_target_node": human_to_target_node,
        "human_to_target_poi": human_target_poi_records,
        "human_records": human_records,
    }


def build_human_distribution(
    graph: nx.DiGraph,
    num_humans: int,
    seed: int = 42,
    target_nodes: Optional[List[Any]] = None,
    target_poi_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for initial human placement.
    """
    residential_nodes = get_residential_nodes(graph)
    return distribute_humans_to_residential_areas(
        graph=graph,
        num_humans=num_humans,
        residential_nodes=residential_nodes,
        target_nodes=target_nodes,
        target_poi_records=target_poi_records,
        seed=seed,
    )
