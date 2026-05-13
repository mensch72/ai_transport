"""
POI distribution utilities.

This module connects the Voronoi partition result to the scenario layer and
generates POIs over the study network.

Current default behavior:

- the study area is partitioned by Voronoi polygons derived from road
  intersections (network nodes)
- each node owns one Voronoi cell
- Voronoi cells are not function-anchored to a single POI category
- each POI category (workplace / supermarket / healthcare) generates
  ``num_nodes`` POIs by default
- each POI has a 0.75 probability of being assigned to a central area and
  a 0.25 probability of being assigned to a peripheral area
- residential nodes are sampled as roughly half of all nodes by default
  with 0.5 / 0.5 central / peripheral area probability
- residential places also receive explicit coordinates inside their owner
  Voronoi polygons, while humans still use the owner node as their runnable
  home/start node
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from shapely.geometry import Point

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG

from .voronoi_partition import build_voronoi_partition


POI_CATEGORIES: Tuple[str, ...] = ("workplace", "supermarket", "healthcare")
RESIDENTIAL_CATEGORY = "residential"
DEFAULT_CENTRAL_COUNT = DEFAULT_EXPERIMENT_CONFIG.scenario.central_count

# These frequency / necessity values are chosen so that with the default
# coefficients a=0.5 and b=0.5, the resulting POI weights are:
# workplace=0.40, supermarket=0.35, healthcare=0.25.
DEFAULT_POI_PROFILES: Dict[str, Dict[str, float]] = {
    "workplace": {"frequency": 0.40, "necessity": 0.40},
    "supermarket": {"frequency": 0.35, "necessity": 0.35},
    "healthcare": {"frequency": 0.25, "necessity": 0.25},
}

DEFAULT_POI_AREA_PROBABILITIES: Dict[str, float] = {
    "central": 0.75,
    "peripheral": 0.25,
}

DEFAULT_RESIDENTIAL_AREA_PROBABILITIES: Dict[str, float] = {
    "central": 0.5,
    "peripheral": 0.5,
}


def compute_poi_weights(
    a: float = 0.5,
    b: float = 0.5,
    poi_profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, float]:
    """
    Compute POI weights from frequency and necessity inputs.

    Args:
        a: Coefficient for the frequency term.
        b: Coefficient for the necessity term.
        poi_profiles: Optional category-specific profile dictionary.

    Returns:
        Mapping from POI category to computed POI weight.
    """
    if poi_profiles is None:
        poi_profiles = DEFAULT_POI_PROFILES

    poi_weights = {}
    for poi_type in POI_CATEGORIES:
        profile = poi_profiles[poi_type]
        poi_weights[poi_type] = (
            a * float(profile["frequency"]) + b * float(profile["necessity"])
        )
    return poi_weights


def _sample_point_in_geometry(
    geometry,
    fallback_xy: Tuple[float, float],
    rng: np.random.RandomState,
    avoid_xy: Optional[Tuple[float, float]] = None,
    min_avoid_distance: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Sample a point inside a polygon-like geometry, optionally avoiding a node.
    """
    if geometry is None or geometry.is_empty:
        return fallback_xy

    min_x, min_y, max_x, max_y = geometry.bounds
    if min_avoid_distance is None:
        span = max(max_x - min_x, max_y - min_y, 1.0)
        min_avoid_distance = 0.08 * span

    best_candidate = None
    best_distance = -1.0
    for _ in range(200):
        x = float(rng.uniform(min_x, max_x))
        y = float(rng.uniform(min_y, max_y))
        if not geometry.contains(Point(x, y)):
            continue
        if avoid_xy is None:
            return x, y

        distance = float(np.hypot(x - avoid_xy[0], y - avoid_xy[1]))
        if distance >= min_avoid_distance:
            return x, y
        if distance > best_distance:
            best_candidate = (x, y)
            best_distance = distance

    if best_candidate is not None:
        return best_candidate

    representative = geometry.representative_point()
    if avoid_xy is not None:
        distance = float(np.hypot(representative.x - avoid_xy[0], representative.y - avoid_xy[1]))
        if distance < 1e-9:
            centroid = geometry.centroid
            return float(centroid.x), float(centroid.y)
    return float(representative.x), float(representative.y)


def _normalize_probabilities(
    probabilities: Dict[str, float],
    available_area_types: List[str],
) -> List[float]:
    """
    Normalize area probabilities over the area types that actually have nodes.
    """
    weights = [max(0.0, float(probabilities.get(area_type, 0.0))) for area_type in available_area_types]
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(available_area_types)] * len(available_area_types)
    return [weight / total for weight in weights]


def _group_nodes_by_area_type(
    nodes: List[Any],
    node_to_central_level: Dict[Any, str],
) -> Dict[str, List[Any]]:
    """
    Group Voronoi owner nodes by central/peripheral area label.
    """
    nodes_by_area_type: Dict[str, List[Any]] = {}
    for node in nodes:
        area_type = node_to_central_level.get(node, "peripheral")
        nodes_by_area_type.setdefault(area_type, []).append(node)
    return nodes_by_area_type


def _sample_node_by_area_probability(
    nodes_by_area_type: Dict[str, List[Any]],
    area_probabilities: Dict[str, float],
    rng: np.random.RandomState,
) -> Any:
    """
    Sample one node by first sampling its central/peripheral area type.
    """
    available_area_types = [
        area_type for area_type, area_nodes in nodes_by_area_type.items()
        if area_nodes
    ]
    if not available_area_types:
        raise ValueError("Cannot sample a node because no area type has nodes.")

    probabilities = _normalize_probabilities(area_probabilities, available_area_types)
    area_type = available_area_types[
        int(rng.choice(len(available_area_types), p=probabilities))
    ]
    area_nodes = nodes_by_area_type[area_type]
    return area_nodes[int(rng.choice(len(area_nodes)))]


def _sample_unique_nodes_by_area_probability(
    nodes_by_area_type: Dict[str, List[Any]],
    area_probabilities: Dict[str, float],
    sample_count: int,
    rng: np.random.RandomState,
) -> List[Any]:
    """
    Sample unique nodes while biasing each draw by central/peripheral probability.
    """
    available = {
        area_type: list(area_nodes)
        for area_type, area_nodes in nodes_by_area_type.items()
        if area_nodes
    }
    selected: List[Any] = []

    while len(selected) < sample_count and any(available.values()):
        available_area_types = [
            area_type for area_type, area_nodes in available.items()
            if area_nodes
        ]
        probabilities = _normalize_probabilities(area_probabilities, available_area_types)
        area_type = available_area_types[
            int(rng.choice(len(available_area_types), p=probabilities))
        ]
        area_nodes = available[area_type]
        selected_index = int(rng.choice(len(area_nodes)))
        selected.append(area_nodes.pop(selected_index))

    return selected


def assign_poi_to_voronoi_cells(
    graph: nx.DiGraph,
    partition_info: Dict[str, Any],
    hierarchy_info: Optional[Dict[str, Any]] = None,
    residential_ratio: float = 0.5,
    a: float = 0.5,
    b: float = 0.5,
    poi_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    poi_count_per_type: Optional[int] = None,
    poi_area_probabilities: Optional[Dict[str, float]] = None,
    residential_area_probabilities: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Assign residential nodes and generate explicit POI records on Voronoi cells.

    The default behavior is:
    - sample residential nodes with 0.5 / 0.5 central / peripheral probability
    - generate a fixed number of POIs per non-residential category
      (default: one full-network pass for each category)
    - sample each POI owner area with 0.75 / 0.25 central / peripheral
      probability
    - sample each POI's position inside its owner Voronoi geometry

    Node attributes are kept for downstream compatibility. Since a node can now
    host multiple POI categories through explicit records, ``poi_type`` on node
    stores the dominant category by record count.

    """
    if poi_profiles is None:
        poi_profiles = DEFAULT_POI_PROFILES
    if poi_area_probabilities is None:
        poi_area_probabilities = DEFAULT_POI_AREA_PROBABILITIES
    if residential_area_probabilities is None:
        residential_area_probabilities = DEFAULT_RESIDENTIAL_AREA_PROBABILITIES

    rng = np.random.RandomState(seed)

    nodes = list(partition_info["nodes"])
    node_to_cell = partition_info["node_to_cell"]
    cells = partition_info["cells"]
    if hierarchy_info is None:
        hierarchy_info = {}
    node_to_central_level = (
        partition_info.get("node_to_area_type")
        or hierarchy_info.get("node_to_level", {})
    )
    nodes_by_area_type = _group_nodes_by_area_type(nodes, node_to_central_level)

    if not nodes:
        raise ValueError("Cannot generate POI distribution on an empty node set.")

    num_nodes = len(nodes)
    residential_count = int(num_nodes * residential_ratio)
    residential_count = max(1, min(residential_count, num_nodes))
    if poi_count_per_type is None:
        poi_count_per_type = num_nodes
    poi_count_per_type = max(0, int(poi_count_per_type))

    poi_weights = compute_poi_weights(a=a, b=b, poi_profiles=poi_profiles)
    poi_area_counts = {poi_type: poi_count_per_type for poi_type in POI_CATEGORIES}

    residential_nodes = _sample_unique_nodes_by_area_probability(
        nodes_by_area_type=nodes_by_area_type,
        area_probabilities=residential_area_probabilities,
        sample_count=residential_count,
        rng=rng,
    )
    residential_node_set = set(residential_nodes)
    residential_records: List[Dict[str, Any]] = []
    residential_record_by_node: Dict[Any, Dict[str, Any]] = {}
    for index, node in enumerate(residential_nodes):
        cell_id = node_to_cell.get(node)
        geometry = cells.get(cell_id) if cell_id is not None else None
        central_level = node_to_central_level.get(node, "peripheral")
        fallback_xy = (
            float(graph.nodes[node].get("x", 0.0)),
            float(graph.nodes[node].get("y", 0.0)),
        )
        residential_x, residential_y = _sample_point_in_geometry(
            geometry=geometry,
            fallback_xy=fallback_xy,
            rng=rng,
            avoid_xy=fallback_xy,
        )
        residential_record = {
            "residential_id": f"residential_{index}",
            "area_node": node,
            "nearest_node": node,
            "node": node,
            "cell_id": cell_id,
            "x": residential_x,
            "y": residential_y,
            "geometry": geometry,
            "central_level": central_level,
        }
        residential_records.append(residential_record)
        residential_record_by_node[node] = residential_record

    node_records: Dict[Any, Dict[str, Any]] = {}
    poi_records: List[Dict[str, Any]] = []
    node_poi_type_counts: Dict[Any, Dict[str, int]] = {
        node: {poi_type: 0 for poi_type in POI_CATEGORIES}
        for node in nodes
    }

    for poi_type in POI_CATEGORIES:
        frequency = float(poi_profiles[poi_type]["frequency"])
        necessity = float(poi_profiles[poi_type]["necessity"])
        poi_weight = float(poi_weights[poi_type])
        for _ in range(poi_count_per_type):
            node = _sample_node_by_area_probability(
                nodes_by_area_type=nodes_by_area_type,
                area_probabilities=poi_area_probabilities,
                rng=rng,
            )
            node_poi_type_counts[node][poi_type] += 1

            cell_id = node_to_cell.get(node)
            geometry = cells.get(cell_id) if cell_id is not None else None
            central_level = node_to_central_level.get(node, "peripheral")
            fallback_xy = (
                float(graph.nodes[node].get("x", 0.0)),
                float(graph.nodes[node].get("y", 0.0)),
            )
            poi_x, poi_y = _sample_point_in_geometry(
                geometry=geometry,
                fallback_xy=fallback_xy,
                rng=rng,
                avoid_xy=fallback_xy,
            )
            poi_records.append({
                "poi_id": f"{poi_type}_{len(poi_records)}",
                "poi_type": poi_type,
                "area_node": node,
                "nearest_node": node,
                "node": node,
                "cell_id": cell_id,
                "x": poi_x,
                "y": poi_y,
                "geometry": geometry,
                "frequency": frequency,
                "necessity": necessity,
                "poi_weight": poi_weight,
                "central_level": central_level,
            })

    poi_nodes_by_type: Dict[str, List[Any]] = {
        poi_type: [
            node for node in nodes if node_poi_type_counts[node][poi_type] > 0
        ]
        for poi_type in POI_CATEGORIES
    }

    area_type_to_nodes: Dict[str, List[Any]] = {
        RESIDENTIAL_CATEGORY: list(residential_nodes),
        **{
            poi_type: list(poi_nodes_by_type[poi_type])
            for poi_type in POI_CATEGORIES
        },
    }

    for node in nodes:
        cell_id = node_to_cell.get(node)
        geometry = cells.get(cell_id) if cell_id is not None else None
        central_level = node_to_central_level.get(node, "peripheral")

        counts_by_type = node_poi_type_counts[node]
        available_types = [
            poi_type for poi_type in POI_CATEGORIES if counts_by_type[poi_type] > 0
        ]
        poi_type = None
        if available_types:
            poi_type = max(
                POI_CATEGORIES,
                key=lambda poi: (counts_by_type[poi], poi_weights[poi]),
            )
        area_type = (
            RESIDENTIAL_CATEGORY
            if node in residential_node_set
            else "mixed_poi"
        )
        if poi_type is None:
            poi_weight = 0.0
            frequency = 0.0
            necessity = 0.0
        else:
            poi_weight = float(poi_weights[poi_type])
            frequency = float(poi_profiles[poi_type]["frequency"])
            necessity = float(poi_profiles[poi_type]["necessity"])

        record = {
            "node": node,
            "cell_id": cell_id,
            "geometry": geometry,
            "area_type": area_type,
            "poi_type": poi_type,
            "poi_weight": poi_weight,
            "frequency": frequency,
            "necessity": necessity,
            "central_level": central_level,
            "residential_place": residential_record_by_node.get(node),
            "poi_types": list(available_types),
            "poi_type_counts": dict(counts_by_type),
        }
        node_records[node] = record

        graph.nodes[node]["area_type"] = area_type
        graph.nodes[node]["poi_type"] = poi_type
        graph.nodes[node]["poi_types"] = list(available_types)
        graph.nodes[node]["poi_type_counts"] = dict(counts_by_type)
        graph.nodes[node]["poi_weight"] = poi_weight
        graph.nodes[node]["frequency"] = frequency
        graph.nodes[node]["necessity"] = necessity
        graph.nodes[node]["central_level"] = central_level
        graph.nodes[node]["residential_place"] = residential_record_by_node.get(node)
        graph.nodes[node]["voronoi_cell_id"] = cell_id
        graph.nodes[node]["voronoi_geometry"] = geometry

    graph.graph["poi_records"] = list(poi_records)
    graph.graph["residential_records"] = list(residential_records)
    graph.graph["poi_nodes_by_type"] = {
        poi_type: list(nodes)
        for poi_type, nodes in poi_nodes_by_type.items()
    }
    graph.graph["residential_nodes"] = list(residential_nodes)

    return {
        "partition_info": partition_info,
        "hierarchy_info": hierarchy_info,
        "voronoi_area_info": {
            "node_to_area_type": dict(partition_info.get("node_to_area_type", {})),
            "central_nodes": list(partition_info.get("central_nodes", [])),
            "peripheral_nodes": list(partition_info.get("peripheral_nodes", [])),
            "node_degrees": dict(partition_info.get("node_degrees", {})),
        },
        "poi_profiles": poi_profiles,
        "poi_weights": poi_weights,
        "poi_area_counts": poi_area_counts,
        "poi_area_probabilities": dict(poi_area_probabilities),
        "residential_area_probabilities": dict(residential_area_probabilities),
        "residential_nodes": residential_nodes,
        "residential_records": residential_records,
        "residential_record_by_node": dict(residential_record_by_node),
        "residential_count": residential_count,
        "poi_count_per_type": poi_count_per_type,
        "poi_nodes_by_type": poi_nodes_by_type,
        "poi_records": poi_records,
        "area_type_to_nodes": area_type_to_nodes,
        "node_records": node_records,
    }


def build_poi_distribution(
    graph: nx.DiGraph,
    partition_info: Optional[Dict[str, Any]] = None,
    hierarchy_info: Optional[Dict[str, Any]] = None,
    hierarchy_k: int = 6,
    central_count: int = DEFAULT_CENTRAL_COUNT,
    residential_ratio: float = 0.5,
    a: float = 0.5,
    b: float = 0.5,
    poi_count_per_type: Optional[int] = None,
    poi_area_probabilities: Optional[Dict[str, float]] = None,
    residential_area_probabilities: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Build the POI distribution for the current study network.

    This is the main scenario entry point for POI assignment. If the Voronoi
    partition or the hierarchy information is not provided, they are generated
    automatically from the input network.
    """
    if partition_info is None:
        partition_info = build_voronoi_partition(
            graph,
            central_count=central_count,
        )
    _ = hierarchy_k

    return assign_poi_to_voronoi_cells(
        graph=graph,
        partition_info=partition_info,
        hierarchy_info=hierarchy_info,
        residential_ratio=residential_ratio,
        a=a,
        b=b,
        poi_count_per_type=poi_count_per_type,
        poi_area_probabilities=poi_area_probabilities,
        residential_area_probabilities=residential_area_probabilities,
        seed=seed,
    )
