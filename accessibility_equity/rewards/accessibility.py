"""
Accessibility metric calculations.

This module implements a first location-based accessibility metric for the
accessibility-equity project.

The intended interpretation is:

- accessibility belongs to the human's effective location within the current
  transport system
- vehicle routes are not the accessibility value itself
- instead, routes influence the effective travel cost of reaching POIs
- the accessibility value aggregates reachable POIs across categories with:
  - category weights
  - distance decay
  - within-category diminishing returns

The current implementation is still an approximation. It remains lightweight
and network-based:

- the human is represented by a current network node
- the current transport system is approximated through an ordered list of route
  nodes that can influence access costs
- POIs are represented by explicit POI records stored on the scenario graph
- effective access cost is computed as a shortest-path travel-time
  approximation on the graph using edge length and speed
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG
from accessibility_equity.scenarios.poi_distribution import POI_CATEGORIES


DEFAULT_REWARD = DEFAULT_EXPERIMENT_CONFIG.reward


def get_poi_nodes_by_type(
    graph: nx.DiGraph,
    poi_type_attr: str = "poi_type",
) -> Dict[str, List[Any]]:
    """
    Collect POI nodes by category from the scenario graph.
    """
    poi_nodes_by_type: Dict[str, List[Any]] = {poi_type: [] for poi_type in POI_CATEGORIES}

    for node, data in graph.nodes(data=True):
        poi_type = data.get(poi_type_attr)
        if poi_type in poi_nodes_by_type:
            poi_nodes_by_type[poi_type].append(node)

    return poi_nodes_by_type


def get_poi_records_by_type(
    graph: nx.DiGraph,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect explicit POI records by category from the scenario graph.

    ``graph.graph["poi_records"]`` is the full POI representation. Node-level
    POI fields such as ``node["poi_type"]`` are only summaries kept for
    compatibility and visualization.
    """
    poi_records_by_type: Dict[str, List[Dict[str, Any]]] = {
        poi_type: [] for poi_type in POI_CATEGORIES
    }

    for record in graph.graph.get("poi_records", []):
        poi_type = record.get("poi_type")
        if poi_type in poi_records_by_type:
            poi_records_by_type[poi_type].append(record)

    return poi_records_by_type


def _edge_time_cost(
    u: Any,
    v: Any,
    data: Dict[str, Any],
) -> float:
    """
    Minimal edge travel-time approximation based on length and speed.
    """
    _ = u, v
    length = float(data.get("length", 1.0))
    speed = float(data.get("speed", 1.0))
    if speed <= 0.0:
        return float("inf")
    return length / speed


def _shortest_path_time_cost(
    graph: nx.DiGraph,
    source: Any,
    target: Any,
) -> float:
    """
    Compute shortest-path travel time, returning infinity when unreachable.
    """
    try:
        return float(
            nx.shortest_path_length(
                graph,
                source=source,
                target=target,
                weight=_edge_time_cost,
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def compute_effective_access_cost(
    graph: nx.DiGraph,
    human_node: Any,
    route_nodes: Sequence[Any],
    poi_node: Any,
    weight: str = "length",
) -> float:
    """
    Compute an effective access cost from a human location to a POI.

    In the intended model, accessibility is location-based and the route is a
    mechanism that changes travel opportunities. The current implementation uses
    the route nodes as a first approximation of the transport system available
    to the human.

    The current effective-cost definition is:

    - the human first reaches some node on the given vehicle route
    - from the route, the human can benefit from the best route node relative
      to the target POI
    - the effective access cost is the minimum over route nodes of:

          time(human_node, route_node) + time(route_node, poi_node)

    This is a first operational approximation of location-based accessibility
    under a route-influenced transport system.
    """
    if not route_nodes:
        return float("inf")

    min_cost = float("inf")
    for route_node in route_nodes:
        cost_to_route = _shortest_path_time_cost(
            graph,
            source=human_node,
            target=route_node,
        )
        cost_from_route = _shortest_path_time_cost(
            graph,
            source=route_node,
            target=poi_node,
        )
        total_cost = cost_to_route + cost_from_route
        if total_cost < min_cost:
            min_cost = total_cost

    return min_cost


def collect_accessible_pois(
    graph: nx.DiGraph,
    human_node: Any,
    route_nodes: Sequence[Any],
    poi_records_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    weight: str = "length",
) -> Dict[str, List[Tuple[Dict[str, Any], float]]]:
    """
    Collect accessible POIs for a human under the current transport-state approximation.

    Returns:
        Mapping from POI type to a list of `(poi_record, effective_access_cost)`.
    """
    if poi_records_by_type is None:
        poi_records_by_type = get_poi_records_by_type(graph)

    opportunities: Dict[str, List[Tuple[Dict[str, Any], float]]] = {
        poi_type: [] for poi_type in POI_CATEGORIES
    }

    for poi_type, poi_records in poi_records_by_type.items():
        for poi_record in poi_records:
            poi_node = poi_record.get("nearest_node", poi_record.get("node"))
            if poi_node is None:
                continue
            access_cost = compute_effective_access_cost(
                graph=graph,
                human_node=human_node,
                route_nodes=route_nodes,
                poi_node=poi_node,
                weight=weight,
            )
            if math.isfinite(access_cost):
                opportunities[poi_type].append((poi_record, access_cost))

    return opportunities


def compute_category_accessibility(
    opportunities: Iterable[Tuple[Any, float]],
    category_weight: float,
    beta: float = DEFAULT_REWARD.beta,
    gamma: float = DEFAULT_REWARD.alpha,
) -> float:
    """
    Compute the accessibility contribution of one POI category.

    This follows the formulation:

        a_c * (sum_j exp(-beta * d_j)) ** gamma
    """
    decay_sum = 0.0
    for _poi_node, access_cost in opportunities:
        decay_sum += math.exp(-beta * float(access_cost))

    if decay_sum <= 0.0:
        return 0.0

    return float(category_weight) * (decay_sum ** float(gamma))


def compute_location_based_accessibility(
    graph: nx.DiGraph,
    human_node: Any,
    route_nodes: Sequence[Any],
    poi_weights: Optional[Dict[str, float]] = None,
    poi_records_by_type: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    beta: float = DEFAULT_REWARD.beta,
    gamma: float = DEFAULT_REWARD.alpha,
    weight: str = "length",
) -> float:
    """
    Compute the location-based accessibility value for a single human.

    The current implementation is route-influenced: route nodes affect the
    effective access cost, which in turn affects accessibility.
    """
    if poi_weights is None:
        poi_weights = {
            "workplace": 0.40,
            "supermarket": 0.35,
            "healthcare": 0.25,
        }

    opportunities = collect_accessible_pois(
        graph=graph,
        human_node=human_node,
        route_nodes=route_nodes,
        poi_records_by_type=poi_records_by_type,
        weight=weight,
    )

    accessibility_value = 0.0
    for poi_type in POI_CATEGORIES:
        accessibility_value += compute_category_accessibility(
            opportunities=opportunities.get(poi_type, []),
            category_weight=float(poi_weights.get(poi_type, 0.0)),
            beta=beta,
            gamma=gamma,
        )

    return accessibility_value


def compute_population_accessibility(
    graph: nx.DiGraph,
    human_to_node: Dict[Any, Any],
    human_to_route: Dict[Any, Sequence[Any]],
    poi_weights: Optional[Dict[str, float]] = None,
    beta: float = DEFAULT_REWARD.beta,
    gamma: float = DEFAULT_REWARD.alpha,
    weight: str = "length",
) -> Dict[str, Any]:
    """
    Compute location-based accessibility for a population of humans under the
    current transport-state approximation.

    Returns:
        A dictionary with per-human accessibility values and the raw total.
    """
    poi_records_by_type = get_poi_records_by_type(graph)

    individual_values: Dict[Any, float] = {}
    for human_id, human_node in human_to_node.items():
        route_nodes = human_to_route.get(human_id, [])
        individual_values[human_id] = compute_location_based_accessibility(
            graph=graph,
            human_node=human_node,
            route_nodes=route_nodes,
            poi_weights=poi_weights,
            poi_records_by_type=poi_records_by_type,
            beta=beta,
            gamma=gamma,
            weight=weight,
        )

    total_accessibility = float(sum(individual_values.values()))

    return {
        "individual_values": individual_values,
        "total_accessibility": total_accessibility,
    }


# Backward-compatible aliases retained while the rest of the project migrates
# from the earlier route-based naming to the newer location-based wording.
compute_route_access_cost = compute_effective_access_cost
collect_route_based_opportunities = collect_accessible_pois
compute_individual_accessibility = compute_location_based_accessibility
