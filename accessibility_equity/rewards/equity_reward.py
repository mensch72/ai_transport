"""
Equity-aware reward calculations.

This module turns route-based individual accessibility values into a
population-level, inequality-sensitive objective following the project
formulation.

Individual accessibility:

    X_h(s_t) = sum_c a_c (sum_{j: c_j=c} exp(-beta d_j)) ** alpha

Population-level utility:

    U(s_t) = -(sum_h X_h(s_t) ** (-xi)) ** eta

To keep the implementation numerically stable and directly usable for
reinforcement learning, this module provides both:

- a state utility function `U(s_t)`
- a step reward based on utility improvement:
      r_t = U(s_{t+1}) - U(s_t)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx

from accessibility_equity.scenarios.poi_distribution import POI_CATEGORIES


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

    The current effective-cost definition is the minimum over route nodes of:

        time(human_node, route_node) + time(route_node, poi_node)
    """
    _ = weight
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
    beta: float = 1.0,
    gamma: float = 0.8,
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
    beta: float = 1.0,
    gamma: float = 0.8,
    weight: str = "length",
) -> float:
    """
    Compute the location-based accessibility value for a single human.
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
    beta: float = 1.0,
    gamma: float = 0.8,
    weight: str = "length",
) -> Dict[str, Any]:
    """
    Compute location-based accessibility for a population of humans.
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


def _config_value(config: Optional[Any], name: str, explicit_value: Optional[float], default: Any) -> Any:
    if explicit_value is not None:
        return float(explicit_value)
    if config is not None and hasattr(config, name):
        return getattr(config, name)
    return default


def compute_population_utility(
    individual_values: Dict[Any, float],
    xi: float = 1.0,
    eta: float = 2.0,
    epsilon: float = 1e-6,
) -> float:
    """
    Aggregate individual accessibility values into the formulation's
    inequality-sensitive utility.

    Args:
        individual_values: Mapping from human id to accessibility value X_h.
        xi: Inequality-aversion exponent inside X_h^(-xi).
        eta: Outer exponent in:
            U(s_t) = -(sum_h X_h(s_t)^(-xi))^eta
        epsilon: Small stabilizer to avoid division by zero.

    Returns:
        Scalar population utility.
    """
    if not individual_values:
        return 0.0

    inequality_sum = 0.0
    for value in individual_values.values():
        stabilized_value = max(float(value), epsilon)
        inequality_sum += stabilized_value ** (-float(xi))

    if inequality_sum <= 0.0:
        return 0.0

    utility = -(inequality_sum ** float(eta))

    return float(utility)


def compute_sum_accessibility_utility(individual_values: Dict[Any, float]) -> float:
    """
    Aggregate individual accessibility values by simple population sum.

    This objective maximizes total accessibility and does not directly penalize
    inequality between people.
    """
    return float(sum(float(value) for value in individual_values.values()))


def compute_utility_by_mode(
    individual_values: Dict[Any, float],
    utility_mode: str = "equity",
    xi: float = 1.0,
    eta: float = 2.0,
    epsilon: float = 1e-6,
) -> float:
    """
    Compute population utility according to the configured objective.
    """
    if utility_mode == "equity":
        return compute_population_utility(
            individual_values=individual_values,
            xi=xi,
            eta=eta,
            epsilon=epsilon,
        )
    if utility_mode == "sum_accessibility":
        return compute_sum_accessibility_utility(individual_values)
    raise ValueError(
        "utility_mode must be one of ['equity', 'sum_accessibility'], "
        f"got {utility_mode!r}."
    )


def compute_state_utility(
    graph: nx.DiGraph,
    human_to_node: Dict[Any, Any],
    human_to_route: Dict[Any, Sequence[Any]],
    poi_weights: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    xi: Optional[float] = None,
    eta: Optional[float] = None,
    weight: str = "length",
    epsilon: Optional[float] = None,
    reward_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compute the population utility U(s_t) from the current state description.

    Note:
        The implementation uses the project's accessibility module, where the
        within-category diminishing-returns parameter is named `gamma`.
        Here it is passed from the formulation's `alpha`.
    """
    beta_value = _config_value(reward_config, "beta", beta, 1.0)
    alpha_value = _config_value(reward_config, "alpha", alpha, 0.8)
    xi_value = _config_value(reward_config, "xi", xi, 1.0)
    eta_value = _config_value(reward_config, "eta", eta, 2.0)
    epsilon_value = _config_value(reward_config, "epsilon", epsilon, 1e-6)
    utility_mode = getattr(reward_config, "utility_mode", "equity")

    accessibility_result = compute_population_accessibility(
        graph=graph,
        human_to_node=human_to_node,
        human_to_route=human_to_route,
        poi_weights=poi_weights,
        beta=beta_value,
        gamma=alpha_value,
        weight=weight,
    )

    individual_values = accessibility_result["individual_values"]
    utility = compute_utility_by_mode(
        individual_values=individual_values,
        utility_mode=utility_mode,
        xi=xi_value,
        eta=eta_value,
        epsilon=epsilon_value,
    )

    return {
        "individual_values": individual_values,
        "total_accessibility": accessibility_result["total_accessibility"],
        "utility": utility,
    }


def compute_step_reward(
    current_utility: float,
    next_utility: float,
    operating_cost: float = 0.0,
    operating_cost_weight: float = 0.0,
) -> float:
    """
    Convert utility improvement into a reinforcement-learning reward.

    This is the recommended training-time reward:

        r_t = U(s_{t+1}) - U(s_t) - lambda * operating_cost_t
    """
    return (
        float(next_utility)
        - float(current_utility)
        - float(operating_cost_weight) * float(operating_cost)
    )


def compute_reward(
    graph: nx.DiGraph,
    current_human_to_node: Dict[Any, Any],
    current_human_to_route: Dict[Any, Sequence[Any]],
    next_human_to_node: Optional[Dict[Any, Any]] = None,
    next_human_to_route: Optional[Dict[Any, Sequence[Any]]] = None,
    poi_weights: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    xi: Optional[float] = None,
    eta: Optional[float] = None,
    weight: str = "length",
    epsilon: Optional[float] = None,
    operating_cost: float = 0.0,
    operating_cost_weight: float = 0.0,
    reward_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Main reward entry point.

    Two modes are supported:

    1. If only the current state is provided, the function returns the
       state utility U(s_t). This is useful for analysis and debugging.
    2. If both current and next state are provided, the function returns the
       step reward based on utility improvement.
    """
    current_result = compute_state_utility(
        graph=graph,
        human_to_node=current_human_to_node,
        human_to_route=current_human_to_route,
        poi_weights=poi_weights,
        beta=beta,
        alpha=alpha,
        xi=xi,
        eta=eta,
        weight=weight,
        epsilon=epsilon,
        reward_config=reward_config,
    )

    reward = current_result["utility"]
    next_result = None

    if next_human_to_node is not None and next_human_to_route is not None:
        next_result = compute_state_utility(
            graph=graph,
            human_to_node=next_human_to_node,
            human_to_route=next_human_to_route,
            poi_weights=poi_weights,
            beta=beta,
            alpha=alpha,
            xi=xi,
            eta=eta,
            weight=weight,
            epsilon=epsilon,
            reward_config=reward_config,
        )
        reward = compute_step_reward(
            current_utility=current_result["utility"],
            next_utility=next_result["utility"],
            operating_cost=operating_cost,
            operating_cost_weight=operating_cost_weight,
        )

    return {
        "reward": reward,
        "current_utility": current_result["utility"],
        "current_individual_values": current_result["individual_values"],
        "current_total_accessibility": current_result["total_accessibility"],
        "next_utility": None if next_result is None else next_result["utility"],
        "next_individual_values": (
            None if next_result is None else next_result["individual_values"]
        ),
        "next_total_accessibility": (
            None if next_result is None else next_result["total_accessibility"]
        ),
    }


def compute_scenario_utility_bounds(
    graph: nx.DiGraph,
    population_size: int,
    poi_weights: Optional[Dict[str, float]] = None,
    beta: Optional[float] = None,
    alpha: Optional[float] = None,
    xi: Optional[float] = None,
    eta: Optional[float] = None,
    epsilon: Optional[float] = None,
    reward_config: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Estimate utility lower/upper bounds for one generated scenario.

    Assumption:
        A person's location-based accessibility is evaluated at each network
        node with a singleton route ``[node]``. The worst node gives ``X_min``
        and the best node gives ``X_max``. Population utility bounds are then:

        - lower utility: all humans have ``X_min``
        - upper utility: all humans have ``X_max``

    This is a scenario diagnostic for the current accessibility approximation;
    it does not multiply by time and does not add residential group terms.
    """
    beta_value = _config_value(reward_config, "beta", beta, 1.0)
    alpha_value = _config_value(reward_config, "alpha", alpha, 0.8)
    xi_value = _config_value(reward_config, "xi", xi, 1.0)
    eta_value = _config_value(reward_config, "eta", eta, 2.0)
    epsilon_value = _config_value(reward_config, "epsilon", epsilon, 1e-6)
    utility_mode = getattr(reward_config, "utility_mode", "equity")

    node_values: Dict[Any, float] = {}
    poi_records_by_type = get_poi_records_by_type(graph)

    for node in graph.nodes():
        node_values[node] = compute_location_based_accessibility(
            graph=graph,
            human_node=node,
            route_nodes=[node],
            poi_weights=poi_weights,
            poi_records_by_type=poi_records_by_type,
            beta=beta_value,
            gamma=alpha_value,
        )

    if not node_values or population_size <= 0:
        return {
            "node_accessibility": node_values,
            "min_accessibility": 0.0,
            "max_accessibility": 0.0,
            "min_accessibility_node": None,
            "max_accessibility_node": None,
            "utility_lower_bound": 0.0,
            "utility_upper_bound": 0.0,
            "population_size": population_size,
        }

    min_node, min_accessibility = min(
        node_values.items(),
        key=lambda item: (item[1], item[0]),
    )
    max_node, max_accessibility = max(
        node_values.items(),
        key=lambda item: (item[1], item[0]),
    )

    worst_individual_values = {
        f"human_{index}": min_accessibility for index in range(population_size)
    }
    best_individual_values = {
        f"human_{index}": max_accessibility for index in range(population_size)
    }

    utility_lower_bound = compute_utility_by_mode(
        worst_individual_values,
        utility_mode=utility_mode,
        xi=xi_value,
        eta=eta_value,
        epsilon=epsilon_value,
    )
    utility_upper_bound = compute_utility_by_mode(
        best_individual_values,
        utility_mode=utility_mode,
        xi=xi_value,
        eta=eta_value,
        epsilon=epsilon_value,
    )

    return {
        "node_accessibility": node_values,
        "min_accessibility": float(min_accessibility),
        "max_accessibility": float(max_accessibility),
        "min_accessibility_node": min_node,
        "max_accessibility_node": max_node,
        "utility_lower_bound": float(utility_lower_bound),
        "utility_upper_bound": float(utility_upper_bound),
        "population_size": population_size,
        "utility_mode": utility_mode,
    }
