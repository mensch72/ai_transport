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

from typing import Any, Dict, Optional, Sequence

import networkx as nx

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG, RewardConfig

from .accessibility import (
    compute_location_based_accessibility,
    compute_population_accessibility,
    get_poi_records_by_type,
)


def _resolve_reward_config(reward_config: Optional[RewardConfig]) -> RewardConfig:
    return DEFAULT_EXPERIMENT_CONFIG.reward if reward_config is None else reward_config


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
    reward_config: Optional[RewardConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the population utility U(s_t) from the current state description.

    Note:
        The implementation uses the project's accessibility module, where the
        within-category diminishing-returns parameter is named `gamma`.
        Here it is passed from the formulation's `alpha`.
    """
    resolved = _resolve_reward_config(reward_config)
    beta_value = resolved.beta if beta is None else float(beta)
    alpha_value = resolved.alpha if alpha is None else float(alpha)
    xi_value = resolved.xi if xi is None else float(xi)
    eta_value = resolved.eta if eta is None else float(eta)
    epsilon_value = resolved.epsilon if epsilon is None else float(epsilon)
    utility_mode = resolved.utility_mode

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
    reward_config: Optional[RewardConfig] = None,
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
    reward_config: Optional[RewardConfig] = None,
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
    resolved = _resolve_reward_config(reward_config)
    beta_value = resolved.beta if beta is None else float(beta)
    alpha_value = resolved.alpha if alpha is None else float(alpha)
    xi_value = resolved.xi if xi is None else float(xi)
    eta_value = resolved.eta if eta is None else float(eta)
    epsilon_value = resolved.epsilon if epsilon is None else float(epsilon)
    utility_mode = resolved.utility_mode

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
