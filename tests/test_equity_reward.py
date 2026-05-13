from accessibility_equity.config import RewardConfig
from accessibility_equity.rewards.equity_reward import (
    compute_population_utility,
    compute_state_utility,
)
from accessibility_equity.rewards.accessibility import compute_location_based_accessibility


def test_population_utility_uses_outer_negative_sign():
    individual_values = {
        "human_0": 1.0,
        "human_1": 2.0,
    }

    assert compute_population_utility(individual_values, xi=1.0, eta=2.0) == -2.25


def test_default_population_utility_increases_with_accessibility():
    low_accessibility = {
        "human_0": 1.0,
        "human_1": 2.0,
    }
    high_accessibility = {
        "human_0": 2.0,
        "human_1": 4.0,
    }

    assert compute_population_utility(high_accessibility) > compute_population_utility(
        low_accessibility
    )


def test_accessibility_counts_explicit_poi_records_not_node_summary():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(0, name="A", poi_type="workplace")
    graph.add_node(1, name="B", poi_type="workplace")
    graph.add_edge(0, 1, length=1.0, speed=1.0, capacity=10)
    graph.add_edge(1, 0, length=1.0, speed=1.0, capacity=10)
    graph.graph["poi_records"] = [
        {"poi_id": "workplace_0", "poi_type": "workplace", "nearest_node": 1},
        {"poi_id": "workplace_1", "poi_type": "workplace", "nearest_node": 1},
    ]

    value = compute_location_based_accessibility(
        graph=graph,
        human_node=0,
        route_nodes=[0],
        poi_weights={"workplace": 1.0, "supermarket": 0.0, "healthcare": 0.0},
        beta=0.0,
        gamma=1.0,
    )

    assert value == 2.0


def test_sum_accessibility_utility_mode_uses_total_accessibility():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node(0, name="A")
    graph.add_node(1, name="B")
    graph.add_edge(0, 1, length=1.0, speed=1.0, capacity=10)
    graph.add_edge(1, 0, length=1.0, speed=1.0, capacity=10)
    graph.graph["poi_records"] = [
        {"poi_id": "workplace_0", "poi_type": "workplace", "nearest_node": 0},
        {"poi_id": "workplace_1", "poi_type": "workplace", "nearest_node": 1},
    ]

    result = compute_state_utility(
        graph=graph,
        human_to_node={"human_0": 0, "human_1": 1},
        human_to_route={"human_0": [0], "human_1": [1]},
        poi_weights={"workplace": 1.0, "supermarket": 0.0, "healthcare": 0.0},
        beta=0.0,
        alpha=1.0,
        reward_config=RewardConfig(utility_mode="sum_accessibility"),
    )

    assert result["total_accessibility"] == 4.0
    assert result["utility"] == 4.0
