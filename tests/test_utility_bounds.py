from accessibility_equity.rewards import compute_scenario_utility_bounds
from accessibility_equity.scenarios import build_transport_scenario


def test_scenario_utility_bounds_are_ordered():
    scenario = build_transport_scenario(
        num_humans=4,
        num_vehicles=1,
        num_nodes=8,
        seed=12,
    )

    bounds = compute_scenario_utility_bounds(
        graph=scenario.network,
        population_size=4,
    )

    assert bounds["min_accessibility"] <= bounds["max_accessibility"]
    assert bounds["utility_lower_bound"] <= bounds["utility_upper_bound"]
    assert bounds["utility_upper_bound"] <= 0.0
