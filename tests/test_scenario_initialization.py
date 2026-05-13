"""
Scenario initialization tests.
"""

from accessibility_equity.scenarios import build_transport_scenario


def test_vehicles_start_where_humans_can_board_by_default():
    """Default scenario placement should collocate vehicles with human starts."""
    scenario = build_transport_scenario(
        num_humans=4,
        num_vehicles=2,
        num_nodes=10,
        seed=7,
    )

    human_nodes = set(
        scenario.human_distribution["human_to_initial_node"].values()
    )
    vehicle_nodes = set(
        scenario.vehicle_distribution["vehicle_to_initial_node"].values()
    )

    assert vehicle_nodes
    assert vehicle_nodes.issubset(human_nodes)
    assert all(
        scenario.network.nodes[node].get("area_type") == "residential"
        for node in vehicle_nodes
    )
