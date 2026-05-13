"""
Scenario inspection script.

Generate a complete TransportScenario and print a compact summary:

- network
- POI / residential assignment
- human homes and targets
- vehicle initial positions
- initial state passed to the environment
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG, save_experiment_config
from accessibility_equity.scenarios import build_transport_scenario
from accessibility_equity.rewards import compute_scenario_utility_bounds
from accessibility_equity.visualization import save_scenario_figure


def main():
    """Generate a scenario and print a compact summary."""
    base_config = DEFAULT_EXPERIMENT_CONFIG
    scenario_config = base_config.scenario
    parser = argparse.ArgumentParser(description="Inspect a generated transport scenario.")
    parser.add_argument("--nodes", type=int, default=scenario_config.num_nodes, help="Number of network nodes.")
    parser.add_argument("--humans", type=int, default=scenario_config.num_humans, help="Number of human agents.")
    parser.add_argument("--vehicles", type=int, default=scenario_config.num_vehicles, help="Number of vehicle agents.")
    parser.add_argument("--seed", type=int, default=scenario_config.seed, help="Random seed.")
    parser.add_argument(
        "--central-count",
        type=int,
        default=scenario_config.central_count,
        help="Number of network nodes classified as central areas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the generated scenario image.",
    )
    args = parser.parse_args()
    scenario_config = scenario_config.with_overrides(
        num_nodes=args.nodes,
        num_humans=args.humans,
        num_vehicles=args.vehicles,
        seed=args.seed,
        central_count=args.central_count,
    )
    experiment_config = base_config.with_overrides(scenario=scenario_config)

    scenario = build_transport_scenario(
        num_humans=scenario_config.num_humans,
        num_vehicles=scenario_config.num_vehicles,
        num_nodes=scenario_config.num_nodes,
        seed=scenario_config.seed,
        network_kwargs={"speed_mean": experiment_config.mobility.edge_speed_kmh},
        poi_kwargs=scenario_config.poi_kwargs,
    )
    graph = scenario.network
    poi_distribution = scenario.poi_distribution
    human_distribution = scenario.human_distribution
    vehicle_distribution = scenario.vehicle_distribution
    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "outputs" / f"scenario_seed{scenario_config.seed}_{scenario_config.num_nodes}nodes.png"
    elif not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    config_path = output_path.with_name("config.json")

    print("=" * 70)
    print("Transport Scenario Inspection")
    print("=" * 70)
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Number of humans: {scenario_config.num_humans}")
    print(f"Number of vehicles: {scenario_config.num_vehicles}")
    print(f"Central area nodes requested: {scenario_config.central_count}")
    print(
        "Mobility: "
        f"human={experiment_config.mobility.human_walking_speed_kmh:g} km/h, "
        f"vehicle={experiment_config.mobility.vehicle_speed_kmh:g} km/h, "
        f"edge={experiment_config.mobility.edge_speed_kmh:g} km/h"
    )
    print(
        "Reward: "
        f"utility_mode={experiment_config.reward.utility_mode}, "
        f"beta={experiment_config.reward.beta:g}, "
        f"alpha={experiment_config.reward.alpha:g}, "
        f"xi={experiment_config.reward.xi:g}, "
        f"eta={experiment_config.reward.eta:g}"
    )

    area_counts = Counter(
        data.get("area_type", "unknown")
        for _, data in graph.nodes(data=True)
    )
    region_counts = Counter(
        data.get("central_level", "unknown")
        for _, data in graph.nodes(data=True)
    )
    print("\nVoronoi region levels:")
    for level, count in sorted(region_counts.items()):
        print(f"  {level}: {count}")

    print("\nResidential / POI area anchors:")
    for area_type, count in sorted(area_counts.items()):
        print(f"  {area_type}: {count}")
    print(f"  explicit POI records: {len(poi_distribution['poi_records'])}")

    print("\nSample POIs:")
    for record in poi_distribution["poi_records"][:8]:
        print(
            f"  {record['poi_id']}: type={record['poi_type']}, "
            f"node={record['node']}, cell={record['cell_id']}, "
            f"x={record['x']:.2f}, y={record['y']:.2f}"
        )

    print("\nHumans:")
    for human_id, record in human_distribution["human_records"].items():
        target_poi = record.get("target_poi") or {}
        target_label = target_poi.get("poi_id", record["target_node"])
        print(
            f"  {human_id}: home={record['home_node']}, "
            f"start={record['initial_node']}, "
            f"target_node={record['target_node']}, target_poi={target_label}"
        )

    print("\nVehicles:")
    for vehicle_id, record in vehicle_distribution["vehicle_records"].items():
        print(f"  {vehicle_id}: start={record['initial_node']}")

    print("\nInitial state for env.reset(options=...):")
    print(f"  agent_positions: {scenario.initial_state['agent_positions']}")
    print(f"  human_aboard: {scenario.initial_state['human_aboard']}")
    print(f"  human_destinations: {scenario.initial_state['human_destinations']}")

    print("\nSample network nodes:")
    for node, data in list(graph.nodes(data=True))[:8]:
        print(
            f"  node {node}: x={data.get('x', 0.0):.2f}, "
            f"y={data.get('y', 0.0):.2f}, "
            f"area={data.get('area_type')}, "
            f"poi={data.get('poi_type')}, "
            f"level={data.get('central_level')}, "
            f"cell={data.get('voronoi_cell_id')}"
        )

    print("\nNetwork edges:")
    for u, v, data in graph.edges(data=True):
        print(
            f"  {u} -> {v}: length={float(data.get('length', 0.0)):.2f} km, "
            f"edge_speed={float(data.get('speed', 0.0)):.1f} km/h"
        )

    utility_bounds = compute_scenario_utility_bounds(
        graph=graph,
        population_size=scenario_config.num_humans,
        reward_config=experiment_config.reward,
    )
    print("\nTheoretical utility range for this scenario (without time):")
    print(
        "  accessibility range: "
        f"X_min={utility_bounds['min_accessibility']:.6f} "
        f"(node {utility_bounds['min_accessibility_node']}), "
        f"X_max={utility_bounds['max_accessibility']:.6f} "
        f"(node {utility_bounds['max_accessibility_node']})"
    )
    print(
        "  utility range: "
        f"lower={utility_bounds['utility_lower_bound']:.6g}, "
        f"upper={utility_bounds['utility_upper_bound']:.6g}"
    )

    footer_lines = [
        (
            "Utility bounds without time: "
            f"lower={utility_bounds['utility_lower_bound']:.3g} "
            f"uses X_h={utility_bounds['min_accessibility']:.3f} "
            f"from node {utility_bounds['min_accessibility_node']}; "
            f"upper={utility_bounds['utility_upper_bound']:.3g} "
            f"uses X_h={utility_bounds['max_accessibility']:.3f} "
            f"from node {utility_bounds['max_accessibility_node']}"
        )
    ]

    save_scenario_figure(
        scenario,
        output_path,
        show_node_labels=True,
        show_edge_length_table=True,
        footer_lines=footer_lines,
        utility_bounds=utility_bounds,
    )
    print(f"\nScenario image saved to: {output_path}")
    save_experiment_config(experiment_config, config_path)
    print(f"Resolved config saved to: {config_path}")


if __name__ == "__main__":
    main()
