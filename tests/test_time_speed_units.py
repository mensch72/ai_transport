import networkx as nx
import numpy as np

from accessibility_equity.envs.transport_env import (
    DEFAULT_HUMAN_WALKING_SPEED_KMH,
    DEFAULT_VEHICLE_SPEED_KMH,
    parallel_env,
)
from accessibility_equity.scenarios import build_transport_scenario


def _two_node_network(length_km: float, vehicle_speed_kmh: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(0, name="A", x=0.0, y=0.0)
    graph.add_node(1, name="B", x=length_km, y=0.0)
    graph.add_edge(0, 1, length=length_km, speed=vehicle_speed_kmh, capacity=10)
    graph.add_edge(1, 0, length=length_km, speed=vehicle_speed_kmh, capacity=10)
    return graph


def test_default_speeds_use_kmh_convention():
    env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        network=_two_node_network(length_km=1.0, vehicle_speed_kmh=30.0),
    )

    assert env.agent_attributes["human_0"]["speed"] == DEFAULT_HUMAN_WALKING_SPEED_KMH
    assert env.agent_attributes["vehicle_0"]["speed"] == DEFAULT_VEHICLE_SPEED_KMH


def test_generated_scenario_edges_default_to_30_kmh_vehicle_speed():
    scenario = build_transport_scenario(
        num_humans=2,
        num_vehicles=1,
        num_nodes=8,
        seed=11,
    )

    edge_speeds = {
        float(data["speed"])
        for _u, _v, data in scenario.network.edges(data=True)
    }

    assert edge_speeds == {DEFAULT_VEHICLE_SPEED_KMH}


def test_vehicle_travels_30_km_edge_in_one_hour():
    env = parallel_env(
        num_humans=0,
        num_vehicles=1,
        network=_two_node_network(length_km=30.0, vehicle_speed_kmh=30.0),
    )
    env.reset(
        seed=1,
        options={"initial_state": {"agent_positions": {"vehicle_0": 0}}},
    )
    env._set_step_type_for_testing("departing")

    env.step({"vehicle_0": 1})

    assert np.isclose(env.real_time, 1.0)
    assert env.agent_positions["vehicle_0"] == 1


def test_vehicle_uses_vehicle_speed_when_vehicle_is_slower_than_edge():
    env = parallel_env(
        num_humans=0,
        num_vehicles=1,
        vehicle_speeds=[20.0],
        network=_two_node_network(length_km=40.0, vehicle_speed_kmh=30.0),
    )
    env.reset(
        seed=1,
        options={"initial_state": {"agent_positions": {"vehicle_0": 0}}},
    )
    env._set_step_type_for_testing("departing")

    env.step({"vehicle_0": 1})

    assert np.isclose(env.real_time, 2.0)
    assert env.agent_positions["vehicle_0"] == 1


def test_vehicle_uses_edge_speed_when_edge_is_slower_than_vehicle():
    env = parallel_env(
        num_humans=0,
        num_vehicles=1,
        vehicle_speeds=[50.0],
        network=_two_node_network(length_km=20.0, vehicle_speed_kmh=10.0),
    )
    env.reset(
        seed=1,
        options={"initial_state": {"agent_positions": {"vehicle_0": 0}}},
    )
    env._set_step_type_for_testing("departing")

    env.step({"vehicle_0": 1})

    assert np.isclose(env.real_time, 2.0)
    assert env.agent_positions["vehicle_0"] == 1


def test_human_walks_3_km_edge_in_one_hour():
    env = parallel_env(
        num_humans=1,
        num_vehicles=0,
        network=_two_node_network(length_km=3.0, vehicle_speed_kmh=30.0),
    )
    env.reset(
        seed=1,
        options={
            "initial_state": {
                "agent_positions": {"human_0": 0},
                "human_aboard": {"human_0": None},
            }
        },
    )
    env._set_step_type_for_testing("departing")

    env.step({"human_0": 1})

    assert np.isclose(env.real_time, 1.0)
    assert env.agent_positions["human_0"] == 1
