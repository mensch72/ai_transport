"""
High-level scenario builder for accessibility-equity experiments.

This module only assembles the separate scenario layers into one object that
can be passed to wrappers and environments. Generation details live in the
specialized modules:

- ``network_builder.py`` builds or loads the network
- ``voronoi_partition.py`` builds spatial regions from intersections
- ``poi_distribution.py`` assigns POIs and residential areas
- ``human_distribution.py`` assigns human homes and demand targets
- ``vehicle_distribution.py`` assigns vehicle initial positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import networkx as nx

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG

from .human_distribution import build_human_distribution
from .network_builder import build_network
from .poi_distribution import build_poi_distribution
from .vehicle_distribution import build_vehicle_distribution


_DEFAULT_SCENARIO = DEFAULT_EXPERIMENT_CONFIG.scenario


@dataclass
class TransportScenario:
    """Fully prepared scenario data for one environment episode family."""

    network: nx.DiGraph
    poi_distribution: Dict[str, Any]
    human_distribution: Dict[str, Any]
    vehicle_distribution: Dict[str, Any]
    initial_agent_positions: Dict[str, Any]
    human_goal_nodes: List[Any] = field(default_factory=list)

    @property
    def initial_state(self) -> Dict[str, Any]:
        """State payload accepted by ``transport_env.reset(options=...)``."""
        return {
            "agent_positions": dict(self.initial_agent_positions),
            "human_aboard": {
                human_id: None
                for human_id in self.human_distribution.get(
                    "human_to_initial_node",
                    {},
                )
            },
            "human_destinations": dict(
                self.human_distribution.get("human_to_target_node", {})
            ),
        }


def build_transport_scenario(
    num_humans: int = _DEFAULT_SCENARIO.num_humans,
    num_vehicles: int = _DEFAULT_SCENARIO.num_vehicles,
    num_nodes: int = _DEFAULT_SCENARIO.num_nodes,
    seed: Optional[int] = None,
    network: Optional[nx.DiGraph] = None,
    network_kwargs: Optional[Dict[str, Any]] = None,
    poi_kwargs: Optional[Dict[str, Any]] = None,
    human_distribution_kwargs: Optional[Dict[str, Any]] = None,
    vehicle_distribution_kwargs: Optional[Dict[str, Any]] = None,
    human_goal_nodes: Optional[List[Any]] = None,
) -> TransportScenario:
    """
    Build a complete transport scenario before creating/resetting the env.
    """
    effective_seed = _DEFAULT_SCENARIO.seed if seed is None else seed
    network_options = dict(network_kwargs or {})
    network_options.setdefault(
        "speed_mean",
        DEFAULT_EXPERIMENT_CONFIG.mobility.edge_speed_kmh,
    )
    poi_options = dict(poi_kwargs or {})
    poi_options.setdefault("central_count", _DEFAULT_SCENARIO.central_count)

    if network is None:
        network = build_network(
            num_nodes=num_nodes,
            seed=effective_seed,
            **network_options,
        )

    poi_distribution = build_poi_distribution(network, **poi_options)

    human_distribution_options = dict(human_distribution_kwargs or {})
    if human_goal_nodes is not None:
        human_distribution_options["target_nodes"] = list(human_goal_nodes)
    else:
        human_distribution_options.setdefault(
            "target_poi_records",
            poi_distribution.get("poi_records", []),
        )
    human_distribution = build_human_distribution(
        graph=network,
        num_humans=num_humans,
        seed=effective_seed,
        **human_distribution_options,
    )

    vehicle_distribution_options = dict(vehicle_distribution_kwargs or {})
    vehicle_distribution_options.setdefault(
        "candidate_nodes",
        list(human_distribution["human_to_initial_node"].values()),
    )
    vehicle_distribution = build_vehicle_distribution(
        graph=network,
        num_vehicles=num_vehicles,
        seed=effective_seed,
        **vehicle_distribution_options,
    )

    initial_agent_positions: Dict[str, Any] = {}
    initial_agent_positions.update(vehicle_distribution["vehicle_to_initial_node"])
    initial_agent_positions.update(human_distribution["human_to_initial_node"])

    return TransportScenario(
        network=network,
        poi_distribution=poi_distribution,
        human_distribution=human_distribution,
        vehicle_distribution=vehicle_distribution,
        initial_agent_positions=initial_agent_positions,
        human_goal_nodes=list(human_distribution["human_goal_nodes"]),
    )
