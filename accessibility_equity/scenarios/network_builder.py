"""
Network construction utilities.

This module provides the first scenario-level network builder for the
accessibility-equity project. The current implementation follows the same
overall idea as `ai_transport.envs.transport_env.create_random_2d_network`:

- sample node coordinates in 2D
- build a geometric neighborhood structure with Delaunay triangulation
- convert the resulting undirected links into a directed road network
- assign edge attributes such as length, speed, and capacity
- enforce strong connectivity so the transport simulator remains usable

The resulting graph is intended to be used as the base network before adding
scenario-specific layers such as hierarchy labels, Voronoi partitions, POIs,
and initial human distributions.

Unit convention:

- edge ``length`` is measured in kilometers
- edge ``speed`` is measured in kilometers per hour
- movement time is measured in hours because duration = length / speed
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG


DEFAULT_EDGE_SPEED_KMH = DEFAULT_EXPERIMENT_CONFIG.mobility.edge_speed_kmh


def _add_edge_with_attributes(
    graph: nx.DiGraph,
    coords: np.ndarray,
    rng: np.random.RandomState,
    source: int,
    target: int,
    speed_mean: float,
    capacity_mean: float,
) -> None:
    """
    Add one directed edge with standard synthetic road attributes.
    """
    dx = coords[target, 0] - coords[source, 0]
    dy = coords[target, 1] - coords[source, 1]
    length = float(np.sqrt(dx**2 + dy**2))
    speed = max(float(speed_mean), 0.1)
    capacity = max(float(rng.exponential(scale=capacity_mean)), 1.0)
    graph.add_edge(source, target, length=length, speed=speed, capacity=capacity)


def _ensure_strong_connectivity(
    graph: nx.DiGraph,
    coords: np.ndarray,
    rng: np.random.RandomState,
    speed_mean: float,
    capacity_mean: float,
) -> None:
    """
    Add bridging edges until the directed graph becomes strongly connected.

    The strategy mirrors the logic already used in `ai_transport`: connect the
    strongly connected components in a directed cycle, and if that is still not
    sufficient, add reverse edges as a final fallback.
    """
    if nx.is_strongly_connected(graph):
        return

    sccs = list(nx.strongly_connected_components(graph))
    reps = [min(scc) for scc in sccs]

    for i in range(len(reps)):
        source = reps[i]
        target = reps[(i + 1) % len(reps)]
        if graph.has_edge(source, target):
            continue

        _add_edge_with_attributes(
            graph=graph,
            coords=coords,
            rng=rng,
            source=source,
            target=target,
            speed_mean=speed_mean,
            capacity_mean=capacity_mean,
        )

    if nx.is_strongly_connected(graph):
        return

    for u, v in list(graph.edges()):
        if graph.has_edge(v, u):
            continue

        data = graph[u][v]
        graph.add_edge(
            v,
            u,
            length=data.get("length", 1.0),
            speed=data.get("speed", 1.0),
            capacity=data.get("capacity", 1.0),
        )
        if nx.is_strongly_connected(graph):
            break


def _add_center_biased_edges(
    graph: nx.DiGraph,
    coords: np.ndarray,
    rng: np.random.RandomState,
    speed_mean: float,
    capacity_mean: float,
    central_node_count: int = 0,
    extra_neighbors_per_central_node: Optional[int] = None,
) -> None:
    """
    Add extra bidirectional links from spatially central nodes to nearby nodes.

    The base Delaunay graph is geometric, but for small synthetic networks it
    can still give peripheral nodes similar or higher degree. This additional
    layer gently nudges the topology toward an urban-center pattern where nodes
    closest to the coordinate centroid are more connected.
    """
    if central_node_count <= 0:
        return

    center = coords.mean(axis=0)
    node_order = list(range(len(coords)))
    ranked_by_centrality = sorted(
        node_order,
        key=lambda node: (
            float(np.linalg.norm(coords[node] - center)),
            node,
        ),
    )
    central_nodes = ranked_by_centrality[: min(central_node_count, len(ranked_by_centrality))]
    if extra_neighbors_per_central_node is None:
        extra_neighbors_per_central_node = max(0, len(coords) - 1)
    if extra_neighbors_per_central_node <= 0:
        return

    for source in central_nodes:
        neighbors = sorted(
            (node for node in node_order if node != source),
            key=lambda node: (
                float(np.linalg.norm(coords[node] - coords[source])),
                node,
            ),
        )
        for target in neighbors[: min(extra_neighbors_per_central_node, len(neighbors))]:
            if not graph.has_edge(source, target):
                _add_edge_with_attributes(
                    graph=graph,
                    coords=coords,
                    rng=rng,
                    source=source,
                    target=target,
                    speed_mean=speed_mean,
                    capacity_mean=capacity_mean,
                )
            if not graph.has_edge(target, source):
                _add_edge_with_attributes(
                    graph=graph,
                    coords=coords,
                    rng=rng,
                    source=target,
                    target=source,
                    speed_mean=speed_mean,
                    capacity_mean=capacity_mean,
                )


def create_random_2d_network(
    num_nodes: int = 10,
    bidirectional_prob: float = 0.85,
    speed_mean: float = DEFAULT_EDGE_SPEED_KMH,
    capacity_mean: float = 10.0,
    coord_mean: float = 0.0,
    coord_std: float = 10.0,
    central_node_count: int = 0,
    extra_neighbors_per_central_node: Optional[int] = None,
    seed: Optional[int] = None,
) -> nx.DiGraph:
    """
    Create a directed 2D road network for scenario generation.

    This function intentionally follows the design already used in
    `ai_transport`, so that the new project stays compatible with the
    assumptions of the underlying simulator.

    Args:
        num_nodes: Number of network nodes to generate.
        bidirectional_prob: Probability that a link is added in both directions.
        speed_mean: Vehicle speed in km/h assigned to generated edges.
        capacity_mean: Mean for the exponential edge-capacity distribution.
        coord_mean: Mean of the Gaussian node-coordinate distribution.
        coord_std: Standard deviation of the Gaussian node-coordinate distribution.
        central_node_count: Number of spatially central nodes to give extra
            connectivity. The default is 0, which keeps the original geometric
            edge structure.
        extra_neighbors_per_central_node: Number of nearest neighbors each
            central node should be linked with bidirectionally. If ``None``,
            each central node links to all other nodes.
        seed: Optional random seed.

    Returns:
        A strongly connected `networkx.DiGraph` whose nodes carry `name`, `x`,
        and `y` attributes and whose edges carry `length` in km, `speed` in
        km/h, and `capacity` attributes.
    """
    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3 for Delaunay triangulation.")

    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
    coords = rng.normal(loc=coord_mean, scale=coord_std, size=(num_nodes, 2))
    triangulation = Delaunay(coords)

    graph = nx.DiGraph()

    for node in range(num_nodes):
        graph.add_node(
            node,
            name=f"Node_{node}",
            x=float(coords[node, 0]),
            y=float(coords[node, 1]),
        )

    undirected_edges = set()
    for simplex in triangulation.simplices:
        for i in range(3):
            u = simplex[i]
            v = simplex[(i + 1) % 3]
            undirected_edges.add((min(u, v), max(u, v)))

    for u, v in undirected_edges:
        dx = coords[v, 0] - coords[u, 0]
        dy = coords[v, 1] - coords[u, 1]
        length = float(np.sqrt(dx**2 + dy**2))
        speed = max(float(speed_mean), 0.1)
        capacity = max(float(rng.exponential(scale=capacity_mean)), 1.0)

        if rng.random() < bidirectional_prob:
            graph.add_edge(u, v, length=length, speed=speed, capacity=capacity)
            graph.add_edge(v, u, length=length, speed=speed, capacity=capacity)
        else:
            if rng.random() < 0.5:
                graph.add_edge(u, v, length=length, speed=speed, capacity=capacity)
            else:
                graph.add_edge(v, u, length=length, speed=speed, capacity=capacity)

        # Match the baseline transport_env.create_random_2d_network behavior:
        # repair strong connectivity during edge construction, not only after
        # all Delaunay edges are added. This keeps the scenario generator close
        # to the original simulator network process.
        _ensure_strong_connectivity(
            graph=graph,
            coords=coords,
            rng=rng,
            speed_mean=speed_mean,
            capacity_mean=capacity_mean,
        )

    _add_center_biased_edges(
        graph=graph,
        coords=coords,
        rng=rng,
        speed_mean=speed_mean,
        capacity_mean=capacity_mean,
        central_node_count=central_node_count,
        extra_neighbors_per_central_node=extra_neighbors_per_central_node,
    )
    _ensure_strong_connectivity(
        graph=graph,
        coords=coords,
        rng=rng,
        speed_mean=speed_mean,
        capacity_mean=capacity_mean,
    )

    return graph


def build_network(
    num_nodes: int = 10,
    seed: Optional[int] = None,
    **kwargs,
) -> nx.DiGraph:
    """
    Convenience entry point for scenario code.

    For now this simply delegates to `create_random_2d_network`, but it gives
    the project one stable function name to call from higher-level scenario
    builders. Later this function can be extended to support imported real
    networks, different random generators, or hierarchy-aware initialization.
    """
    return create_random_2d_network(num_nodes=num_nodes, seed=seed, **kwargs)
