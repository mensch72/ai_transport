"""
Voronoi-based spatial partition logic.

This module follows the same overall idea used in
`Beijing_simulation/2.get_voronoi.py`, but adapts it to the small synthetic
networks used in `accessibility_equity`.

The key workflow is:

- use network nodes as generating points
- build a clipping polygon from the node convex hull plus a buffer
- construct Voronoi cells
- clip the cells to the study-area polygon
- classify cells into central / peripheral areas using network degree:
  the highest-degree nodes own central cells, all others own peripheral cells

At the current stage the implementation works directly with in-memory geometric
objects and does not require exporting GIS files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import MultiPoint, Polygon

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG


DEFAULT_CENTRAL_COUNT = DEFAULT_EXPERIMENT_CONFIG.scenario.central_count


def _extract_coordinates(graph: nx.DiGraph) -> Tuple[List[Any], np.ndarray]:
    """
    Extract 2D node coordinates from the graph.

    Supports the same coordinate conventions used elsewhere in the project:
    either separate `x` / `y` attributes or a `pos` tuple.
    """
    nodes = list(graph.nodes())
    coords = []

    for node in nodes:
        data = graph.nodes[node]
        if "x" in data and "y" in data:
            coords.append([float(data["x"]), float(data["y"])])
        elif "pos" in data:
            pos = data["pos"]
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                coords.append([float(pos[0]), float(pos[1])])
            else:
                raise ValueError(f"Node {node} has invalid 'pos' attribute: {pos}")
        else:
            raise ValueError(
                f"Node {node} is missing coordinates. Expected 'x'/'y' or 'pos'."
            )

    return nodes, np.array(coords)


def build_clip_polygon(
    graph: nx.DiGraph,
    buffer_distance: Optional[float] = None,
    buffer_ratio: float = 0.1,
) -> Polygon:
    """
    Build the study-area polygon from the network convex hull plus a buffer.

    This mirrors the logic used in the Beijing workflow where a clipping polygon
    is required before Voronoi polygons become finite and meaningful.

    Args:
        graph: Network whose nodes define the study area.
        buffer_distance: Optional absolute buffer distance. If `None`, a buffer
            is derived from the network span using `buffer_ratio`.
        buffer_ratio: Relative buffer used when `buffer_distance` is not given.

    Returns:
        A shapely polygon that can be used to clip Voronoi cells.
    """
    nodes, coords = _extract_coordinates(graph)
    if len(nodes) < 3:
        raise ValueError("At least three nodes are needed to build a clip polygon.")

    multipoint = MultiPoint([tuple(coord) for coord in coords])
    hull = multipoint.convex_hull

    if buffer_distance is None:
        span_x = float(coords[:, 0].max() - coords[:, 0].min())
        span_y = float(coords[:, 1].max() - coords[:, 1].min())
        scale = max(span_x, span_y, 1.0)
        buffer_distance = scale * buffer_ratio

    clip_polygon = hull.buffer(buffer_distance)
    return clip_polygon


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None):
    """
    Reconstruct finite Voronoi regions from a 2D Voronoi diagram.

    This helper follows the common SciPy recipe for turning infinite Voronoi
    regions into large finite polygons that can then be clipped to the study
    area.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input.")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        region_vertices = np.asarray([new_vertices[v] for v in new_region])
        region_center = region_vertices.mean(axis=0)
        angles = np.arctan2(
            region_vertices[:, 1] - region_center[1],
            region_vertices[:, 0] - region_center[0],
        )
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def classify_voronoi_cells_by_node_degree(
    graph: nx.DiGraph,
    nodes: List[Any],
    central_count: int = DEFAULT_CENTRAL_COUNT,
) -> Dict[str, Any]:
    """
    Classify Voronoi cells from their generator nodes' network degree.

    The top ``central_count`` nodes with the largest total directed degree are
    central areas. All other node-owned cells are peripheral areas. Ties are
    resolved by spatial closeness to the coordinate centroid and then node id
    string for deterministic scenarios.
    """
    node_degrees = {node: int(graph.degree(node)) for node in nodes}
    coordinates = {}
    for node in nodes:
        data = graph.nodes[node]
        if "x" in data and "y" in data:
            coordinates[node] = np.array([float(data["x"]), float(data["y"])])
        else:
            pos = data["pos"]
            coordinates[node] = np.array([float(pos[0]), float(pos[1])])
    centroid = np.mean(np.array(list(coordinates.values())), axis=0)
    node_center_distances = {
        node: float(np.linalg.norm(coordinates[node] - centroid))
        for node in nodes
    }
    ranked_nodes = sorted(
        nodes,
        key=lambda node: (
            -node_degrees[node],
            node_center_distances[node],
            str(node),
        ),
    )
    central_nodes = ranked_nodes[: min(central_count, len(ranked_nodes))]
    central_node_set = set(central_nodes)
    peripheral_nodes = [node for node in nodes if node not in central_node_set]
    node_to_area_type = {
        node: "central" if node in central_node_set else "peripheral"
        for node in nodes
    }

    return {
        "node_degrees": node_degrees,
        "node_center_distances": node_center_distances,
        "central_nodes": central_nodes,
        "peripheral_nodes": peripheral_nodes,
        "node_to_area_type": node_to_area_type,
    }


classify_voronoi_cells_by_spatial_centrality = classify_voronoi_cells_by_node_degree


def build_voronoi_partition(
    graph: nx.DiGraph,
    clip_polygon: Optional[Polygon] = None,
    buffer_distance: Optional[float] = None,
    buffer_ratio: float = 0.1,
    central_count: int = DEFAULT_CENTRAL_COUNT,
) -> Dict[str, Any]:
    """
    Build Voronoi polygons for the network nodes and clip them to the study area.

    Args:
        graph: Network whose nodes define the Voronoi generators.
        clip_polygon: Optional externally provided clipping polygon.
        buffer_distance: Optional absolute buffer for the default clipping polygon.
        buffer_ratio: Relative buffer for the default clipping polygon.

    Returns:
        A dictionary with:
        - `clip_polygon`
        - `node_to_cell`
        - `cells`
        - `nodes`
        - `coordinates`
        - `node_to_area_type`
        - `central_nodes`
        - `peripheral_nodes`
        - `node_degrees`
    """
    nodes, coords = _extract_coordinates(graph)
    if len(nodes) < 3:
        raise ValueError("At least three nodes are needed to build Voronoi cells.")

    if clip_polygon is None:
        clip_polygon = build_clip_polygon(
            graph,
            buffer_distance=buffer_distance,
            buffer_ratio=buffer_ratio,
        )

    vor = Voronoi(coords)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    cells: Dict[int, Polygon] = {}
    node_to_cell: Dict[Any, int] = {}

    for i, region in enumerate(regions):
        polygon = Polygon(vertices[region])
        clipped = polygon.intersection(clip_polygon)
        if clipped.is_empty:
            continue

        cells[i] = clipped
        node_to_cell[nodes[i]] = i

    area_info = classify_voronoi_cells_by_node_degree(
        graph=graph,
        nodes=nodes,
        central_count=central_count,
    )
    for node, area_type in area_info["node_to_area_type"].items():
        graph.nodes[node]["voronoi_area_type"] = area_type
        graph.nodes[node]["central_level"] = area_type

    return {
        "clip_polygon": clip_polygon,
        "node_to_cell": node_to_cell,
        "cells": cells,
        "nodes": nodes,
        "coordinates": coords,
        **area_info,
    }
