"""
Shared visualization logic for static scenarios and episode frames.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon


AREA_COLORS = {
    "residential": "#e6bd45",
    "workplace": "#8d7bb8",
    "supermarket": "#7fbe7e",
    "healthcare": "#ff5a52",
}

HUMAN_COLOR = "#1899d4"
HUMAN_LINE_COLOR = "#67bfe8"
VEHICLE_COLOR = "#8b7356"
VEHICLE_GUIDE_COLOR = "#a16207"

REGION_COLORS = {
    "central": "#dcebf7",
    "peripheral": "#f3eadf",
}

POI_MARKERS = {
    "workplace": "s",
    "supermarket": "D",
    "healthcare": "P",
}

HOUSE_MARKER = MplPath(
    [
        (-0.55, -0.45),
        (-0.55, 0.05),
        (0.0, 0.55),
        (0.55, 0.05),
        (0.55, -0.45),
        (-0.55, -0.45),
    ],
    [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO, MplPath.CLOSEPOLY],
)

CAR_MARKER = MplPath(
    [
        (-0.75, -0.35),
        (-0.75, 0.25),
        (0.75, 0.25),
        (0.75, -0.35),
        (0.5, -0.35),
        (0.42, -0.52),
        (0.18, -0.52),
        (0.1, -0.35),
        (-0.22, -0.35),
        (-0.3, -0.52),
        (-0.54, -0.52),
        (-0.62, -0.35),
        (-0.75, -0.35),
    ],
    [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ],
)


def _node_xy(graph, node):
    data = graph.nodes[node]
    return float(data.get("x", 0.0)), float(data.get("y", 0.0))


def _position_xy(graph, position):
    if position is None:
        return None
    if isinstance(position, tuple):
        edge, coord = position
        if not edge or len(edge) < 2:
            return None
        u, v = edge[0], edge[1]
        x1, y1 = _node_xy(graph, u)
        x2, y2 = _node_xy(graph, v)
        edge_length = float(graph[u][v].get("length", 0.0))
        if edge_length <= 1e-9:
            return x1, y1
        t = max(0.0, min(float(coord) / edge_length, 1.0))
        return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
    return _node_xy(graph, position)


def _draw_voronoi_cells(ax, graph):
    for _, data in graph.nodes(data=True):
        geometry = data.get("voronoi_geometry")
        if geometry is None or geometry.is_empty:
            continue

        polygons = [geometry]
        if hasattr(geometry, "geoms"):
            polygons = list(geometry.geoms)

        for polygon in polygons:
            if not hasattr(polygon, "exterior"):
                continue
            coords = list(polygon.exterior.coords)
            patch = MplPolygon(
                coords,
                closed=True,
                facecolor=REGION_COLORS.get(data.get("central_level"), "#eeeeee"),
                edgecolor="#ffffff",
                linewidth=0.7,
                alpha=0.45,
                zorder=1,
            )
            ax.add_patch(patch)


def _draw_edge_length_label(ax, x1, y1, x2, y2, length_value, offset=0.0):
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    dx = x2 - x1
    dy = y2 - y1
    edge_span = math.hypot(dx, dy)
    if edge_span > 1e-9 and offset:
        mid_x += -dy / edge_span * offset
        mid_y += dx / edge_span * offset

    ax.text(
        mid_x,
        mid_y,
        f"{float(length_value):.1f} km",
        fontsize=6.5,
        color="#111827",
        ha="center",
        va="center",
        zorder=8,
        bbox={
            "boxstyle": "round,pad=0.15",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "linewidth": 0.35,
            "alpha": 0.78,
        },
    )


def _draw_network(ax, graph, show_node_labels=False):
    node_to_xy = {}
    xs = []
    ys = []
    for node in graph.nodes():
        x, y = _node_xy(graph, node)
        node_to_xy[node] = (x, y)
        xs.append(x)
        ys.append(y)

    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0
    twin_offset = max(0.0025 * max(span_x, span_y), 0.025)

    drawn_two_way = set()
    for u, v in graph.edges():
        if u == v:
            continue

        is_two_way = graph.has_edge(v, u)
        if is_two_way:
            edge_key = frozenset((u, v))
            if edge_key in drawn_two_way:
                continue
            drawn_two_way.add(edge_key)
        x1, y1 = node_to_xy[u]
        x2, y2 = node_to_xy[v]
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue

        normal_x = -dy / length
        normal_y = dx / length

        if is_two_way:
            for direction_sign in (-1.0, 1.0):
                offset_x = normal_x * twin_offset * direction_sign
                offset_y = normal_y * twin_offset * direction_sign
                ax.plot(
                    [x1 + offset_x, x2 + offset_x],
                    [y1 + offset_y, y2 + offset_y],
                    color="#4b5563",
                    linewidth=0.62,
                    alpha=0.62,
                    zorder=2,
                    solid_capstyle="round",
                )
        else:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="#6b7280",
                linewidth=0.65,
                alpha=0.35,
                zorder=2,
                solid_capstyle="round",
            )

    ax.scatter(xs, ys, s=18, color="#374151", alpha=0.78, zorder=6, label="network node")
    if show_node_labels:
        for node, (x, y) in node_to_xy.items():
            ax.text(
                x,
                y,
                str(node),
                fontsize=8,
                fontweight="bold",
                color="#111827",
                ha="center",
                va="center",
                zorder=9,
                bbox={
                    "boxstyle": "circle,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#374151",
                    "linewidth": 0.45,
                    "alpha": 0.9,
                },
            )


def _draw_utility_bound_nodes(ax, graph, utility_bounds=None):
    if not utility_bounds:
        return

    markers = [
        (
            utility_bounds.get("max_accessibility_node"),
            "best",
            "#16a34a",
            (8, 8),
        ),
        (
            utility_bounds.get("min_accessibility_node"),
            "worst",
            "#dc2626",
            (8, -12),
        ),
    ]
    for node, label, color, offset in markers:
        if node is None or node not in graph.nodes:
            continue
        x, y = _node_xy(graph, node)
        ax.scatter(
            [x],
            [y],
            s=285,
            marker="o",
            facecolor="none",
            edgecolor=color,
            linewidth=1.6,
            zorder=10,
        )
        ax.annotate(
            label,
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            zorder=11,
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.45,
                "alpha": 0.88,
            },
        )


def _edge_length_table_text(graph):
    lines = []
    for u, v, data in graph.edges(data=True):
        length = float(data.get("length", 0.0))
        lines.append(f"{u}->{v}: {length:.1f} km")
    return lines


def _add_edge_length_table(fig, graph, footer_lines=None):
    lines = _edge_length_table_text(graph)
    footer_lines = list(footer_lines or [])
    if not lines and not footer_lines:
        return

    column_count = 4 if len(lines) > 12 else 3
    rows = []
    for start in range(0, len(lines), column_count):
        rows.append("    ".join(lines[start : start + column_count]))

    text_parts = []
    if footer_lines:
        text_parts.extend(footer_lines)
    if rows:
        edge_text = "Edge lengths: " + rows[0]
        if len(rows) > 1:
            edge_text += "\n" + "\n".join(" " * 14 + row for row in rows[1:])
        text_parts.append(edge_text)

    text = "\n".join(text_parts)

    fig.text(
        0.5,
        0.02,
        text,
        ha="center",
        va="bottom",
        fontsize=7,
        color="#111827",
        family="monospace",
    )


def _draw_pois(ax, poi_records):
    for poi_type, marker in POI_MARKERS.items():
        records = [record for record in poi_records if record["poi_type"] == poi_type]
        if not records:
            continue
        ax.scatter(
            [record["x"] for record in records],
            [record["y"] for record in records],
            s=54,
            marker=marker,
            edgecolor="#1f2937",
            linewidth=0.75,
            color=AREA_COLORS.get(poi_type, "#eeeeee"),
            zorder=5,
            label=poi_type,
        )


def _draw_residential_places(ax, residential_records):
    if not residential_records:
        return
    ax.scatter(
        [record["x"] for record in residential_records],
        [record["y"] for record in residential_records],
        s=140,
        marker=HOUSE_MARKER,
        facecolor=AREA_COLORS["residential"],
        edgecolor="#3f2a12",
        linewidth=0.9,
        zorder=5,
    )


def _draw_static_agents(ax, graph, human_distribution, vehicle_distribution):
    for human_id, record in human_distribution["human_records"].items():
        hx, hy = _node_xy(graph, record["home_node"])
        ax.scatter([hx], [hy], s=90, marker="o", color=HUMAN_COLOR, edgecolor="white", linewidth=1.0, zorder=7)
        ax.text(hx, hy, human_id.replace("human_", "h"), fontsize=8, color="#0f6f9b", zorder=7)

    for vehicle_id, record in vehicle_distribution["vehicle_records"].items():
        vx, vy = _node_xy(graph, record["initial_node"])
        ax.scatter([vx], [vy], s=185, marker=CAR_MARKER, color=VEHICLE_COLOR, edgecolor="white", linewidth=1.0, zorder=7)
        ax.text(vx, vy, vehicle_id.replace("vehicle_", "v"), fontsize=8, color=VEHICLE_COLOR, zorder=8)


def _draw_dynamic_agents(ax, scenario, agent_positions, human_destinations, vehicle_destinations):
    graph = scenario.network
    human_records = scenario.human_distribution["human_records"]
    for human_id, record in human_records.items():
        pos_xy = _position_xy(graph, agent_positions.get(human_id, record["home_node"]))
        if pos_xy is None:
            continue
        hx, hy = pos_xy
        _ = human_destinations
        ax.scatter([hx], [hy], s=90, marker="o", color=HUMAN_COLOR, edgecolor="white", linewidth=1.0, zorder=7)
        ax.text(hx, hy, human_id.replace("human_", "h"), fontsize=8, color="#0f6f9b", zorder=7)

    for vehicle_id in scenario.vehicle_distribution["vehicle_records"]:
        pos_xy = _position_xy(graph, agent_positions.get(vehicle_id))
        if pos_xy is None:
            continue
        vx, vy = pos_xy
        target_node = vehicle_destinations.get(vehicle_id)
        if target_node is not None and target_node in graph.nodes:
            tx, ty = _node_xy(graph, target_node)
            ax.plot(
                [vx, tx],
                [vy, ty],
                color=VEHICLE_GUIDE_COLOR,
                linewidth=1.25,
                alpha=0.72,
                linestyle="--",
                zorder=4,
            )
        ax.scatter([vx], [vy], s=185, marker=CAR_MARKER, color=VEHICLE_COLOR, edgecolor="white", linewidth=1.0, zorder=7)
        ax.text(vx, vy, vehicle_id.replace("vehicle_", "v"), fontsize=8, color=VEHICLE_COLOR, zorder=8)


def _build_legend(ax, scenario, include_vehicle_target=False):
    human_count = len(scenario.human_distribution["human_records"])
    vehicle_count = len(scenario.vehicle_distribution["vehicle_records"])
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=HUMAN_COLOR, markersize=8, label=f"human (n={human_count})"),
        Line2D([0], [0], marker=CAR_MARKER, color="w", markerfacecolor=VEHICLE_COLOR, markeredgecolor="white", markersize=10, label=f"vehicle (n={vehicle_count})"),
        Line2D([0], [0], marker=HOUSE_MARKER, color="w", markerfacecolor=AREA_COLORS["residential"], markeredgecolor="#3f2a12", markersize=10, label="residential place"),
    ]
    if include_vehicle_target:
        legend_handles.append(
            Line2D([0], [0], color=VEHICLE_GUIDE_COLOR, linestyle="--", linewidth=1.25, label="vehicle target")
        )
    for level, color in REGION_COLORS.items():
        legend_handles.append(
            Line2D([0], [0], marker="s", color="w", markerfacecolor=color, markeredgecolor="#ffffff", markersize=9, label=f"{level} region")
        )
    for poi_type, marker in POI_MARKERS.items():
        legend_handles.append(
            Line2D([0], [0], marker=marker, color="w", markerfacecolor=AREA_COLORS[poi_type], markeredgecolor="#111827", markersize=8, label=poi_type)
        )
    ax.legend(handles=legend_handles, loc="best", fontsize=8, frameon=True)


def _finalize_axes(fig, ax, title, bottom_margin=0.05):
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)
    fig.tight_layout(rect=(0, bottom_margin, 1, 1))


def save_scenario_figure(
    scenario,
    output_path,
    title="Generated Transport Scenario",
    show_node_labels=False,
    show_edge_length_table=False,
    footer_lines=None,
    utility_bounds=None,
):
    graph = scenario.network
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    _draw_voronoi_cells(ax, graph)
    _draw_network(ax, graph, show_node_labels=show_node_labels)
    _draw_utility_bound_nodes(ax, graph, utility_bounds=utility_bounds)
    _draw_residential_places(ax, scenario.poi_distribution["residential_records"])
    _draw_pois(ax, scenario.poi_distribution["poi_records"])
    _draw_static_agents(ax, graph, scenario.human_distribution, scenario.vehicle_distribution)
    _build_legend(ax, scenario)
    if show_edge_length_table or footer_lines:
        _add_edge_length_table(fig, graph, footer_lines=footer_lines)
    _finalize_axes(
        fig,
        ax,
        title,
        bottom_margin=0.16 if show_edge_length_table else 0.05,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_episode_frame(
    scenario,
    agent_positions: Dict[str, Any],
    human_destinations: Optional[Dict[str, Any]],
    output_path,
    vehicle_destinations: Optional[Dict[str, Any]] = None,
    title="Episode Frame",
):
    fig = make_episode_frame_figure(
        scenario=scenario,
        agent_positions=agent_positions,
        human_destinations=human_destinations,
        vehicle_destinations=vehicle_destinations,
        title=title,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def make_episode_frame_figure(
    scenario,
    agent_positions: Dict[str, Any],
    human_destinations: Optional[Dict[str, Any]],
    vehicle_destinations: Optional[Dict[str, Any]] = None,
    title="Episode Frame",
):
    graph = scenario.network
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    _draw_voronoi_cells(ax, graph)
    _draw_network(ax, graph)
    _draw_residential_places(ax, scenario.poi_distribution["residential_records"])
    _draw_pois(ax, scenario.poi_distribution["poi_records"])
    _draw_dynamic_agents(
        ax,
        scenario=scenario,
        agent_positions=agent_positions,
        human_destinations=human_destinations or {},
        vehicle_destinations=vehicle_destinations or {},
    )
    _build_legend(
        ax,
        scenario,
        include_vehicle_target=bool(vehicle_destinations),
    )
    _finalize_axes(fig, ax, title)

    return fig


def render_episode_frame_array(
    scenario,
    agent_positions: Dict[str, Any],
    human_destinations: Optional[Dict[str, Any]],
    vehicle_destinations: Optional[Dict[str, Any]] = None,
    title="Episode Frame",
):
    fig = make_episode_frame_figure(
        scenario=scenario,
        agent_positions=agent_positions,
        human_destinations=human_destinations,
        vehicle_destinations=vehicle_destinations,
        title=title,
    )
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = memoryview(fig.canvas.buffer_rgba()).cast("B")
    frame = bytearray(image)
    plt.close(fig)
    import numpy as np

    return np.frombuffer(frame, dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
