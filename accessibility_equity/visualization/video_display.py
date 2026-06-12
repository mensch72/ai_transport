"""
Scenario-aware helpers for environment video rendering.

The transport environment owns simulation state and frame timing. This module
owns display choices: scenario background, human state styling, accessibility
labels, and vehicle decision annotations.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np

from accessibility_equity.rewards.accessibility import compute_population_accessibility
from accessibility_equity.visualization.scenario_plot import (
    AREA_COLORS,
    CAR_MARKER,
    HOUSE_MARKER,
    POI_MARKERS,
    REGION_COLORS,
)


HUMAN_STATE_STYLES = {
    "waiting": {"facecolor": "#9ca3af", "edgecolor": "#4b5563"},
    "moving": {"facecolor": "#f59e0b", "edgecolor": "#92400e"},
    "riding": {"facecolor": "#8b5cf6", "edgecolor": "#5b21b6"},
    "boarding": {"facecolor": "#ec4899", "edgecolor": "#9d174d"},
}

VEHICLE_FACE_COLOR = "#2563eb"
VEHICLE_EDGE_COLOR = "#1e3a8a"
VEHICLE_DECISION_COLOR = "#0f766e"

def get_video_scenario(env: Any) -> Any:
    """Return a scenario object attached to an env, if one is available."""
    return getattr(env, "video_scenario", None) or getattr(env, "scenario", None)


def configure_video_axes(env: Any) -> None:
    """Prepare axes for a clean video frame without grid or borders."""
    if env.ax is None:
        return
    env.ax.set_aspect("equal")
    env.ax.axis("off")
    env.ax.margins(0.16)
    env.ax.autoscale_view()

    x_min, x_max = env.ax.get_xlim()
    y_min, y_max = env.ax.get_ylim()
    span_x = max(x_max - x_min, 1.0)
    span_y = max(y_max - y_min, 1.0)
    pad = 0.08 * max(span_x, span_y)
    env.ax.set_xlim(x_min - pad, x_max + pad)
    env.ax.set_ylim(y_min - pad, y_max + pad)


def draw_scenario_background(env: Any, scenario: Any = None) -> bool:
    """
    Draw a scenario-aware static layer.

    Returns True when a scenario background was drawn. The env can use this to
    skip its older network-only fallback drawing.
    """
    scenario = scenario or get_video_scenario(env)
    if scenario is None or env.ax is None:
        return False

    graph = getattr(scenario, "network", None)
    if graph is None:
        return False

    _draw_voronoi_regions(env.ax, graph)
    _draw_light_network(env.ax, graph, env._network_pos)
    _draw_residential_places(env.ax, getattr(scenario, "poi_distribution", {}))
    _draw_pois(env.ax, getattr(scenario, "poi_distribution", {}))
    configure_video_axes(env)
    return True


def initialize_video_display_artists(env: Any) -> None:
    """Initialize optional text artists used by the video overlay."""
    env._human_label_artists = {}
    env._vehicle_label_artists = {}
    env._video_summary_artist = env.fig.text(
        0.015,
        0.975,
        "",
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#d1d5db",
            "linewidth": 0.6,
            "alpha": 0.82,
        },
        zorder=20,
    )
    env._video_legend = _build_video_legend(env)

    for human in env.human_agents:
        env._human_label_artists[human] = env.ax.text(
            0.0,
            0.0,
            "",
            fontsize=7,
            ha="left",
            va="bottom",
            color="#111827",
            zorder=15,
            bbox={
                "boxstyle": "round,pad=0.12",
                "facecolor": "white",
                "edgecolor": "#e5e7eb",
                "linewidth": 0.4,
                "alpha": 0.75,
            },
        )

    for vehicle in env.vehicle_agents:
        env._vehicle_label_artists[vehicle] = env.ax.text(
            0.0,
            0.0,
            "",
            fontsize=8,
            ha="center",
            va="top",
            color=VEHICLE_DECISION_COLOR,
            fontweight="bold",
            zorder=15,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "#99f6e4",
                "linewidth": 0.5,
                "alpha": 0.82,
            },
        )


def human_display_state(env: Any, human: str) -> str:
    """Classify a human's current display state from env state."""
    if env.human_aboard.get(human) is not None:
        return "riding"
    position = env.agent_positions.get(human)
    if isinstance(position, tuple):
        return "moving"
    return "waiting"


def human_style(env: Any, human: str) -> Dict[str, str]:
    """Return marker styling for a human."""
    state = human_display_state(env, human)
    return dict(HUMAN_STATE_STYLES.get(state, HUMAN_STATE_STYLES["waiting"]))


def compute_human_accessibility_values(env: Any) -> Dict[str, float]:
    """
    Compute per-human accessibility values for the current displayed state.

    This mirrors the reward wrapper's lightweight state extraction so that video
    overlays use the same accessibility definition as training diagnostics.
    """
    human_to_node: Dict[str, Any] = {}
    human_to_route: Dict[str, Sequence[Any]] = {}

    for human in env.human_agents:
        human_node = _position_to_node(env.agent_positions.get(human))
        if human_node is None:
            continue
        human_to_node[human] = human_node

        aboard = env.human_aboard.get(human)
        if aboard is not None:
            human_to_route[human] = _vehicle_route_nodes(env, aboard)
        else:
            human_to_route[human] = [human_node]

    if not human_to_node:
        return {}

    try:
        result = compute_population_accessibility(
            graph=env.network,
            human_to_node=human_to_node,
            human_to_route=human_to_route,
        )
    except Exception:
        return {}
    return {
        str(human): float(value)
        for human, value in result.get("individual_values", {}).items()
    }


def update_human_overlay(
    env: Any,
    human: str,
    xy: Tuple[float, float],
    accessibility_values: Mapping[str, float],
) -> None:
    """Update the marker style and label for one human."""
    style = human_style(env, human)
    artist = env._human_artists.get(human)
    if artist is not None:
        artist.set_facecolor(style["facecolor"])
        artist.set_edgecolor(style["edgecolor"])

    label_artist = getattr(env, "_human_label_artists", {}).get(human)
    if label_artist is None:
        return

    target = getattr(env, "human_destinations", {}).get(human)
    target_text = "target=NA" if target is None else f"target={target}"
    value = accessibility_values.get(human)
    value_text = "X=NA" if value is None else f"X={value:.2f}"
    label_artist.set_position((xy[0] + 0.18, xy[1] + 0.18))
    label_artist.set_text(f"{human.replace('human_', 'h')} {target_text}\n{value_text}")
    label_artist.set_visible(True)


def update_passenger_overlay(env: Any, vehicle: str, passengers: Sequence[str]) -> None:
    """Style passenger markers for humans riding inside a vehicle."""
    for index, human in enumerate(passengers):
        passenger_artist = env._passenger_artists.get((vehicle, index))
        if passenger_artist is None:
            continue
        style = HUMAN_STATE_STYLES["riding"]
        passenger_artist.set_facecolor(style["facecolor"])
        passenger_artist.set_edgecolor(style["edgecolor"])


def update_vehicle_overlay(env: Any, vehicle: str, xy: Tuple[float, float]) -> None:
    """Update vehicle marker styling and decision label."""
    artist = env._vehicle_artists.get(vehicle)
    if artist is not None:
        artist.set_facecolor(VEHICLE_FACE_COLOR)
        artist.set_edgecolor(VEHICLE_EDGE_COLOR)

    label_artist = getattr(env, "_vehicle_label_artists", {}).get(vehicle)
    if label_artist is None:
        return

    destination = env.vehicle_destinations.get(vehicle)
    destination_text = "target=NA" if destination is None else f"target={destination}"
    label_artist.set_position((xy[0], xy[1] - 0.35))
    label_artist.set_text(f"{vehicle.replace('vehicle_', 'v')} {destination_text}")
    label_artist.set_visible(True)


def update_summary_overlay(
    env: Any,
    accessibility_values: Mapping[str, float],
    title: Optional[str] = None,
) -> None:
    """Update a compact frame summary in the top-left corner."""
    summary_artist = getattr(env, "_video_summary_artist", None)
    if summary_artist is None:
        return

    values = list(accessibility_values.values())
    if values:
        accessibility_text = (
            f"X min={min(values):.2f} "
            f"mean={float(np.mean(values)):.2f} "
            f"max={max(values):.2f}"
        )
    else:
        accessibility_text = "X unavailable"

    moving_counts = {"waiting": 0, "moving": 0, "riding": 0}
    for human in env.human_agents:
        state = human_display_state(env, human)
        moving_counts[state] = moving_counts.get(state, 0) + 1

    summary_artist.set_text(
        f"{accessibility_text}\n"
        f"humans: waiting={moving_counts.get('waiting', 0)}, "
        f"moving={moving_counts.get('moving', 0)}, "
        f"riding={moving_counts.get('riding', 0)}"
    )


def initialize_accessibility_histogram(env: Any) -> None:
    """Prepare fixed histogram bins for per-human accessibility values."""
    hist_ax = getattr(env, "hist_ax", None)
    if hist_ax is None:
        return

    x_min, x_max = _compute_accessibility_histogram_bounds(env)
    env._accessibility_hist_x_min = x_min
    env._accessibility_hist_x_max = x_max
    env._accessibility_hist_bins = np.linspace(x_min, x_max, 11)
    env._accessibility_hist_y_max = max(1, len(getattr(env, "human_agents", [])))

    hist_ax.clear()
    _format_accessibility_histogram_axis(env)


def update_accessibility_histogram(
    env: Any,
    accessibility_values: Mapping[str, float],
) -> None:
    """Draw the current population distribution of X_h on the video side panel."""
    hist_ax = getattr(env, "hist_ax", None)
    if hist_ax is None:
        return

    values = np.asarray(list(accessibility_values.values()), dtype=float)
    values = values[np.isfinite(values)]

    hist_ax.clear()
    bins = getattr(env, "_accessibility_hist_bins", None)
    if bins is None:
        initialize_accessibility_histogram(env)
        bins = getattr(env, "_accessibility_hist_bins", None)

    if bins is not None and values.size:
        hist_ax.hist(
            values,
            bins=bins,
            color="#2563eb",
            edgecolor="#ffffff",
            linewidth=0.8,
            alpha=0.82,
        )
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        hist_ax.axvline(
            mean_value,
            color="#dc2626",
            linewidth=1.6,
            label=f"mean={mean_value:.2f}",
        )
        hist_ax.axvline(
            median_value,
            color="#7c3aed",
            linewidth=1.3,
            linestyle="--",
            label=f"median={median_value:.2f}",
        )
        hist_ax.legend(loc="upper right", fontsize=7, frameon=False)
    else:
        hist_ax.text(
            0.5,
            0.5,
            "X unavailable",
            ha="center",
            va="center",
            transform=hist_ax.transAxes,
            fontsize=10,
            color="#6b7280",
        )

    _format_accessibility_histogram_axis(env)


def _compute_accessibility_histogram_bounds(env: Any) -> Tuple[float, float]:
    """Compute fixed X_h bounds from node-based accessibility values."""
    reward_config = getattr(env, "reward_config", None)
    beta = None if reward_config is None else getattr(reward_config, "beta", None)
    gamma = None if reward_config is None else getattr(reward_config, "alpha", None)
    node_values = []
    for node in getattr(env, "network", nx.DiGraph()).nodes():
        kwargs = {
            "graph": env.network,
            "human_to_node": {"human": node},
            "human_to_route": {"human": [node]},
        }
        if beta is not None:
            kwargs["beta"] = beta
        if gamma is not None:
            kwargs["gamma"] = gamma
        result = compute_population_accessibility(**kwargs)
        value = result.get("individual_values", {}).get("human")
        if value is not None and math.isfinite(float(value)):
            node_values.append(float(value))

    if not node_values:
        return 0.0, 1.0

    x_min = min(node_values)
    x_max = max(node_values)
    if math.isclose(x_min, x_max):
        padding = max(abs(x_min) * 0.05, 0.5)
    else:
        padding = 0.06 * (x_max - x_min)
    return max(0.0, x_min - padding), x_max + padding


def _format_accessibility_histogram_axis(env: Any) -> None:
    hist_ax = getattr(env, "hist_ax", None)
    if hist_ax is None:
        return

    hist_ax.set_title("Population $X_h$", fontsize=11, fontweight="bold")
    hist_ax.set_xlabel("$X_h$", fontsize=9)
    hist_ax.set_ylabel("humans", fontsize=9)
    hist_ax.set_xlim(
        getattr(env, "_accessibility_hist_x_min", 0.0),
        getattr(env, "_accessibility_hist_x_max", 1.0),
    )
    hist_ax.set_ylim(0, getattr(env, "_accessibility_hist_y_max", 1))
    hist_ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
    hist_ax.tick_params(axis="both", labelsize=8)
    for spine in ("top", "right"):
        hist_ax.spines[spine].set_visible(False)


def reset_dynamic_labels(env: Any) -> None:
    """Clear dynamic labels before updating a frame."""
    for label in getattr(env, "_human_label_artists", {}).values():
        label.set_text("")
        label.set_visible(False)
    for label in getattr(env, "_vehicle_label_artists", {}).values():
        label.set_text("")
        label.set_visible(False)


def render_uniform_frames(env: Any, goal_info=None, value_dict=None, title=None) -> None:
    """
    Migrated copy of TransportParallelEnv._render_uniform_frames.

    This keeps the env's original click-based interpolation behavior while
    making video_display.py the owner of video rendering logic.
    """
    t_currentevent = env.real_time

    if not hasattr(env, "_last_event_time"):
        env._last_event_time = 0.0
        env._positions_at_last_event = {}
        env._speeds_at_last_event = {}

    t_lastevent = env._last_event_time
    positions_at_lastevent = (
        env._positions_at_last_event
        if env._positions_at_last_event
        else env.agent_positions.copy()
    )
    speeds_at_lastevent = env._speeds_at_last_event if env._speeds_at_last_event else {}

    click_interval = env._time_per_frame
    first_click_index = int(t_lastevent / click_interval) + 1
    last_click_index = int(t_currentevent / click_interval)

    if first_click_index <= last_click_index:
        saved_positions = env.agent_positions.copy()

        for click_index in range(first_click_index, last_click_index + 1):
            t_click = click_index * click_interval

            for agent in env.agents:
                pos_at_lastevent = positions_at_lastevent.get(agent)
                speed = speeds_at_lastevent.get(agent, 0.0)

                if isinstance(pos_at_lastevent, tuple):
                    edge, coord_at_lastevent = pos_at_lastevent
                    elapsed = t_click - t_lastevent
                    coord_at_click = coord_at_lastevent + speed * elapsed
                    edge_data = env.network[edge[0]][edge[1]]
                    edge_length = edge_data["length"]

                    if coord_at_click >= edge_length - 0.001:
                        elapsed_to_current = t_currentevent - t_lastevent
                        coord_at_currentevent = coord_at_lastevent + speed * elapsed_to_current

                        if coord_at_currentevent >= edge_length - 0.001:
                            env.agent_positions[agent] = edge[1]
                        else:
                            env.agent_positions[agent] = (edge, min(coord_at_click, edge_length))
                    else:
                        env.agent_positions[agent] = (edge, coord_at_click)
                else:
                    env.agent_positions[agent] = pos_at_lastevent

            for human in env.human_agents:
                aboard = env.human_aboard.get(human)
                if aboard is not None:
                    env.agent_positions[human] = env.agent_positions[aboard]

            humans_aboard = sum(
                1 for h in env.human_agents if env.human_aboard.get(h) is not None
            )
            frame_title = f"Time: {t_click:.2f}s | Humans aboard: {humans_aboard}"

            render_single_frame(
                env,
                goal_info=goal_info,
                value_dict=value_dict,
                title=frame_title,
                capture_frame=True,
            )

        env.agent_positions = saved_positions

    env._last_event_time = t_currentevent
    env._positions_at_last_event = env.agent_positions.copy()
    env._speeds_at_last_event = {}
    for agent in env.agents:
        pos = env.agent_positions.get(agent)
        if isinstance(pos, tuple):
            edge, _ = pos
            edge_data = env.network[edge[0]][edge[1]]
            env._speeds_at_last_event[agent] = env._get_agent_speed(agent, edge_data)
        else:
            env._speeds_at_last_event[agent] = 0.0


def render_single_frame(
    env: Any,
    goal_info=None,
    value_dict=None,
    title=None,
    capture_frame: bool = False,
):
    """Migrated copy of TransportParallelEnv._render_single_frame."""
    if not env._artists_initialized:
        initialize_artists(env)

    from matplotlib.transforms import Affine2D

    pos = env._network_pos
    accessibility_values = compute_human_accessibility_values(env)
    reset_dynamic_labels(env)

    for vehicle in env.vehicle_agents:
        vehicle_pos = env.agent_positions.get(vehicle)
        artist = env._vehicle_artists[vehicle]

        if vehicle_pos is None:
            artist.set_visible(False)
            continue

        if isinstance(vehicle_pos, tuple):
            edge, coord = vehicle_pos
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            edge_length = env.network[edge[0]][edge[1]]["length"]

            if edge_length > 0:
                t = coord / edge_length
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                dx = x2 - x1
                dy = y2 - y1
                rotation_angle = np.degrees(np.arctan2(dy, dx))

                if env.network.has_edge(edge[1], edge[0]):
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        px = -dy / length * 0.15
                        py = dx / length * 0.15
                        x += px
                        y += py
            else:
                x, y = x1, y1
                rotation_angle = 0
        else:
            x, y = pos[vehicle_pos]
            rotation_angle = 0

        capacity = env.agent_attributes.get(vehicle, {}).get("capacity", 4)
        vehicle_width = max(0.8, 0.4 + capacity * 0.25)
        vehicle_height = 0.3

        artist.set_xy((x - vehicle_width / 2, y - vehicle_height / 2))
        artist.set_width(vehicle_width)
        artist.set_height(vehicle_height)

        transform = Affine2D().rotate_deg_around(x, y, rotation_angle) + env.ax.transData
        artist.set_transform(transform)
        artist.set_visible(True)
        update_vehicle_overlay(env, vehicle, (x, y))

        passengers = [h for h in env.human_agents if env.human_aboard.get(h) == vehicle]
        num_passengers = len(passengers)

        for i in range(capacity):
            passenger_artist = env._passenger_artists.get((vehicle, i))
            if passenger_artist:
                if i < num_passengers:
                    if num_passengers == 1:
                        offset_x = 0
                    else:
                        spacing = vehicle_width * 0.8
                        offset_x = -spacing / 2 + (i * spacing / (num_passengers - 1))

                    passenger_artist.set_center((x + offset_x, y))
                    passenger_artist.set_visible(True)
                else:
                    passenger_artist.set_visible(False)
        update_passenger_overlay(env, vehicle, passengers)

        dest = env.vehicle_destinations.get(vehicle)
        dest_artist = env._destination_artists[vehicle]

        if dest is not None and dest in pos:
            dest_x, dest_y = pos[dest]
            dx = dest_x - x
            dy = dest_y - y
            distance = np.sqrt(dx**2 + dy**2)

            if distance > 0.5:
                base_angle = np.arctan2(dy, dx)
                ctrl_distance = distance * 0.3
                start_offset = np.pi / 3
                end_offset = np.pi / 3

                ctrl1_x = x + ctrl_distance * np.cos(base_angle + start_offset)
                ctrl1_y = y + ctrl_distance * np.sin(base_angle + start_offset)
                ctrl2_x = dest_x + ctrl_distance * np.cos(base_angle + np.pi - end_offset)
                ctrl2_y = dest_y + ctrl_distance * np.sin(base_angle + np.pi - end_offset)

                t_values = np.linspace(0, 1, 40)
                curve_x, curve_y = [], []
                for t in t_values:
                    s = 1 - t
                    bx = (
                        s**3 * x
                        + 3 * s**2 * t * ctrl1_x
                        + 3 * s * t**2 * ctrl2_x
                        + t**3 * dest_x
                    )
                    by = (
                        s**3 * y
                        + 3 * s**2 * t * ctrl1_y
                        + 3 * s * t**2 * ctrl2_y
                        + t**3 * dest_y
                    )
                    curve_x.append(bx)
                    curve_y.append(by)

                dest_artist.set_data(curve_x, curve_y)
                dest_artist.set_visible(True)
            else:
                dest_artist.set_visible(False)
        else:
            dest_artist.set_visible(False)

    for human in env.human_agents:
        aboard = env.human_aboard.get(human)
        artist = env._human_artists[human]

        if aboard is not None:
            artist.set_visible(False)
            continue

        human_pos = env.agent_positions.get(human)
        if human_pos is None:
            artist.set_visible(False)
            continue

        if isinstance(human_pos, tuple):
            edge, coord = human_pos
            x1, y1 = pos[edge[0]]
            x2, y2 = pos[edge[1]]
            edge_length = env.network[edge[0]][edge[1]]["length"]

            if edge_length > 0:
                t = coord / edge_length
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                u, v = edge[0], edge[1]
                if env.network.has_edge(u, v) and env.network.has_edge(v, u):
                    dx = x2 - x1
                    dy = y2 - y1
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        px = -dy / length * 0.15
                        py = dx / length * 0.15
                        x += px
                        y += py
            else:
                x, y = x1, y1
        else:
            x, y = pos[human_pos]

        artist.set_center((x, y))
        artist.set_visible(True)
        update_human_overlay(env, human, (x, y), accessibility_values)

    if title is not None:
        env.ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        env.ax.set_title(
            f"Transport Network - Time: {env.real_time:.2f}, Step: {env.step_type}",
            fontsize=14,
            fontweight="bold",
        )
    update_summary_overlay(env, accessibility_values, title=title)
    update_accessibility_histogram(env, accessibility_values)

    if env.fig is not None:
        env.fig.canvas.draw()
        env.fig.canvas.flush_events()

    if capture_frame and hasattr(env, "_recording") and env._recording:
        env.fig.canvas.draw()
        buf = env.fig.canvas.buffer_rgba()
        frame = np.asarray(buf)
        frame = frame[:, :, :3].copy()
        env.frames.append(frame)

    return env.fig


def start_video_recording(env: Any) -> None:
    """Migrated copy of TransportParallelEnv.start_video_recording."""
    env._recording = True
    env.frames = []
    env._use_graphical = True
    initialize_artists(env)


def initialize_artists(env: Any) -> None:
    """Migrated copy of TransportParallelEnv._initialize_artists."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for graphical rendering")
        return

    if env.fig is None or env.ax is None or not hasattr(env, "hist_ax"):
        if env.fig is not None:
            plt.close(env.fig)
        env.fig, (env.ax, env.hist_ax) = plt.subplots(
            1,
            2,
            figsize=(14, 8),
            dpi=150,
            gridspec_kw={"width_ratios": [2.25, 1.0]},
        )

    env.fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.08, wspace=0.18)

    env.ax.clear()
    env.hist_ax.clear()
    env.ax.set_aspect("equal")
    env.ax.axis("off")

    env._network_pos = {}
    for node in env.network.nodes():
        if "x" in env.network.nodes[node] and "y" in env.network.nodes[node]:
            env._network_pos[node] = (
                env.network.nodes[node]["x"],
                env.network.nodes[node]["y"],
            )
        else:
            env._network_pos = nx.spring_layout(env.network, seed=42)
            break

    if env._network_pos:
        x_vals = [p[0] for p in env._network_pos.values()]
        y_vals = [p[1] for p in env._network_pos.values()]
        x_margin = (max(x_vals) - min(x_vals)) * 0.03 + 0.5
        y_margin = (max(y_vals) - min(y_vals)) * 0.03 + 0.5
        env.ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
        env.ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    if not draw_scenario_background(env):
        draw_static_network(env)

    initialize_agent_artists(env)
    initialize_video_display_artists(env)
    initialize_accessibility_histogram(env)

    env._artists_initialized = True


def draw_static_network(env: Any) -> None:
    """Migrated copy of TransportParallelEnv._draw_static_network."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle

    pos = env._network_pos

    env._edge_artists = []
    for u, v in env.network.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        is_bidirectional = env.network.has_edge(v, u)

        if is_bidirectional and u < v:
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                px = -dy / length * 0.15
                py = dx / length * 0.15

                line1 = Line2D(
                    [x1 + px, x2 + px],
                    [y1 + py, y2 + py],
                    color="gray",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=1,
                )
                env.ax.add_line(line1)
                env._edge_artists.append(line1)

                line2 = Line2D(
                    [x2 - px, x1 - px],
                    [y2 - py, y1 - py],
                    color="gray",
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=1,
                )
                env.ax.add_line(line2)
                env._edge_artists.append(line2)
        elif not is_bidirectional:
            line = Line2D(
                [x1, x2],
                [y1, y2],
                color="gray",
                linewidth=1.5,
                alpha=0.6,
                zorder=1,
            )
            env.ax.add_line(line)
            env._edge_artists.append(line)

    env._node_artists = {}
    for node in env.network.nodes():
        x, y = pos[node]
        circle = Circle(
            (x, y),
            radius=1.0,
            color="lightblue",
            ec="black",
            linewidth=2,
            zorder=2,
        )
        env.ax.add_patch(circle)
        env._node_artists[node] = circle

        env.ax.text(
            x,
            y,
            str(node),
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )


def initialize_agent_artists(env: Any) -> None:
    """Migrated copy of TransportParallelEnv._initialize_agent_artists."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Circle, Rectangle

    env._vehicle_artists = {}
    for vehicle in env.vehicle_agents:
        capacity = env.agent_attributes.get(vehicle, {}).get("capacity", 4)
        vehicle_width = max(0.8, 0.4 + capacity * 0.25)
        vehicle_height = 0.3

        rect = Rectangle(
            (0, 0),
            vehicle_width,
            vehicle_height,
            color="cornflowerblue",
            ec="darkblue",
            linewidth=1.5,
            zorder=4,
            alpha=0.7,
        )
        rect.set_visible(False)
        env.ax.add_patch(rect)
        env._vehicle_artists[vehicle] = rect

    env._human_artists = {}
    for human in env.human_agents:
        circle = Circle(
            (0, 0),
            radius=0.15,
            color="red",
            ec="darkred",
            linewidth=1.5,
            zorder=5,
        )
        circle.set_visible(False)
        env.ax.add_patch(circle)
        env._human_artists[human] = circle

    env._passenger_artists = {}
    for vehicle in env.vehicle_agents:
        capacity = env.agent_attributes.get(vehicle, {}).get("capacity", 4)
        for i in range(capacity):
            circle = Circle(
                (0, 0),
                radius=0.10,
                color="red",
                ec="darkred",
                linewidth=1,
                zorder=6,
            )
            circle.set_visible(False)
            env.ax.add_patch(circle)
            env._passenger_artists[(vehicle, i)] = circle

    env._destination_artists = {}
    for vehicle in env.vehicle_agents:
        line = Line2D(
            [],
            [],
            color="cornflowerblue",
            linestyle=":",
            linewidth=2,
            alpha=0.6,
            zorder=2,
        )
        line.set_visible(False)
        env.ax.add_line(line)
        env._destination_artists[vehicle] = line


def save_video(env: Any, filename: str = "transport_video.mp4", fps: int = 20) -> None:
    """Migrated copy of TransportParallelEnv.save_video."""
    if not env.frames:
        print("No frames recorded. Call start_video_recording() first.")
        return

    print(f"Saving {len(env.frames)} frames...")

    try:
        is_gif = filename.lower().endswith(".gif")

        if not is_gif:
            try:
                import imageio

                writer = imageio.get_writer(
                    filename,
                    fps=fps,
                    codec="libx264",
                    pixelformat="yuv420p",
                    quality=8,
                )

                for frame in env.frames:
                    writer.append_data(frame)

                writer.close()
                print(f"Video saved to {filename} ({len(env.frames)} frames)")
                return

            except Exception as e:
                print(f"Could not save MP4 with imageio ({e}), trying GIF...")
                filename = filename.replace(".mp4", ".gif")

        from PIL import Image

        pil_frames = [Image.fromarray(frame) for frame in env.frames]

        duration_ms = int(1000 / fps)
        pil_frames[0].save(
            filename,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"Video saved as GIF to {filename} ({len(env.frames)} frames)")

    except Exception as e:
        print(f"Error saving video: {e}")
    finally:
        env._recording = False
        env.frames = []


def _draw_voronoi_regions(ax: Any, graph: nx.DiGraph) -> None:
    from matplotlib.patches import Polygon

    for _, data in graph.nodes(data=True):
        geometry = data.get("voronoi_geometry")
        if geometry is None or getattr(geometry, "is_empty", False):
            continue
        polygons = list(getattr(geometry, "geoms", [geometry]))
        for polygon in polygons:
            if not hasattr(polygon, "exterior"):
                continue
            patch = Polygon(
                list(polygon.exterior.coords),
                closed=True,
                facecolor=REGION_COLORS.get(data.get("central_level"), "#f8fafc"),
                edgecolor="#ffffff",
                linewidth=0.55,
                alpha=0.42,
                zorder=0,
            )
            ax.add_patch(patch)


def _draw_light_network(ax: Any, graph: nx.DiGraph, pos: Mapping[Any, Tuple[float, float]]) -> None:
    from matplotlib.lines import Line2D

    xs = [float(point[0]) for point in pos.values()]
    ys = [float(point[1]) for point in pos.values()]
    span_x = max(xs) - min(xs) if xs else 1.0
    span_y = max(ys) - min(ys) if ys else 1.0
    twin_offset = max(0.0025 * max(span_x, span_y), 0.025)

    drawn_two_way = set()
    for u, v in graph.edges():
        if u == v:
            continue
        if u not in pos or v not in pos:
            continue
        is_two_way = graph.has_edge(v, u)
        if is_two_way:
            edge_key = frozenset((u, v))
            if edge_key in drawn_two_way:
                continue
            drawn_two_way.add(edge_key)

        x1, y1 = pos[u]
        x2, y2 = pos[v]
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
                line = Line2D(
                    [x1 + offset_x, x2 + offset_x],
                    [y1 + offset_y, y2 + offset_y],
                    color="#4b5563",
                    linewidth=0.62,
                    alpha=0.62,
                    zorder=2,
                    solid_capstyle="round",
                )
                ax.add_line(line)
        else:
            line = Line2D(
                [x1, x2],
                [y1, y2],
                color="#6b7280",
                linewidth=0.65,
                alpha=0.35,
                zorder=2,
                solid_capstyle="round",
            )
            ax.add_line(line)

    if xs and ys:
        ax.scatter(xs, ys, s=18, color="#374151", alpha=0.78, zorder=6)
        for node, (x, y) in pos.items():
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


def _build_video_legend(env: Any) -> Any:
    scenario = get_video_scenario(env)
    poi_distribution = getattr(scenario, "poi_distribution", {}) if scenario is not None else {}

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=HUMAN_STATE_STYLES["waiting"]["facecolor"],
            markeredgecolor=HUMAN_STATE_STYLES["waiting"]["edgecolor"],
            markersize=6,
            label="waiting human",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=HUMAN_STATE_STYLES["moving"]["facecolor"],
            markeredgecolor=HUMAN_STATE_STYLES["moving"]["edgecolor"],
            markersize=6,
            label="moving human",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=HUMAN_STATE_STYLES["riding"]["facecolor"],
            markeredgecolor=HUMAN_STATE_STYLES["riding"]["edgecolor"],
            markersize=6,
            label="riding human",
        ),
        Line2D(
            [0],
            [0],
            marker=CAR_MARKER,
            color="w",
            markerfacecolor=VEHICLE_FACE_COLOR,
            markeredgecolor=VEHICLE_EDGE_COLOR,
            markersize=9,
            label=f"vehicle (n={len(env.vehicle_agents)})",
        ),
        Line2D(
            [0],
            [0],
            marker=HOUSE_MARKER,
            color="w",
            markerfacecolor=AREA_COLORS["residential"],
            markeredgecolor="#3f2a12",
            markersize=8,
            label="residential place",
        ),
    ]

    for level, color in REGION_COLORS.items():
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=color,
                markeredgecolor="#ffffff",
                markersize=7,
                label=f"{level} region",
            )
        )

    records = poi_distribution.get("poi_records", []) if poi_distribution else []
    available_poi_types = {record.get("poi_type") for record in records}
    for poi_type, marker in POI_MARKERS.items():
        if records and poi_type not in available_poi_types:
            continue
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor=AREA_COLORS[poi_type],
                markeredgecolor="#111827",
                markersize=7,
                label=poi_type,
            )
        )

    legend = env.fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.988, 0.985),
        fontsize=7,
        frameon=True,
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.2,
        columnspacing=0.8,
        labelspacing=0.35,
    )
    legend.get_frame().set_edgecolor("#d1d5db")
    legend.get_frame().set_linewidth(0.6)
    legend.get_frame().set_alpha(0.88)
    return legend


def _draw_residential_places(ax: Any, poi_distribution: Mapping[str, Any]) -> None:
    records = poi_distribution.get("residential_records", []) if poi_distribution else []
    if not records:
        return
    ax.scatter(
        [record["x"] for record in records],
        [record["y"] for record in records],
        s=80,
        marker=HOUSE_MARKER,
        facecolor=AREA_COLORS["residential"],
        edgecolor="#3f2a12",
        linewidth=0.65,
        alpha=0.82,
        zorder=3,
    )


def _draw_pois(ax: Any, poi_distribution: Mapping[str, Any]) -> None:
    records = poi_distribution.get("poi_records", []) if poi_distribution else []
    for poi_type, marker in POI_MARKERS.items():
        selected = [record for record in records if record.get("poi_type") == poi_type]
        if not selected:
            continue
        ax.scatter(
            [record["x"] for record in selected],
            [record["y"] for record in selected],
            s=35,
            marker=marker,
            color=AREA_COLORS.get(poi_type, "#e5e7eb"),
            edgecolor="#334155",
            linewidth=0.55,
            alpha=0.84,
            zorder=3,
        )


def _position_to_node(position: Any) -> Optional[Any]:
    if position is None:
        return None
    if isinstance(position, tuple):
        edge, _coord = position
        if edge and len(edge) >= 2:
            return edge[0]
        return None
    return position


def _vehicle_route_nodes(env: Any, vehicle: str) -> Sequence[Any]:
    vehicle_node = _position_to_node(env.agent_positions.get(vehicle))
    if vehicle_node is None:
        return []
    destination = env.vehicle_destinations.get(vehicle)
    if destination is None or destination == vehicle_node:
        return [vehicle_node]
    try:
        return list(
            nx.shortest_path(
                env.network,
                source=vehicle_node,
                target=destination,
                weight="length",
            )
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [vehicle_node]
