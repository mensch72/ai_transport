"""
Hexagonal Grid Transport Simulation
====================================

This script demonstrates a transport simulation on a hexagonal grid with:
- A hexagonal grid of roughly 200 nodes (radius 8 = 217 nodes)
- 100 passengers randomly distributed on boundary nodes
- Each passenger's goal is to reach the opposite boundary
- Vehicles use ShortestPathVehiclePolicy
- Humans use HeuristicRoutingHumanPolicy
- Produces a movie showing both network dynamics and a timeseries of goals reached

Run this script to generate:
- hexagonal_grid_simulation.mp4: Video with network dynamics + goals-reached timeseries
- Console output with simulation progress and summary statistics
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ai_transport import parallel_env
from ai_transport.policies import HeuristicRoutingHumanPolicy, ShortestPathVehiclePolicy


def create_hexagonal_grid(radius=8, edge_speed=3.0, edge_capacity=10, spacing=3.0):
    """
    Create a hexagonal grid graph using axial coordinates.

    A hexagonal grid of radius R contains all cells (q, r) satisfying
    max(|q|, |r|, |q+r|) <= R, giving 3*R^2 + 3*R + 1 nodes.

    For R=8, this yields 217 nodes.

    Args:
        radius: Grid radius (number of rings around center). R=8 gives 217 nodes.
        edge_speed: Speed on each edge.
        edge_capacity: Capacity of each edge.
        spacing: Distance between adjacent hex centers.

    Returns:
        G: NetworkX DiGraph with node attributes (x, y, name) and
           edge attributes (length, speed, capacity).
        boundary_nodes: Set of node IDs on the outermost ring.
    """
    G = nx.DiGraph()

    # Generate all hex cells using axial coordinates (q, r)
    # A hex cell is in the grid if max(|q|, |r|, |q+r|) <= radius
    axial_to_id = {}
    node_id = 0
    boundary_nodes = set()

    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            s = -q - r  # cube coordinate constraint: q + r + s = 0
            if max(abs(q), abs(r), abs(s)) <= radius:
                # Convert axial to cartesian (pointy-top orientation)
                x = spacing * (np.sqrt(3) * q + np.sqrt(3) / 2 * r)
                y = spacing * (3.0 / 2 * r)

                G.add_node(node_id, name=f"hex_{q}_{r}", x=float(x), y=float(y))
                axial_to_id[(q, r)] = node_id

                # Check if boundary node (on outermost ring)
                if max(abs(q), abs(r), abs(s)) == radius:
                    boundary_nodes.add(node_id)

                node_id += 1

    # Add edges: connect each node to its 6 hex neighbors (bidirectional)
    # Axial neighbor directions
    hex_directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

    for (q, r), nid in axial_to_id.items():
        for dq, dr in hex_directions:
            neighbor = (q + dq, r + dr)
            if neighbor in axial_to_id:
                neighbor_id = axial_to_id[neighbor]
                # Compute edge length from Euclidean distance
                x1, y1 = G.nodes[nid]['x'], G.nodes[nid]['y']
                x2, y2 = G.nodes[neighbor_id]['x'], G.nodes[neighbor_id]['y']
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                G.add_edge(nid, neighbor_id,
                           length=float(length),
                           speed=float(edge_speed),
                           capacity=int(edge_capacity))

    return G, boundary_nodes


def get_opposite_boundary_node(node_id, boundary_nodes, G):
    """
    Find the boundary node approximately diametrically opposite to the given node.

    The opposite node is the boundary node farthest from the given node in
    Euclidean distance (which for a centered hex grid is the one across the center).

    Args:
        node_id: Source boundary node.
        boundary_nodes: Set of all boundary node IDs.
        G: The network graph.

    Returns:
        The node ID of the opposite boundary node.
    """
    x0, y0 = G.nodes[node_id]['x'], G.nodes[node_id]['y']
    best_node = None
    best_dist = -1.0
    for bn in boundary_nodes:
        if bn == node_id:
            continue
        xb, yb = G.nodes[bn]['x'], G.nodes[bn]['y']
        dist = np.sqrt((xb - x0) ** 2 + (yb - y0) ** 2)
        if dist > best_dist:
            best_dist = dist
            best_node = bn
    return best_node


def main():
    print("=" * 70)
    print("AI Transport - Hexagonal Grid Simulation")
    print("=" * 70)
    print()

    # ---- Configuration ----
    HEX_RADIUS = 8          # Hex grid radius -> 3*64 + 24 + 1 = 217 nodes
    NUM_PASSENGERS = 100
    NUM_VEHICLES = 20
    NUM_STEPS = 4000         # Total environment steps (routing+unboarding+boarding+departing)
    RENDER_INTERVAL = 40     # Render a frame every N steps (keep video manageable)
    SEED = 42
    VIDEO_FPS = 10

    np.random.seed(SEED)

    # ---- 1. Create hexagonal grid ----
    print("1. Creating hexagonal grid network...")
    network, boundary_nodes = create_hexagonal_grid(
        radius=HEX_RADIUS, edge_speed=3.0, edge_capacity=10, spacing=3.0
    )
    boundary_list = sorted(boundary_nodes)
    print(f"   Nodes: {network.number_of_nodes()}, Edges: {network.number_of_edges()}")
    print(f"   Boundary nodes: {len(boundary_list)}")

    # ---- 2. Assign passengers to boundary nodes with opposite-boundary goals ----
    print("\n2. Distributing passengers on boundary...")
    rng = np.random.RandomState(SEED)
    start_nodes = rng.choice(boundary_list, size=NUM_PASSENGERS, replace=True)
    target_nodes_list = []
    for sn in start_nodes:
        target_nodes_list.append(get_opposite_boundary_node(sn, boundary_nodes, network))

    print(f"   {NUM_PASSENGERS} passengers placed on boundary nodes")

    # ---- 3. Create environment ----
    print("\n3. Creating environment...")
    vehicle_capacities = rng.randint(6, 9, NUM_VEHICLES).tolist()
    vehicle_speeds = rng.uniform(2.5, 3.5, NUM_VEHICLES).tolist()
    human_speeds = rng.uniform(1.0, 2.0, NUM_PASSENGERS).tolist()

    env = parallel_env(
        num_humans=NUM_PASSENGERS,
        num_vehicles=NUM_VEHICLES,
        network=network,
        vehicle_capacities=vehicle_capacities,
        vehicle_speeds=vehicle_speeds,
        human_speeds=human_speeds,
        observation_scenario='full',
        render_mode='human',
    )

    obs, info = env.reset(seed=SEED)

    # Place passengers at their assigned boundary starting positions
    for i in range(NUM_PASSENGERS):
        env.agent_positions[f"human_{i}"] = int(start_nodes[i])
        env.human_aboard[f"human_{i}"] = None

    # Spread vehicles across interior nodes
    interior_nodes = [n for n in network.nodes() if n not in boundary_nodes]
    vehicle_start_nodes = rng.choice(interior_nodes, size=NUM_VEHICLES, replace=True)
    for i in range(NUM_VEHICLES):
        env.agent_positions[f"vehicle_{i}"] = int(vehicle_start_nodes[i])

    # ---- 4. Create policies ----
    print("\n4. Creating policies...")
    policies = {}
    for i in range(NUM_PASSENGERS):
        human_id = f"human_{i}"
        policies[human_id] = HeuristicRoutingHumanPolicy(
            human_id, network,
            target_nodes={int(target_nodes_list[i])},
            p_wait=0.5,
            seed=SEED + i,
        )
    for i in range(NUM_VEHICLES):
        vehicle_id = f"vehicle_{i}"
        policies[vehicle_id] = ShortestPathVehiclePolicy(
            vehicle_id, network, seed=SEED + 1000 + i,
        )
    print(f"   Humans: HeuristicRoutingHumanPolicy")
    print(f"   Vehicles: ShortestPathVehiclePolicy")

    # ---- 5. Run simulation, capture snapshot frames and track goals ----
    print("\n5. Running simulation...")

    # Initialise rendering for single-frame capture (no built-in video recording)
    env.enable_rendering('graphical')
    env._initialize_artists()

    # Track which humans have reached their goal (first arrival only)
    goals_reached = set()
    # Snapshots: one network frame + goals count per render interval
    network_frames = []
    frame_goals = []
    frame_times = []

    for step in range(NUM_STEPS):
        # Get actions from policies
        actions = {}
        for agent in env.agents:
            action_space_size = env.action_space(agent).n
            action, _ = policies[agent].get_action(obs[agent], action_space_size)
            actions[agent] = action

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Check for newly reached goals
        for i in range(NUM_PASSENGERS):
            human_id = f"human_{i}"
            if human_id in goals_reached:
                continue
            pos = env.agent_positions.get(human_id)
            if not isinstance(pos, tuple) and pos == target_nodes_list[i]:
                goals_reached.add(human_id)

        # Capture a snapshot every RENDER_INTERVAL steps
        if (step + 1) % RENDER_INTERVAL == 0:
            frame = env.render_frame()
            if frame is not None:
                network_frames.append(frame.copy())
                frame_goals.append(len(goals_reached))
                frame_times.append(env.real_time)

        if (step + 1) % 200 == 0:
            print(f"   Step {step + 1}/{NUM_STEPS}  time={env.real_time:.1f}s  "
                  f"goals reached={len(goals_reached)}/{NUM_PASSENGERS}")

    print(f"\n   Simulation complete: {len(network_frames)} frames captured")
    print(f"   Goals reached: {len(goals_reached)}/{NUM_PASSENGERS}")

    env.close()

    # ---- 6. Build composite video (network + timeseries side-by-side) ----
    print("\n6. Building composite video...")
    n_frames = len(network_frames)
    if n_frames == 0:
        print("   No frames captured; skipping video.")
        return

    max_goals = max(max(frame_goals), 1)

    composite_frames = []
    for idx in range(n_frames):
        fig, (ax_net, ax_ts) = plt.subplots(
            1, 2, figsize=(18, 8),
            gridspec_kw={'width_ratios': [2, 1]},
        )

        # Left panel: network snapshot
        ax_net.imshow(network_frames[idx])
        ax_net.set_axis_off()

        # Right panel: cumulative goals-reached timeseries
        ax_ts.plot(
            frame_times[: idx + 1],
            frame_goals[: idx + 1],
            color='green', linewidth=2,
        )
        ax_ts.set_xlim(0, frame_times[-1] if frame_times[-1] > 0 else 1)
        ax_ts.set_ylim(0, max(max_goals * 1.1, 1))
        ax_ts.set_xlabel('Simulation time (s)', fontsize=12)
        ax_ts.set_ylabel('Goals reached', fontsize=12)
        ax_ts.set_title('Cumulative Goals Reached', fontsize=14, fontweight='bold')
        ax_ts.grid(True, alpha=0.3)
        ax_ts.axhline(y=NUM_PASSENGERS, color='grey', linestyle='--', alpha=0.5,
                       label=f'Total passengers ({NUM_PASSENGERS})')
        ax_ts.legend(loc='upper left', fontsize=10)

        fig.tight_layout()

        # Convert figure to RGB numpy array
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame_rgba = np.asarray(buf)
        # Ensure dimensions divisible by 16 for H.264 codec compatibility
        h, w = frame_rgba.shape[:2]
        h = h - (h % 16)
        w = w - (w % 16)
        composite_frames.append(frame_rgba[:h, :w, :3].copy())
        plt.close(fig)

        if (idx + 1) % 20 == 0:
            print(f"   Composited {idx + 1}/{n_frames} frames")

    # ---- 7. Save composite video ----
    print("\n7. Saving video...")
    filename = 'hexagonal_grid_simulation.mp4'
    try:
        import imageio
        writer = imageio.get_writer(
            filename, fps=VIDEO_FPS,
            codec='libx264', pixelformat='yuv420p', quality=8,
        )
        for cf in composite_frames:
            writer.append_data(cf)
        writer.close()
        print(f"   ✓ Video saved to {filename} "
              f"({len(composite_frames)} frames, {VIDEO_FPS} fps)")
    except Exception as e:
        gif_filename = filename.replace('.mp4', '.gif')
        try:
            from PIL import Image
            pil_frames = [Image.fromarray(f) for f in composite_frames]
            pil_frames[0].save(
                gif_filename, save_all=True, append_images=pil_frames[1:],
                duration=int(1000 / VIDEO_FPS), loop=0,
            )
            print(f"   ✓ Video saved as GIF to {gif_filename} "
                  f"({len(composite_frames)} frames)")
        except Exception as e2:
            print(f"   ✗ Could not save video: {e}, {e2}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"  Network: hexagonal grid, radius {HEX_RADIUS}, "
          f"{network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"  Boundary nodes: {len(boundary_list)}")
    print(f"  Passengers: {NUM_PASSENGERS}  (start on boundary, goal = opposite boundary)")
    print(f"  Vehicles: {NUM_VEHICLES}")
    print(f"  Steps: {NUM_STEPS}")
    print(f"  Goals reached: {len(goals_reached)} / {NUM_PASSENGERS}")
    print(f"  Frames captured: {len(composite_frames)}")
    print(f"  Output: {filename}")
    print("=" * 70)


if __name__ == '__main__':
    main()
