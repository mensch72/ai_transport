"""
Example demonstrating visualization and video recording.

This example shows:
1. Graphical rendering of the transport network
2. Recording frames during simulation
3. Saving video as MP4
"""

import numpy as np
import networkx as nx
from ai_transport import parallel_env


def main():
    print("=" * 70)
    print("AI Transport Environment - Visualization Demo")
    print("=" * 70)
    
    # Create a random 2D network
    print("\n1. Creating random 2D network...")
    env = parallel_env(num_humans=3, num_vehicles=2)
    network = env.create_random_2d_network(
        num_nodes=8,
        bidirectional_prob=0.4,
        speed_mean=5.0,
        capacity_mean=10.0,
        coord_std=10.0,
        seed=42
    )
    
    # Create environment with the network
    env = parallel_env(
        num_humans=3,
        num_vehicles=2,
        network=network,
        render_mode="human"
    )
    
    env.reset(seed=42)
    env.initialize_random_positions(seed=42)
    
    print(f"   Network: {len(network.nodes())} nodes, {len(network.edges())} edges")
    print(f"   Agents: {len(env.human_agents)} humans, {len(env.vehicle_agents)} vehicles")
    
    # Enable graphical rendering
    print("\n2. Enabling graphical rendering...")
    env.enable_rendering('graphical')
    
    # Render initial state
    print("\n3. Rendering initial state...")
    env.render()
    env.save_frame('transport_initial.png')
    print("   Saved initial frame to transport_initial.png")
    
    # Start video recording
    print("\n4. Starting video recording...")
    env.start_video_recording()
    
    # Run simulation for several steps
    print("\n5. Running simulation...")
    num_steps = 20
    
    for step in range(num_steps):
        # Determine actions based on step type
        actions = {}
        
        if env.step_type == 'routing':
            # Vehicles set random destinations
            for agent in env.agents:
                if agent in env.vehicle_agents:
                    pos = env.agent_positions[agent]
                    if not isinstance(pos, tuple):  # At a node
                        # Randomly choose to set destination or pass
                        if np.random.random() < 0.5:
                            nodes = list(env.network.nodes())
                            dest_idx = np.random.randint(len(nodes) + 1)
                            actions[agent] = dest_idx  # 0 = None, 1..N = nodes
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif env.step_type == 'unboarding':
            # Some humans unboard
            for agent in env.agents:
                if agent in env.human_agents:
                    aboard = env.human_aboard.get(agent)
                    if aboard is not None:
                        vehicle_pos = env.agent_positions[aboard]
                        if not isinstance(vehicle_pos, tuple):  # Vehicle at node
                            # Randomly unboard or stay
                            actions[agent] = 1 if np.random.random() < 0.3 else 0
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif env.step_type == 'boarding':
            # Humans try to board
            for agent in env.agents:
                if agent in env.human_agents:
                    pos = env.agent_positions[agent]
                    aboard = env.human_aboard.get(agent)
                    if not isinstance(pos, tuple) and aboard is None:  # At node, not aboard
                        # Find vehicles at same node
                        vehicles_here = []
                        for v in env.vehicle_agents:
                            v_pos = env.agent_positions[v]
                            if not isinstance(v_pos, tuple) and v_pos == pos:
                                vehicles_here.append(v)
                        if vehicles_here:
                            # Try to board first vehicle
                            actions[agent] = 1
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif env.step_type == 'departing':
            # Agents depart on random edges
            for agent in env.agents:
                pos = env.agent_positions[agent]
                if not isinstance(pos, tuple):  # At a node
                    if agent in env.vehicle_agents:
                        outgoing = list(env.network.out_edges(pos))
                        if outgoing and np.random.random() < 0.6:
                            actions[agent] = 1  # Depart on first edge
                        else:
                            actions[agent] = 0
                    elif agent in env.human_agents:
                        aboard = env.human_aboard.get(agent)
                        if aboard is None:  # Not aboard
                            outgoing = list(env.network.out_edges(pos))
                            if outgoing and np.random.random() < 0.4:
                                actions[agent] = 1  # Walk on first edge
                            else:
                                actions[agent] = 0
                        else:
                            actions[agent] = 0
                else:
                    actions[agent] = 0
        
        # Take step
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Render (this will record the frame)
        env.render()
        
        # Cycle through step types
        step_types = ['routing', 'unboarding', 'boarding', 'departing']
        current_idx = step_types.index(env.step_type)
        env.step_type = step_types[(current_idx + 1) % len(step_types)]
        
        if (step + 1) % 5 == 0:
            print(f"   Step {step + 1}/{num_steps} completed (time: {env.real_time:.2f})")
    
    # Save video
    print("\n6. Saving video...")
    env.save_video('transport_simulation.mp4', fps=2)
    
    # Save final frame
    print("\n7. Saving final frame...")
    env.save_frame('transport_final.png')
    print("   Saved final frame to transport_final.png")
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - transport_initial.png (initial state)")
    print("  - transport_final.png (final state)")
    print("  - transport_simulation.mp4 (full simulation video)")
    print("\nNote: To enable video recording, install imageio with:")
    print("  pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()
