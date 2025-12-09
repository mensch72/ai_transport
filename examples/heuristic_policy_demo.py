"""
Demonstration of HeuristicRoutingHumanPolicy with video recording.

This example shows:
1. Humans using heuristic policy to board vehicles strategically
2. Hand-crafted initial situation ensuring humans board vehicles
3. Graphical rendering with video recording
4. Humans reaching their target destinations intelligently
"""

import os
import networkx as nx
from ai_transport import parallel_env
from ai_transport.policies import (
    HeuristicRoutingHumanPolicy,
    ShortestPathVehiclePolicy
)


def main():
    print("=" * 70)
    print("AI Transport - Heuristic Routing Human Policy with Video Demo")
    print("=" * 70)
    
    # Create a hand-crafted network for clear demonstration
    print("\n1. Creating custom network...")
    G = nx.DiGraph()
    
    # Create a network with clear paths:
    #     1---2---3
    #    /         \
    #   0           4---5
    #    \         /
    #     6---7---8
    
    # Add nodes with coordinates
    positions = {
        0: (0, 0),
        1: (5, 5),
        2: (10, 5),
        3: (15, 5),
        4: (20, 0),
        5: (25, 0),
        6: (5, -5),
        7: (10, -5),
        8: (15, -5)
    }
    
    for node_id, (x, y) in positions.items():
        G.add_node(node_id, name=f"Node_{node_id}", x=x, y=y)
    
    # Add edges with length and speed
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),  # Upper path
        (0, 6), (6, 7), (7, 8), (8, 4),  # Lower path
        (1, 6), (2, 7), (3, 8)  # Cross connections
    ]
    
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        G.add_edge(u, v, length=length, speed=3.0, capacity=10)
    
    # Create environment
    env = parallel_env(
        num_humans=3,
        num_vehicles=2,
        network=G,
        render_mode="human"
    )
    
    env.reset(seed=42)
    
    # Hand-craft initial positions: all humans and vehicles at node 0
    print("\n2. Setting up hand-crafted initial scenario...")
    print("   - All agents start at node 0")
    print("   - Humans have different target destinations")
    print("   - Vehicles will head toward targets, allowing humans to board")
    
    for agent in env.agents:
        env.agent_positions[agent] = 0
        if agent in env.human_agents:
            env.human_aboard[agent] = None
        if agent in env.vehicle_agents:
            env.vehicle_destinations[agent] = None
    
    print(f"\n   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Agents: 3 humans, 2 vehicles (all starting at node 0)")
    
    # Create policies
    print("\n3. Creating policies...")
    
    # Define target nodes for humans
    # human_0 wants to reach node 5 (far end of upper path)
    # human_1 wants to reach node 4 (junction of paths)  
    # human_2 wants to reach node 8 (far end of lower path)
    target_0 = {5}
    target_1 = {4}
    target_2 = {8}
    
    policies = {
        'human_0': HeuristicRoutingHumanPolicy(
            'human_0', G, target_nodes=target_0, p_wait=0.2, seed=1
        ),
        'human_1': HeuristicRoutingHumanPolicy(
            'human_1', G, target_nodes=target_1, p_wait=0.3, seed=2
        ),
        'human_2': HeuristicRoutingHumanPolicy(
            'human_2', G, target_nodes=target_2, p_wait=0.2, seed=3
        ),
        'vehicle_0': ShortestPathVehiclePolicy('vehicle_0', G, seed=5),
        'vehicle_1': ShortestPathVehiclePolicy('vehicle_1', G, seed=6)
    }
    
    print(f"   - human_0: HeuristicRoutingHumanPolicy (target={target_0}, p_wait=0.2)")
    print(f"   - human_1: HeuristicRoutingHumanPolicy (target={target_1}, p_wait=0.3)")
    print(f"   - human_2: HeuristicRoutingHumanPolicy (target={target_2}, p_wait=0.2)")
    print("   - vehicle_0: ShortestPathVehiclePolicy")
    print("   - vehicle_1: ShortestPathVehiclePolicy")
    
    # Enable graphical rendering and start video recording
    print("\n4. Starting video recording...")
    env.enable_rendering('graphical')
    env.start_video_recording()
    
    # Render initial state
    env.render()
    env.save_frame('heuristic_policy_initial.png')
    print("   Saved initial frame to heuristic_policy_initial.png")
    
    # Run simulation
    print("\n5. Running simulation...")
    obs = {agent: env._generate_observation_for_agent(agent) for agent in env.agents}
    
    boarding_events = []
    unboarding_events = []
    frame_counter = 0
    
    for cycle in range(50):  # More cycles to see movement
        current_step = env.step_type
        
        # Print vehicle destinations at routing step
        if current_step == 'routing' and cycle % 10 == 0:
            print(f"\n   Routing step (cycle {cycle}):")
            for vehicle in ['vehicle_0', 'vehicle_1']:
                dest = env.vehicle_destinations.get(vehicle)
                pos = env.agent_positions.get(vehicle)
                print(f"      {vehicle} at {pos}, destination: {dest}")
        
        # Get actions from policies
        actions = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                action, justification = policy.get_action(obs[agent], action_space_size)
                actions[agent] = action
                
                # Print vehicle routing decisions
                if current_step == 'routing' and agent in env.vehicle_agents and action > 0:
                    new_dest = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                    print(f"      {agent} setting destination to {new_dest}")
                
                # Track boarding events
                if current_step == 'boarding' and agent in env.human_agents and action > 0:
                    vehicle_id = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                    boarding_events.append(f"Cycle {cycle}: {agent} boards {vehicle_id} - {justification}")
                    print(f"\n   🚌 BOARDING: {agent} boards {vehicle_id}")
                    print(f"      Reason: {justification}")
                
                # Track unboarding events
                if current_step == 'unboarding' and agent in env.human_agents and action > 0:
                    aboard = env.human_aboard.get(agent)
                    unboarding_events.append(f"Cycle {cycle}: {agent} unboards from {aboard}")
                    print(f"\n   🚶 UNBOARDING: {agent} unboards from {aboard}")
            else:
                actions[agent] = 0
        
        # Take step
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Render after departing step (when movement happens)
        if env.step_type == 'routing':  # Just finished departing
            env.render()
            frame_counter += 1
            
        if (cycle + 1) % 10 == 0:
            print(f"\n   Progress: Cycle {cycle + 1}/50 (time: {env.real_time:.2f}s, step: {env.step_type})")
            # Show current positions
            for agent_id in ['human_0', 'human_1', 'human_2']:
                pos = env.agent_positions.get(agent_id)
                aboard = env.human_aboard.get(agent_id)
                target = policies[agent_id].target_nodes
                pos_str = str(pos) if not isinstance(pos, tuple) else f"edge {pos[0]}"
                aboard_str = f" (aboard {aboard})" if aboard else ""
                at_target = "✓ AT TARGET" if pos in target else ""
                print(f"      {agent_id}: at {pos_str}{aboard_str} -> target {target} {at_target}")
    
    # Save video
    print("\n6. Saving video...")
    env.save_video('heuristic_policy_demo.mp4', fps=5)
    
    # Save final frame
    env.save_frame('heuristic_policy_final.png')
    print("   Saved final frame to heuristic_policy_final.png")
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    print("\n📊 Summary:")
    print(f"   - Total boarding events: {len(boarding_events)}")
    print(f"   - Total unboarding events: {len(unboarding_events)}")
    print(f"   - Video frames captured: {frame_counter}")
    
    if boarding_events:
        print("\n🚌 Boarding events:")
        for event in boarding_events[:10]:  # Show first 10
            print(f"   {event}")
        if len(boarding_events) > 10:
            print(f"   ... and {len(boarding_events) - 10} more")
    
    print("\n📁 Generated files:")
    print("   - heuristic_policy_initial.png (initial state)")
    print("   - heuristic_policy_final.png (final state)")
    print("   - heuristic_policy_demo.mp4 (full simulation video)")
    
    print("\n💡 Policy Behavior Demonstrated:")
    print("   - Humans start at node 0 with vehicles")
    print("   - Vehicles set destinations using ShortestPathVehiclePolicy")
    print("   - Humans evaluate which vehicles go toward their targets")
    print("   - Humans board vehicles that take them closer to targets")
    print("   - Humans may unboard at intermediate nodes to switch routes")
    print("   - Humans walk if no good vehicle options available")
    
    print("\n📝 Note: To enable video recording, install imageio with:")
    print("   pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()
