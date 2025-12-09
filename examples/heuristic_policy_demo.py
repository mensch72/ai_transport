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
from ai_transport.policies import HeuristicRoutingHumanPolicy


class SimpleVehiclePolicy:
    """Simple vehicle policy that keeps destination and waits before departing."""
    
    def __init__(self, agent_id, destination, wait_cycles=2, seed=None):
        self.agent_id = agent_id
        self.destination = destination
        self.wait_cycles = wait_cycles  # Number of departing steps to wait before actually departing
        self.depart_count = 0
        self.seed = seed
    
    def get_action(self, observation, action_space_size):
        step_type = observation.get('step_type')
        
        if step_type == 'routing':
            # Keep our destination - find the action that sets it
            action_mapping = observation.get('action_mapping', {})
            details = action_mapping.get('details', {})
            for action_idx in range(action_space_size):
                if details.get(action_idx) == self.destination:
                    return action_idx, f"Keeping destination {self.destination}"
            return 0, "Passing"
        
        elif step_type == 'departing':
            # Wait a few cycles before departing to give humans time to board
            if self.depart_count < self.wait_cycles:
                self.depart_count += 1
                return 0, f"Waiting at node (count {self.depart_count}/{self.wait_cycles})"
            
            # Now depart if we can
            if action_space_size > 1:
                # Take first available edge (action 1)
                return 1, f"Departing toward destination {self.destination}"
            return 0, "Passing"
        
        else:
            return 0, "Passing"
    
    def reset(self):
        self.depart_count = 0


def main():
    print("=" * 70)
    print("AI Transport - Heuristic Routing Human Policy with Video Demo")
    print("=" * 70)
    
    # Create a hand-crafted network for clear demonstration
    print("\n1. Creating custom network...")
    G = nx.DiGraph()
    
    # Create a simple linear network: 0 -> 1 -> 2 -> 3
    # This ensures vehicles going to node 3 pass through nodes 1 and 2
    G.add_node(0, name="Start", x=0.0, y=0.0)
    G.add_node(1, name="Mid1", x=10.0, y=0.0)
    G.add_node(2, name="Mid2", x=20.0, y=0.0)
    G.add_node(3, name="End", x=30.0, y=0.0)
    
    # Add edges
    G.add_edge(0, 1, length=10.0, speed=3.0, capacity=10)
    G.add_edge(1, 2, length=10.0, speed=3.0, capacity=10)
    G.add_edge(2, 3, length=10.0, speed=3.0, capacity=10)
    
    # Create environment
    env = parallel_env(
        num_humans=2,
        num_vehicles=1,
        network=G,
        observation_scenario='full',  # Important: need full observations to see vehicle destinations!
        render_mode="human"
    )
    
    env.reset(seed=42)
    
    # Hand-craft initial positions: all agents at node 0
    print("\n2. Setting up hand-crafted initial scenario...")
    print("   - All agents start at node 0")
    print("   - Vehicle will head to node 3")
    print("   - Humans want to reach node 3")
    
    for agent in env.agents:
        env.agent_positions[agent] = 0
        if agent in env.human_agents:
            env.human_aboard[agent] = None
        if agent in env.vehicle_agents:
            # Pre-set vehicle destination to node 3
            env.vehicle_destinations[agent] = 3
    
    print(f"\n   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Agents: 2 humans, 1 vehicle (all starting at node 0)")
    print(f"   Pre-set destination: vehicle_0 -> node 3")
    
    # Create policies
    print("\n3. Creating policies...")
    
    # Both humans want to reach node 3
    target = {3}
    
    policies = {
        'human_0': HeuristicRoutingHumanPolicy(
            'human_0', G, target_nodes=target, p_wait=0.9, seed=1  # Very high p_wait
        ),
        'human_1': HeuristicRoutingHumanPolicy(
            'human_1', G, target_nodes=target, p_wait=0.9, seed=2  # Very high p_wait
        ),
        'vehicle_0': SimpleVehiclePolicy('vehicle_0', destination=3, wait_cycles=2)  # Wait 2 cycles
    }
    
    print(f"   - human_0: HeuristicRoutingHumanPolicy (target={target}, p_wait=0.9)")
    print(f"   - human_1: HeuristicRoutingHumanPolicy (target={target}, p_wait=0.9)")
    print("   - vehicle_0: SimpleVehiclePolicy (destination=3, wait=2 cycles)")
    print("   Note: Very high p_wait (0.9) + simple network ensures boarding")
    
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
    # Generate fresh observations after manual position setup
    obs = {agent: env._generate_observation_for_agent(agent) for agent in env.agents}
    print(f"   Initial observation for human_0 my_position: {obs['human_0'].get('my_position')}")
    
    boarding_events = []
    unboarding_events = []
    frame_counter = 0
    
    for cycle in range(50):  # More cycles to see movement
        current_step = env.step_type
        
        # Debug output for first few cycles
        if cycle < 5:
            print(f"\n   === Cycle {cycle}: {current_step} ===")
            print(f"      Vehicle destinations before actions: {env.vehicle_destinations}")
            if current_step == 'boarding' and cycle == 2:
                # Check what humans see - print FULL observation for debugging
                print(f"      human_0 FULL observation keys: {obs.get('human_0', {}).keys()}")
                print(f"      human_0 observation vehicle_destinations: {obs.get('human_0', {}).get('vehicle_destinations')}")
                print(f"      human_0 observation action_mapping: {obs.get('human_0', {}).get('action_mapping')}")
                print(f"      human_0 observation my_position: {obs.get('human_0', {}).get('my_position')}")
                print(f"      human_0 observation agent_attributes: {obs.get('human_0', {}).get('agent_attributes')}")
        
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
                
                # Print departing decisions
                if current_step == 'departing' and action > 0:
                    if agent in env.vehicle_agents:
                        edge = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                        print(f"      {agent} departing on edge {edge}")
                    elif agent in env.human_agents:
                        edge = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                        print(f"      {agent} walking on edge {edge}")
                
                # Track boarding events - also print when humans consider boarding
                if current_step == 'boarding' and agent in env.human_agents:
                    if action > 0:
                        vehicle_id = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                        boarding_events.append(f"Cycle {cycle}: {agent} boards {vehicle_id} - {justification}")
                        print(f"\n   🚌 BOARDING: {agent} boards {vehicle_id}")
                        print(f"      Reason: {justification}")
                    else:
                        # Print why not boarding
                        print(f"      {agent}: NOT boarding - {justification}")
                
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
            for agent_id in ['human_0', 'human_1']:
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
