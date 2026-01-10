"""
Demonstration of HeuristicRoutingHumanPolicy with video recording.

This example shows:
1. Humans using heuristic policy with intelligent unboarding
2. Complex scenario with multiple vehicles going different directions
3. Humans boarding, unboarding when vehicle goes wrong way, and walking
4. Graphical rendering with video recording
"""

import networkx as nx
from ai_transport import parallel_env
from ai_transport.policies import HeuristicRoutingHumanPolicy
import matplotlib.pyplot as plt
plt.set_loglevel('warning')




class SimpleVehiclePolicy:
    """Simple vehicle policy that keeps destination and waits before departing."""
    
    def __init__(self, agent_id, destination,graph, wait_cycles=2, seed=None):
        self.agent_id = agent_id
        self.destination = destination
        self.wait_cycles = wait_cycles  # Number of departing steps to wait before actually departing
        self.depart_count = 0
        self.seed = seed
        self.graph =graph
    
    def get_action(self, observation, action_space_size):
        step_type = observation.get('step_type')
        action_mapping = observation.get("action_mapping", {})
        details = action_mapping.get("details", {})
        pos = observation.get('agent_positions')[self.agent_id]


        if step_type == 'routing':
            if pos == self.destination:
                #TODO：find a new destination?
                return 0, f"At destination {self.destination}, passing"
            else:
                for action_idx in range(action_space_size):
                    if details.get(action_idx) == self.destination:
                        return action_idx, f"Keeping destination {self.destination}"
            return 0, "Passing"
        
        elif step_type == 'departing':
            # Wait a few cycles before departing to give humans time to board
            if pos == self.destination:
                return 0, f"At destination {self.destination}, passing"

            if self.depart_count < self.wait_cycles:
                self.depart_count += 1
                return 0, f"Waiting at node (count {self.depart_count}/{self.wait_cycles})"

            # Now depart if we can
            if action_space_size > 1:
                best_action = 0
                best_dist = float("inf")
                for action_id, edge in details.items():
                    if edge is None:
                        continue
                    _, v = edge  # edge = (current_node, next_node)
                    try:
                        dist = nx.shortest_path_length(self.graph, source=v,target=self.destination,weight="length")
                    except nx.NetworkXNoPath:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_action = action_id
                if best_action != 0:
                    return best_action, (f"Departing via best edge, next={details[best_action][1]}, "f"dist_to_dest={best_dist:.2f}")
                return 0, "Passing"


            return 0, "Passing"
        
        else:
            return 0, "Passing"
    
    def reset(self):
        self.depart_count = 0




def main():

    print("=" * 70)
    print("AI Transport - Complex Heuristic Routing Demo with Unboarding")
    print("=" * 70)
    
    # Create a branching network for complex scenarios
    print("\n1. Creating complex network...")
    G = nx.DiGraph()
    
    # Network structure - branching paths:
    #       2---3---4
    #      /         \
    #     1           5---6
    #    /             \
    #   0               7
    #    \             /
    #     8---9------10
    
    positions = {
        0: (0, 0),
        1: (5, 5), 2: (10, 8), 3: (15, 8), 4: (20, 8),
        5: (23, 4), 6: (28, 4), 7: (23, 0),
        8: (5, -5), 9: (10, -5), 10: (18, -3)
    }
    
    for node_id, (x, y) in positions.items():
        G.add_node(node_id, name=f"Node_{node_id}", x=x, y=y)
    
    # Add edges - create multiple paths
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7),  # Upper path
        (0, 8), (8, 9), (9, 10), (10, 7),  # Lower path
        (10, 5)  # Connection
    ]
    
    for u, v in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        G.add_edge(u, v, length=length, speed=3.0, capacity=10)
        G.add_edge(v, u, length=length, speed=3.0, capacity=10)
    
    # Create environment
    env = parallel_env(
        num_humans=3,
        num_vehicles=2,
        network=G,
        observation_scenario='full',  # Important for policy
        render_mode="human"
    )
    
    env.reset(seed=42)
    
    # Hand-craft initial positions for interesting scenario
    print("\n2. Setting up complex scenario...")
    print("   SCENARIO:")
    print("   - human_0 at node 0, target: node 6 (far upper right)")
    print("   - human_1 at node 0, target: node 7 (middle right)")
    print("   - human_2 at node 0, target: node 10 (lower right)")
    print("   - vehicle_0 at node 0, destination: node 4 (upper path - WRONG for most)")
    print("   - vehicle_1 at node 0, destination: node 10 (lower path - GOOD for human_2)")
    
    # Position everyone at node 0
    for agent in env.agents:
        env.agent_positions[agent] = 0
        if agent in env.human_agents:
            env.human_aboard[agent] = None
        if agent in env.vehicle_agents:
            if agent == 'vehicle_0':
                env.vehicle_destinations[agent] = 4  # Goes upper path
            else:
                env.vehicle_destinations[agent] = 10  # Goes lower path

    print(f"\n   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Agents: 3 humans, 2 vehicles (all starting at node 0)")


    # Create policies
    print("\n3. Creating policies...")
    human_wait=0.7
    policies = {'human_0': HeuristicRoutingHumanPolicy('human_0', G, target_nodes={1}, p_wait=human_wait, seed=1),
        'human_1': HeuristicRoutingHumanPolicy('human_1', G, target_nodes={2}, p_wait=human_wait, seed=2),
        'human_2': HeuristicRoutingHumanPolicy('human_2', G, target_nodes={8}, p_wait=human_wait, seed=3),
        'vehicle_0': SimpleVehiclePolicy('vehicle_0', destination=1,graph=G, wait_cycles=2),
        'vehicle_1': SimpleVehiclePolicy('vehicle_1', destination=8, graph=G,wait_cycles=2)}
    
    print(f"   - human_0: target {1} (far upper), p_wait={human_wait}")
    print(f"   - human_1: target {2} (middle), p_wait={human_wait}")
    print(f"   - human_2: target {8} (lower), p_wait={human_wait}")
    print("   - vehicle_0: destination 1 (upper path)")
    print("   - vehicle_1: destination 8 (lower path)")
    
    # Enable graphical rendering and start video recording
    print("\n4. Starting video recording...")
    env.enable_rendering('graphical')
    env.start_video_recording()
    
    env.render()
    env.save_frame('complex_heuristic_initial.png')
    print("   Saved initial frame")
    
    # Run simulation
    print("\n5. Running complex simulation...")


    obs = {agent: env._generate_observation_for_agent(agent) for agent in env.agents}
    
    boarding_events = []
    unboarding_events = []
    walking_events = []
    frame_counter = 0
    
    for cycle in range(80):  # More cycles for complex scenario
        current_step = env.step_type


        # Get actions from policies
        actions = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                action, justification = policy.get_action(obs[agent], action_space_size)
                actions[agent] = action

                # Track events
                if current_step == 'boarding' and agent in env.human_agents and action > 0:
                    vehicle_id = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                    boarding_events.append(f"Cycle {cycle}: {agent} boards {vehicle_id}")
                    print(f"\n   🚌 BOARDING (cycle {cycle}): {agent} boards {vehicle_id}")
                    print(f"      {justification}")
                
                if current_step == 'unboarding' and agent in env.human_agents and action > 0:
                    aboard = env.human_aboard.get(agent)
                    unboarding_events.append(f"Cycle {cycle}: {agent} unboards from {aboard}")
                    print(f"\n   🚶 UNBOARDING (cycle {cycle}): {agent} unboards from {aboard}")
                    print(f"      {justification}")
                
                if current_step == 'departing' and agent in env.human_agents and action > 0:
                    if env.human_aboard.get(agent) is None:  # Walking, not riding
                        edge = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                        walking_events.append(f"Cycle {cycle}: {agent} walks")
                        if len(walking_events) <= 5:  # Only print first few
                            print(f"\n   🚶 WALKING (cycle {cycle}): {agent}")
                            print(f"      {justification}")
            else:
                actions[agent] = 0
        
        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)

        if any(terms.values()) or any(truncs.values()):
            print(f"Episode ended at cycle {cycle}")
            break

        # Render after departing step
        if env.step_type == 'routing':
            env.render()
            frame_counter += 1
        
        # Progress report
        if (cycle + 1) % 20 == 0:
            print(f"\n   Progress: Cycle {cycle + 1}/80")
            for agent_id in ['human_0', 'human_1', 'human_2']:
                pos = env.agent_positions.get(agent_id)
                aboard = env.human_aboard.get(agent_id)
                target = policies[agent_id].target_nodes
                pos_str = str(pos) if not isinstance(pos, tuple) else f"edge {pos[0]}"
                aboard_str = f" (aboard {aboard})" if aboard else ""
                at_target = "✓ TARGET" if pos in target else ""
                print(f"      {agent_id}: {pos_str}{aboard_str} -> {target} {at_target}")
    
    # Save video
    print("\n6. Saving video...")
    env.save_video('complex_heuristic_demo.mp4', fps=5)
    env.save_frame('complex_heuristic_final.png')
    print("   Saved video and final frame")


    # Close environment

    env.close()
    
    print("\n" + "=" * 70)
    print("Complex Demo Complete!")
    print("=" * 70)
    
    print("\n📊 EVENT SUMMARY:")
    print(f"   - Boarding events: {len(boarding_events)}")
    print(f"   - Unboarding events: {len(unboarding_events)}")
    print(f"   - Walking events: {len(walking_events)}")
    
    if boarding_events:
        print("\n🚌 BOARDING EVENTS:")
        for event in boarding_events:
            print(f"   {event}")
    
    if unboarding_events:
        print("\n🚶 UNBOARDING EVENTS:")
        for event in unboarding_events:
            print(f"   {event}")
    
    print("\n📁 Generated files:")
    print("   - complex_heuristic_initial.png")
    print("   - complex_heuristic_final.png")
    print("   - complex_heuristic_demo.mp4")
    
    print("\n💡 SCENARIO OUTCOMES:")
    print("   This demo showcases:")
    print("   1. Intelligent boarding - humans board vehicles going toward their targets")
    print("   2. Smart unboarding - humans exit when:")
    print("      a) Vehicle reaches their planned exit node")
    print("      b) Vehicle goes wrong direction (distance to target increases)")
    print("      c) Better vehicle becomes available")
    print("   3. Walking as fallback - humans walk when no good vehicles available")
    print("   4. Different strategies - each human makes independent decisions")


if __name__ == "__main__":
    main()


