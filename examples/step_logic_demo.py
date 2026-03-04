"""
Simple demo to check if step logic is correct.
Verifies that step_type cycles correctly: routing -> unboarding -> boarding -> departing -> routing...
"""

import networkx as nx
from ai_transport import parallel_env


def main():
    # Create a simple network
    G = nx.DiGraph()
    G.add_node(0, name="A")
    G.add_node(1, name="B")
    G.add_node(2, name="C")

    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=8)
    G.add_edge(2, 0, length=12.0, speed=6.0, capacity=12)

    # Create environment
    env = parallel_env(
        num_humans=2,
        num_vehicles=2,
        network=G
    )

    env.reset(seed=345)

    # Set all agents at node 0
    env.agent_positions['human_0'] = 0
    env.agent_positions['human_1'] = 0
    env.agent_positions['vehicle_0'] = 0
    env.agent_positions['vehicle_1'] = 0
    env.human_aboard['human_0'] = None
    env.human_aboard['human_1'] = None

    # Set destinations so agents know where to go
    env.vehicle_destinations['vehicle_0'] = 2
    env.vehicle_destinations['vehicle_1'] = 2
    env.human_destinations['human_0'] = 2
    env.human_destinations['human_1'] = 1

    print("=" * 70)
    print("AI Transport Environment - Step Logic Demo")
    print("=" * 70)
    print(f"\nInitial state:")
    print(f"  positions: {env.agent_positions}")
    print(f"  aboard: {env.human_aboard}")
    print(f"  vehicle_destinations: {env.vehicle_destinations}")
    print(f"  human_destinations: {env.human_destinations}\n")

    # Run 16 steps to see 4 complete cycles
    for step_num in range(9):
        current_step_type = env.step_type

        print('-' * 70)
        print(f"Step {step_num}: {current_step_type.upper()}")
        print('-' * 70)

        # Show current state BEFORE step
        print(f"  Before:")
        print(f"    real_time: {env.real_time:.2f}")
        print(f"    positions: {env.agent_positions}")
        print(f"    aboard: {env.human_aboard}")

        # Choose meaningful actions based on step_type
        actions = {}
        for agent in env.agents:
            space = env.action_space(agent)
            mapping = env._get_action_mapping(agent)
            action = 0  # Default: pass

            # ROUTING: vehicles set destinations
            if current_step_type == 'routing' and agent in env.vehicle_agents:
                # If destination not set, set it
                dest = env.vehicle_destinations.get(agent)
                if dest is None:
                    # Find action to set destination to node 1 or 2
                    for act_idx, detail in mapping.get('details', {}).items():
                        if detail == 1 or detail == 2:
                            action = act_idx
                            break

            # UNBOARDING: humans check if they reached destination
            elif current_step_type == 'unboarding' and agent in env.human_agents:
                # If aboard and at destination node, unboard
                aboard_vehicle = env.human_aboard.get(agent)
                if aboard_vehicle is not None:
                    dest = env.human_destinations.get(agent)
                    current_pos = env.agent_positions.get(agent)
                    # If reached destination, unboard
                    if current_pos == dest:
                        space_n = getattr(space, 'n', 1)
                        if space_n > 1:
                            action = 1  # Unboard

            # BOARDING: humans try to board vehicles
            elif current_step_type == 'boarding' and agent in env.human_agents:
                # Check if already at destination - if yes, don't board
                dest = env.human_destinations.get(agent)
                current_pos = env.agent_positions.get(agent)

                # Only board if: not aboard, not at destination, and vehicles available
                if env.human_aboard.get(agent) is None and current_pos != dest:
                    space_n = getattr(space, 'n', 1)
                    if space_n > 1:
                        # Choose first available vehicle (action 1)
                        action = 1

            # DEPARTING: vehicles depart on edges
            elif current_step_type == 'departing' and agent in env.vehicle_agents:
                # If at node with outgoing edges, depart
                space_n = getattr(space, 'n', 1)
                if space_n > 1:
                    action = 1  # Depart on first edge

            actions[agent] = action

        # Print actions with descriptions
        print(f"  Actions:")
        for agent, action in actions.items():
            mapping = env._get_action_mapping(agent)
            desc = mapping.get('description', {}).get(action, 'unknown')
            detail = mapping.get('details', {}).get(action, None)
            print(f"    {agent}: action={action}, desc='{desc}', detail={detail}")


        # Execute the sampled actions
        env.step(actions)

        # Show state AFTER step
        print(f"  After:")
        print(f"    real_time: {env.real_time:.2f}")
        print(f"    positions: {env.agent_positions}")
        print(f"    aboard: {env.human_aboard}")
        print(f"    -> next step_type: {env.step_type.upper()}\n")



if __name__ == "__main__":
    main()

