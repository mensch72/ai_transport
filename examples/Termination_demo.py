# python
"""
Termination_demo.py

This script builds simple synthetic observations and prints policy decisions.
"""
import sys
import networkx as nx
import numpy as np
from ai_transport.envs import parallel_env
from ai_transport.policies import ShortestPathVehiclePolicy,TargetDestinationHumanPolicy, HeuristicRoutingHumanPolicy



def main():
    seed = 42
    rng = np.random.RandomState(seed)

    # 1) Create network
    print("\n1. Creating network...")
    print("=" * 70)
    tmp_env = parallel_env(num_humans=3, num_vehicles=2, observation_scenario="full", render_mode="human")
    G = tmp_env.create_random_2d_network(num_nodes=10, bidirectional_prob=0.85,
                                     speed_mean=5.0, capacity_mean=10.0,
                                     coord_mean=0.0, coord_std=10.0, seed=seed)
    tmp_env.close()
    print(f"\n   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


    # Create environment
    env = parallel_env(
        num_humans=3,
        num_vehicles=2,
        network=G,
        observation_scenario='full',  # Important for policy
        render_mode="human"
    )

    env.reset(seed=seed )




    # 2) Create policies for agents (with waiting and random-edge noise)
    print("\n2. Creating policies...")
    print("=" * 70)
    #paremeters
    start_node = 0
    candidate_nodes = [n for n in G.nodes() if n != start_node]
    human_agents = [a for a in env.agents if a in env.human_agents]
    human_dests = rng.choice(candidate_nodes, size=len(human_agents), replace=False)
    human_dest_map = dict(zip(human_agents, map(int, human_dests)))

    human_wait = 0.3
    human_target_change_rate = 0.3
    vehicle_wait_cycles = 2
    target_policy_fraction = 0.33

    policies = {}
    for agent in env.agents:
        env.agent_positions[agent] =  start_node
        if agent in env.human_agents:
            dest = human_dest_map[agent]
            env.human_aboard[agent] = None
            env.human_destinations[agent] = dest
            if rng.random() < target_policy_fraction:
                policies[agent] = TargetDestinationHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_change_rate=human_target_change_rate,
                    seed=seed
                )
            else:
                policies[agent] = HeuristicRoutingHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_nodes={dest},
                    p_wait=human_wait,
                    seed=seed
                )
        elif agent in env.vehicle_agents:
            policies[agent] = ShortestPathVehiclePolicy(
                agent_id=agent,
                network=G,
                wait_cycles=vehicle_wait_cycles,
                seed=seed
            )
    print("\n   Agent targets:")
    for agent in sorted(env.agents):
        dest = env.human_destinations.get(agent) or env.vehicle_destinations.get(agent)
        agent_type = "human" if agent in env.human_agents else "vehicle"
        print(f"   - {agent}: {agent_type} → {dest}")


    # 3) Run simulation
    print("\n3. Running complex simulation...")
    print("=" * 70)

    #Enable graphical rendering and start video recording
    env.enable_rendering('graphical')
    env.start_video_recording()
    env.render()

    #Run simulation
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
                result = policy.get_action(obs[agent], action_space_size)
                if result is None:
                    # 某些 policy 可能在某些步骤返回 None
                    actions[agent] = 0
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    action, justification = result
                else:
                    # 只返回 action（如 ShortestPathVehiclePolicy）
                    action = result
                    justification = "No justification available"

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

        # --- DEBUG: show how terminate becomes True ---
        should_end = any(terms.values())

        # 只在接近结束或已经结束时打印（避免刷屏）
        # 你也可以改成每 N 步打印一次
        if should_end or (cycle % 20 == 0 and env.step_type == 'routing'):
            has_active_dest = False
            all_reached = True
            print(f"  cycle={cycle} | env.step_type(after step)={env.step_type} | terminated={should_end}")
            for a in sorted(env.agents):
                if a in env.human_agents:
                    dest = env.human_destinations.get(a)
                else:
                    dest = env.vehicle_destinations.get(a)
                pos = env.agent_positions.get(a)
                if dest is not None:
                    has_active_dest = True
                on_edge = isinstance(pos, tuple)
                reached = (dest is not None) and (not on_edge) and (pos == dest)
                if dest is not None and not reached:
                    all_reached = False
                who = "human" if a in env.human_agents else "vehicle"
                pos_str = f"edge{pos}" if on_edge else str(pos)
                print(f"  - {a:10s} ({who}) pos={pos_str:>10s} dest={dest} reached={reached}")
            print(f"  => has_active_dest={has_active_dest} | all_reached={all_reached} | terminate() would return {has_active_dest and all_reached}")
        # --- END DEBUG ---

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
            for agent_id in env.human_agents:
                pos = env.agent_positions.get(agent_id)
                aboard = env.human_aboard.get(agent_id)
                policy = policies[agent_id]
                if hasattr(policy, "target_nodes"):
                    target = policy.target_nodes
                else:
                    # TargetDestinationHumanPolicy 没有 target_nodes，那就用 env 里存的目的地
                    target = {env.human_destinations.get(agent_id)}
                pos_str = str(pos) if not isinstance(pos, tuple) else f"edge {pos[0]}"
                aboard_str = f" (aboard {aboard})" if aboard else ""
                at_target = "✓ TARGET" if pos in target else ""
                print(f"      {agent_id}: {pos_str}{aboard_str} -> {target} {at_target}")

    # Save video
    env.save_video('Termination_demo.mp4', fps=5)

    # Close environment
    env.close()

    print("Complex Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()