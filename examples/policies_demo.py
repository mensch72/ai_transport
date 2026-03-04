# python
"""
env_human_vehicle_policy_test
This script builds simple synthetic observations and prints policy decisions.
"""
import sys
import networkx as nx
import numpy as np
from ai_transport.envs import parallel_env
from ai_transport.policies import ShortestPathVehiclePolicy,TargetDestinationHumanPolicy, HeuristicRoutingHumanPolicy,TraceDestinationHumanPolicy



def main():
    seed = 42
    rng = np.random.default_rng(seed)

    # 1) Create network
    print("\n1. Creating network...")
    print("=" * 70)
    tmp_env = parallel_env(num_humans=4, num_vehicles=3, observation_scenario="full", render_mode="human")
    G = tmp_env.create_random_2d_network(num_nodes=10, bidirectional_prob=0.85,
                                     speed_mean=5.0, capacity_mean=10.0,
                                     coord_mean=0.0, coord_std=10.0, seed=seed)
    tmp_env.close()
    print(f"\n   Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")


    # Create environment
    env = parallel_env(
        num_humans=4,
        num_vehicles=3,
        network=G,
        observation_scenario='full',  # Important for policy
        render_mode="human"
    )

    env.reset(seed=seed )




    # 2) Create policies for agents (with waiting and random-edge noise)
    def build_human_target_config(human_agents, candidate_nodes, rng, trajectory_length=5):
        config = {}
        for agent in human_agents:
            #1.generate a random list of nodes (trajectory)
            trajectory = [int(rng.choice(candidate_nodes)) for _ in range(trajectory_length)]
            # choose a random node from the trajectory as the target (can be any node in the trajectory)
            target = int(rng.choice(trajectory))
            # create a set of unique nodes in the trajectory for heuristic policy
            target_nodes = set(trajectory)
            config[agent] = {
                "target": target,
                "target_nodes": target_nodes,
                "target_trajectory": trajectory
            }
        return config

    print("\n2. Creating policies...")
    print("=" * 70)
    #paremeters
    start_node = 0
    candidate_nodes = [n for n in G.nodes() if n != start_node]
    human_agents = [a for a in env.agents if a in env.human_agents]
    human_dests = build_human_target_config(human_agents,candidate_nodes,rng)


    human_wait = 0.7
    human_target_change_rate = 0.01
    vehicle_wait_cycles = 2
    target_policy_probablity = 0.1
    trace_policy_probablity = 0.4

    policies = {}
    for agent in env.agents:
        env.agent_positions[agent] =  start_node
        if agent in env.human_agents:
            env.human_aboard[agent] = None
            r = rng.random()
            if r < target_policy_probablity:
                dest = human_dests[agent]["target"]
                policies[agent] = TargetDestinationHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_change_rate=human_target_change_rate,
                    seed=seed
                )
            elif r < target_policy_probablity + trace_policy_probablity:
                dest_list = human_dests[agent]["target_trajectory"]
                #env.human_destinations[agent] = dest_list
                policies[agent] = TraceDestinationHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_trajectory=dest_list,
                    p_wait=human_wait,
                    seed=seed
                )
            else:
                dest_set = human_dests[agent]["target_nodes"]
                #env.human_destinations[agent] = dest_set
                policies[agent] = HeuristicRoutingHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_nodes=dest_set,
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
        policy = policies.get(agent)
        policy_name = policy.__class__.__name__ if policy is not None else "None"
        if agent in env.human_agents:
            env_dest = env.human_destinations.get(agent)
            agent_type = "human"
            if isinstance(policy, TargetDestinationHumanPolicy):
                policy_dest = policy.target
            elif isinstance(policy, HeuristicRoutingHumanPolicy):
                policy_dest = policy.target_nodes
            elif isinstance(policy, TraceDestinationHumanPolicy):
                policy_dest = policy._current_target()
            else:
                policy_dest = None
        else:
            env_dest = env.vehicle_destinations.get(agent)
            agent_type = "vehicle"
            # 策略内部真正使用的目标
            if isinstance(policy, ShortestPathVehiclePolicy):
                policy_dest = policy.current_destination
            else:
                policy_dest = None
        if hasattr(env_dest, "item"):
            env_dest = env_dest.item()
        print(f"   - {agent}: {agent_type} | "f"env_dest={env_dest} | "f"policy_dest={policy_dest} | "f"policy={policy_name}")

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

    for agent, policy in policies.items():
        if agent in env.human_agents:
            # TargetDestinationHumanPolicy: policy.target (single node)
            if isinstance(policy, TargetDestinationHumanPolicy):
                env.human_destinations[agent] = policy.target


            # HeuristicRoutingHumanPolicy: policy.target_nodes (set of nodes)
            elif isinstance(policy, HeuristicRoutingHumanPolicy):
                env.human_destinations[agent] = set(policy.target_nodes)

            # TraceDestinationHumanPolicy: policy._current_target() (current node in trajectory)
            elif isinstance(policy, TraceDestinationHumanPolicy):
                env.human_destinations[agent] = policy._current_target()

        elif agent in env.vehicle_agents:
            # ShortestPathVehiclePolicy: policy.current_destination
            if hasattr(policy, "current_destination"):
                env.vehicle_destinations[agent] = policy.current_destination

    env.termination_mode = "current_active"
    env.freeze_termination_participants()
    print("FROZEN termination_participants =", env.termination_participants)

    for cycle in range(120):  # More cycles for complex scenario
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
        active = [v for v in terms.values() if v is not None]
        should_term=(len(active) > 0) and all(active)


        if should_term:
            print(f"Episode ended at cycle {cycle}")
            break

        # Render after departing step
        if env.step_type == 'routing':
            env.render()
            frame_counter += 1

        # Progress report
        if (cycle + 1) % 10 == 0:
            print(f"\n   Progress: Cycle {cycle + 1}/120")
            #print('should end',terms)
            for agent_id in env.human_agents:
                pos = env.agent_positions.get(agent_id)
                aboard = env.human_aboard.get(agent_id)
                policy = policies[agent_id]
                if isinstance(policy, TargetDestinationHumanPolicy):
                    target = policy.target
                elif isinstance(policy, HeuristicRoutingHumanPolicy):
                    target = policy.target_nodes
                elif isinstance(policy, TraceDestinationHumanPolicy):
                    target = policy._current_target()
                else:
                    target = None
                pos_str = f"node {str(pos)}" if not isinstance(pos, tuple) else f"edge ({int(pos[0][0])}, {int(pos[0][1])})"
                aboard_str = f" (aboard {aboard})" if aboard else ""
                at_target = ""
                if target is not None and (not isinstance(pos, tuple)):
                    if isinstance(target, (int, np.integer)):
                        at_target = "✓ TARGET" if pos == target else ""
                    else:
                        at_target = "✓ TARGET" if pos in target else ""
                print(f"  {agent_id}: {pos_str}{aboard_str} -> target={target} {at_target}")

            for vehicle_id in env.vehicle_agents:
                pos = env.agent_positions.get(vehicle_id)
                policy = policies.get(vehicle_id)

                env_dest = env.vehicle_destinations.get(vehicle_id)
                policy_dest = getattr(policy, "current_destination", None)
                if hasattr(env_dest, "item"):
                    env_dest = env_dest.item()

                pos_str = f"node {str(pos)}" if not isinstance(pos, tuple) else f"edge ({int(pos[0][0])}, {int(pos[0][1])})"
                at_target = ""
                if target is not None and (not isinstance(pos, tuple)):
                    if isinstance(target, (int, np.integer)):
                        at_target = "✓ TARGET" if pos == target else ""
                    else:
                        at_target = "✓ TARGET" if pos in target else ""

                print(f"  {vehicle_id}: {pos_str} -> target={target} {at_target}")

    # Save video
    env.save_video('policies_demo.mp4', fps=5)

    # Close environment
    env.close()

    print("Complex Demo Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()