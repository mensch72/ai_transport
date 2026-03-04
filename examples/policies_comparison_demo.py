# python
"""
policies_demo.py - three focused demos
Each demo pairs one human policy with ShortestPathVehiclePolicy (vehicles)
to inspect cooperation. Demos save a short video and print simple event logs.
"""
import sys
import networkx as nx
import numpy as np
from ai_transport.envs import parallel_env
import ai_transport.policies as at_policies  # avoid repeating the original import line

def build_human_target_config(human_agents, candidate_nodes, rng, trajectory_length=5):
    config = {}
    for agent in human_agents:
        trajectory = [int(rng.choice(candidate_nodes)) for _ in range(trajectory_length)]
        target = int(rng.choice(trajectory))
        target_nodes = set(trajectory)
        config[agent] = {
            "target": target,
            "target_nodes": target_nodes,
            "target_trajectory": trajectory
        }
    return config

def run_demo(demo_name, human_policy_type, seed=42, cycles=80):
    rng = np.random.default_rng(seed)
    print(f"\n--- Demo: {demo_name} ({human_policy_type}) ---")
    # Create network
    tmp_env = parallel_env(num_humans=4, num_vehicles=3, observation_scenario="full", render_mode="human")
    G = tmp_env.create_random_2d_network(num_nodes=10, bidirectional_prob=0.85,
                                        speed_mean=5.0, capacity_mean=10.0,
                                        coord_mean=0.0, coord_std=10.0, seed=seed)
    tmp_env.close()

    # Create environment
    env = parallel_env(
        num_humans=4,
        num_vehicles=3,
        network=G,
        observation_scenario='full',
        render_mode="human"
    )
    env.reset(seed=seed)

    # Prepare human targets
    start_node = 0
    candidate_nodes = [n for n in G.nodes() if n != start_node]
    human_agents = [a for a in env.agents if a in env.human_agents]
    human_dests = build_human_target_config(human_agents, candidate_nodes, rng)

    # Build policies: all humans use same policy type; vehicles use shortest path
    policies = {}
    vehicle_wait_cycles = 2
    human_wait = 0.7
    human_target_change_rate = 0.01

    for agent in env.agents:
        env.agent_positions[agent] = start_node
        if agent in env.human_agents:
            env.human_aboard[agent] = None
            cfg = human_dests[agent]
            if human_policy_type == "target":
                policies[agent] = at_policies.TargetDestinationHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_change_rate=human_target_change_rate,
                    seed=seed
                )
            elif human_policy_type == "heuristic":
                policies[agent] = at_policies.HeuristicRoutingHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_nodes=cfg["target_nodes"],
                    p_wait=human_wait,
                    seed=seed
                )
            elif human_policy_type == "trace":
                policies[agent] = at_policies.TraceDestinationHumanPolicy(
                    agent_id=agent,
                    network=G,
                    target_trajectory=cfg["target_trajectory"],
                    p_wait=human_wait,
                    seed=seed
                )
            else:
                raise ValueError("unknown human_policy_type")
        else:
            policies[agent] = at_policies.ShortestPathVehiclePolicy(
                agent_id=agent,
                network=G,
                wait_cycles=vehicle_wait_cycles,
                seed=seed
            )

    # Initialize env destinations from policy state so human/vehicle goals are visible in env
    for agent, policy in policies.items():
        if agent in env.human_agents:
            if isinstance(policy, at_policies.TargetDestinationHumanPolicy):
                env.human_destinations[agent] = policy.target
            elif isinstance(policy, at_policies.HeuristicRoutingHumanPolicy):
                env.human_destinations[agent] = set(policy.target_nodes)
            elif isinstance(policy, at_policies.TraceDestinationHumanPolicy):
                env.human_destinations[agent] = policy._current_target()
        else:
            if hasattr(policy, "current_destination"):
                env.vehicle_destinations[agent] = policy.current_destination

    # Prepare rendering / recording
    env.enable_rendering('graphical')
    env.start_video_recording()
    env.render()

    # initial observations
    obs = {agent: env._generate_observation_for_agent(agent) for agent in env.agents}

    boarding_events = []
    unboarding_events = []
    walking_events = []

    env.termination_mode = "current_active"
    env.freeze_termination_participants()

    for cycle in range(cycles):
        current_step = env.step_type
        actions = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                result = policy.get_action(obs[agent], action_space_size)
                if result is None:
                    actions[agent] = 0
                    continue
                if isinstance(result, tuple) and len(result) == 2:
                    action, justification = result
                else:
                    action = result
                    justification = "No justification"

                actions[agent] = action

                if current_step == 'boarding' and agent in env.human_agents and action > 0:
                    vehicle_id = obs[agent].get('action_mapping', {}).get('details', {}).get(action)
                    boarding_events.append((cycle, agent, vehicle_id))
                if current_step == 'unboarding' and agent in env.human_agents and action > 0:
                    aboard = env.human_aboard.get(agent)
                    unboarding_events.append((cycle, agent, aboard))
                if current_step == 'departing' and agent in env.human_agents and action > 0:
                    if env.human_aboard.get(agent) is None:
                        walking_events.append((cycle, agent))
            else:
                actions[agent] = 0

        obs, rewards, terms, truncs, infos = env.step(actions)
        active = [v for v in terms.values() if v is not None]
        should_term = (len(active) > 0) and all(active)
        if should_term:
            break

        if env.step_type == 'routing':
            env.render()

    # Save, close and summarize
    outname = f"example_outputs/{demo_name}.mp4"
    env.save_video(outname, fps=5)
    env.close()

    print(f"Demo {demo_name} finished. Summary:")
    print(f"  Boarding events: {len(boarding_events)} (first 5: {boarding_events[:5]})")
    print(f"  Unboarding events: {len(unboarding_events)} (first 5: {unboarding_events[:5]})")
    print(f"  Walking events: {len(walking_events)} (first 5: {walking_events[:5]})")
    print(f"  Video saved to `{outname}`")

def main():
    run_demo("Target_H + ShortestPath_V", "target")
    run_demo("Heuristic_H + ShortestPath_V", "heuristic")
    run_demo("Trace_H + ShortestPath_V", "trace")
    print("\nAll demos complete.")

if __name__ == "__main__":
    main()
