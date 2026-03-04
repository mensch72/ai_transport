from ai_transport import parallel_env
import networkx as nx
import sys


def setup_demo_state(env, step_type: str):
    """
    Set up environment state to demonstrate each step type.

    In real gameplay, agent positions are random and hard to predict.
    To test action_space logic, we need CONTROLLED, IDEAL states where:
    - All agents are in the correct position for this step type
    - We can see the COMPLETE action space (not limited by random position)

    """
    if step_type == 'routing':
        # All agents at node 0, not aboard
        # This allows us to see:
        #   - humans: action_space.n = 1 (only pass)
        #   - vehicles: action_space.n = num_nodes + 1 (can set destination)
        for agent in env.vehicle_agents:
            env.agent_positions[agent] = 0
        for agent in env.human_agents:
            env.agent_positions[agent] = 0
            env.human_aboard[agent] = None

    elif step_type == 'unboarding':
        # One human aboard a vehicle at node, others not aboard
        # This allows us to see:
        #   - human aboard: action_space.n = 2 (pass or unboard)
        #   - humans not aboard: action_space.n = 1 (only pass)
        #   - vehicles: action_space.n = 1 (only pass)
        for agent in env.agent_positions:
            env.agent_positions[agent] = 0
        if env.human_agents and env.vehicle_agents:
            first_human = sorted(env.human_agents)[0]
            first_vehicle = sorted(env.vehicle_agents)[0]
            env.human_aboard[first_human] = first_vehicle
            for h in env.human_agents:
                if h != first_human:
                    env.human_aboard[h] = None

    elif step_type == 'boarding':
        # All agents at node, humans not aboard
        # This allows us to see:
        #   - humans: action_space.n = num_vehicles + 1 (can board any vehicle)
        #   - vehicles: action_space.n = 1 (only pass)
        for agent in env.agent_positions:
            env.agent_positions[agent] = 0
        for human in env.human_agents:
            env.human_aboard[human] = None

    elif step_type == 'departing':
        # All agents at node; one human aboard vehicle
        # This allows us to see:
        #   - human aboard: action_space.n = 1 (only pass)
        #   - humans not aboard: action_space.n = num_outgoing_edges + 1 (can walk)
        #   - vehicles: action_space.n = num_outgoing_edges + 1 (can depart)
        for agent in env.agent_positions:
            env.agent_positions[agent] = 0
        if env.human_agents and env.vehicle_agents:
            first_human = sorted(env.human_agents)[0]
            first_vehicle = sorted(env.vehicle_agents)[0]
            env.human_aboard[first_human] = first_vehicle
            for h in env.human_agents:
                if h != first_human:
                    env.human_aboard[h] = None


def validate_action_space(env, agent: str, has_error_list: list):
    """
    Validate a single agent's action space for consistency.

    Three checks:Size consistency, Mapping completeness, Sampling works

    Returns mapping if valid, or None if errors detected.
    """
    space = env.action_space(agent)
    mapping = env._get_action_mapping(agent)
    desc = mapping.get('description', {})
    details = mapping.get('details', {})

    # Get action space size
    try:
        space_n = getattr(space, 'n', None)
    except Exception:
        space_n = None

    print(f"    {agent}: action_space.n={space_n}")
    print(f"      mapping['description']: {desc}")
    print(f"      mapping['details']: {details}")

    # Check 1: mapping length vs action space size
    if space_n is not None and len(desc) != space_n:
        print(f"      ⚠ ERROR: action_space.n={space_n} but mapping has {len(desc)} description entries")
        has_error_list.append(True)

    # Check 2: each description entry should have a details entry
    missing_details = [k for k in desc.keys() if k not in details]
    if missing_details:
        print(f"      ⚠ ERROR: mapping `details` missing keys: {missing_details}")
        has_error_list.append(True)

    # Check 3: sample from action space (verify it works)
    try:
        sample = space.sample()
        sample_desc = desc.get(sample, '<no description>')
        sample_detail = details.get(sample, '<no detail>')
        print(f"      sample test: action={sample} (desc='{sample_desc}', detail={sample_detail})")
    except Exception as e:
        print(f"      ⚠ ERROR: failed to sample action_space: {e}")
        has_error_list.append(True)

    return mapping


def main(seed: int = 234):
    """
    Lightweight validator for dynamic action spaces.

    - Uses the environment's own `action_space()` and `_get_action_mapping()` methods
    - Sets up realistic state for each step type (following action_spaces_demo.py patterns)
    - Performs consistency checks:
      * action_space.n equals number of mapping entries
      * every mapping key has a detail entry
      * sampling from the action space works
    """
    print("=" * 70)
    print("AI Transport Action Space Validation Demo")
    print("=" * 70)

    # Create environment with custom network (like action_spaces_demo.py)
    G = nx.DiGraph()
    G.add_node(0, name="Station_A")
    G.add_node(1, name="Station_B")
    G.add_node(2, name="Station_C")

    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=8)
    G.add_edge(2, 0, length=12.0, speed=6.0, capacity=12)

    env = parallel_env(num_humans=2, num_vehicles=2, network=G)

    obs, infos = env.reset(seed=seed)

    print(f"\nEnvironment:")
    print(f"  - {env.num_humans} humans, {env.num_vehicles} vehicles")
    print(f"  - Network: {len(env.network.nodes())} nodes, {len(env.network.edges())} edges")
    print(f"  - Initial positions: {env.agent_positions}")

    has_errors = []

    # Validate each step type
    step_types = ['routing', 'unboarding', 'boarding', 'departing']

    for step_idx, step_type in enumerate(step_types):
        print(f"\n{'-' * 70}")
        print(f"Step Type {step_idx + 1}/4: {step_type.upper()}")
        print(f"{'-' * 70}")

        # Set up state for this step type
        setup_demo_state(env, step_type)

        # Manually set step type (for validation purposes only - normally set by step())
        env._set_step_type_for_testing(step_type)

        # Show current state
        print(f"\nState setup:")
        print(f"  Positions: {env.agent_positions}")
        print(f"  Aboard: {env.human_aboard}")
        if step_type == 'departing':
            outgoing = list(env.network.out_edges(0))
            print(f"  Outgoing edges from node 0: {outgoing}")

        print(f"\nAction space validation:")
        for agent in sorted(env.agents):
            validate_action_space(env, agent, has_errors)

        # Step with all-pass actions to advance to next step type naturally
        actions = {agent: 0 for agent in env.agents}
        try:
            obs, rewards, terminations, truncations, infos = env.step(actions)
        except Exception as e:
            print(f"\n⚠ ERROR during env.step(): {e}")
            has_errors.append(True)

    print(f"\n{'=' * 70}")
    if has_errors:
        print("✗ Validation FAILED: issues were detected. See warnings above.")
        sys.exit(1)
    else:
        print("✓ Validation PASSED: no issues found.")
        sys.exit(0)


if __name__ == '__main__':
    main()
