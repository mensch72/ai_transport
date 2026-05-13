"""
Inspect DQN vehicle action masks phase by phase.

This script does not train a model. It prints the wrapper-level unified vehicle
actions and the underlying vehicle action that will be sent to the base
environment, so the masking design can be checked by eye.
"""

from __future__ import annotations

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG
from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.wrappers import create_dqn_env


def _position_to_text(position):
    if isinstance(position, tuple):
        edge, coord = position
        return f"edge={edge}, coord={float(coord):.2f}"
    return f"node={position}"


def _describe_action(env, action):
    if action == 0:
        return "0: pass/stay/no target"

    node = env.node_order[action - 1]
    step_type = env.base_env.env.step_type
    if step_type == "routing":
        return f"{action}: set destination to node {node}"
    if step_type == "departing":
        return f"{action}: depart to neighbor node {node}"
    return f"{action}: node {node} masked in {step_type}"


def _print_mask_state(env, label):
    base = env.base_env.env
    position = base.agent_positions.get(env.vehicle_agent)
    mask = env.action_masks()
    legal_actions = [index for index, is_valid in enumerate(mask) if is_valid]

    print("=" * 80)
    print(label)
    print(f"step_type: {base.step_type}")
    print(f"vehicle: {env.vehicle_agent}")
    print(f"vehicle_position: {_position_to_text(position)}")
    print(f"vehicle_destination: {base.vehicle_destinations.get(env.vehicle_agent)}")
    print(f"raw mask: {mask.astype(int).tolist()}")
    print("legal wrapper actions:")
    for action in legal_actions:
        underlying = env._masked_node_action_to_underlying_action(action)
        print(f"  {_describe_action(env, action)} -> underlying vehicle action {underlying}")


def _choose_demo_action(env):
    """Pick a deterministic legal action that advances the phase clearly."""
    base = env.base_env.env
    mask = env.action_masks()

    if base.step_type == "routing":
        legal_nodes = [action for action in range(1, env.action_space.n) if mask[action]]
        return legal_nodes[0] if legal_nodes else 0

    if base.step_type == "departing":
        legal_nodes = [action for action in range(1, env.action_space.n) if mask[action]]
        return legal_nodes[0] if legal_nodes else 0

    return 0


def main():
    config = DEFAULT_EXPERIMENT_CONFIG
    env = create_dqn_env(
        num_humans=config.scenario.num_humans,
        num_vehicles=config.scenario.num_vehicles,
        num_nodes=config.scenario.num_nodes,
        seed=config.scenario.seed,
        human_speeds=[config.mobility.human_walking_speed_kmh] * config.scenario.num_humans,
        vehicle_speeds=[config.mobility.vehicle_speed_kmh] * config.scenario.num_vehicles,
        network_kwargs={"speed_mean": config.mobility.edge_speed_kmh},
        poi_kwargs=config.scenario.poi_kwargs,
        reward_config=config.reward,
        mobility_config=config.mobility,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={"p_wait": 0.5},
        max_steps=config.dqn.max_steps,
        use_action_masking=True,
        decision_mode=config.dqn.decision_mode,
    )

    try:
        _obs, info = env.reset(seed=config.scenario.seed)
        print("Initial info action_mask:", info["action_mask"].astype(int).tolist())

        for step_index in range(12):
            _print_mask_state(env, f"Before wrapper step {step_index + 1}")
            action = _choose_demo_action(env)
            _obs, reward, terminated, truncated, info = env.step(action)
            print(f"selected wrapper action: {action}")
            print(f"action was valid: {info['dqn_action_was_valid']}")
            print(f"underlying vehicle action: {info['dqn_underlying_vehicle_action']}")
            print(f"raw_reward: {info['raw_reward']:.6f}")
            print()

            if terminated or truncated:
                print(f"Episode ended: terminated={terminated}, truncated={truncated}")
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
