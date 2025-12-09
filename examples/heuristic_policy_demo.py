"""
Demonstration of HeuristicRoutingHumanPolicy for AI Transport environment.

Shows how humans use the heuristic policy to board vehicles strategically
to get closer to their target destinations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_transport import parallel_env
from ai_transport.policies import (
    HeuristicRoutingHumanPolicy,
    RandomVehiclePolicy,
    ShortestPathVehiclePolicy
)

def main():
    print("="*70)
    print("AI Transport Environment - Heuristic Routing Human Policy Demo")
    print("="*70)
    print()
    
    # Create environment with random 2D network
    print("1. Creating environment with random network...")
    env = parallel_env(num_humans=3, num_vehicles=2, observation_scenario='full')
    network = env.create_random_2d_network(num_nodes=12, bidirectional_prob=0.4, seed=42)
    env = parallel_env(
        num_humans=3,
        num_vehicles=2,
        network=network,
        observation_scenario='full',
        render_mode='human'
    )
    
    obs, info = env.reset(seed=42)
    print(f"   Network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"   Agents: 3 humans, 2 vehicles")
    print()
    
    # Create policies for agents
    print("2. Creating policies...")
    policies = {}
    
    # Get some target nodes
    all_nodes = list(network.nodes())
    target_1 = {all_nodes[-1]}  # Last node
    target_2 = {all_nodes[-2], all_nodes[-3]}  # Two target nodes
    target_3 = {all_nodes[len(all_nodes)//2]}  # Middle node
    
    # Human policies with heuristic routing
    policies['human_0'] = HeuristicRoutingHumanPolicy(
        'human_0', network, target_nodes=target_1, p_wait=0.3, seed=1
    )
    policies['human_1'] = HeuristicRoutingHumanPolicy(
        'human_1', network, target_nodes=target_2, p_wait=0.6, seed=2
    )
    policies['human_2'] = HeuristicRoutingHumanPolicy(
        'human_2', network, target_nodes=target_3, p_wait=0.4, seed=3
    )
    
    # Vehicle policies
    policies['vehicle_0'] = RandomVehiclePolicy('vehicle_0', pass_prob_routing=0.2, seed=5)
    policies['vehicle_1'] = ShortestPathVehiclePolicy('vehicle_1', network, seed=6)
    
    print("   Policies:")
    print(f"   - human_0: HeuristicRoutingHumanPolicy (target={target_1}, p_wait=0.3)")
    print(f"   - human_1: HeuristicRoutingHumanPolicy (target={target_2}, p_wait=0.6)")
    print(f"   - human_2: HeuristicRoutingHumanPolicy (target={target_3}, p_wait=0.4)")
    print("   - vehicle_0: RandomVehiclePolicy (pass_prob_routing=0.2)")
    print("   - vehicle_1: ShortestPathVehiclePolicy")
    print()
    
    # Run simulation
    print("3. Running simulation with heuristic routing policy...")
    print()
    
    for step in range(40):
        # Print summary every 4 steps (one full cycle)
        if step % 4 == 0:
            print(f"   Step {step}:")
            print(f"   - Step type: {env.step_type}")
            print(f"   - Real time: {env.real_time:.2f}s")
            print(f"   - Active agents: {len(env.agents)}")
            
            # Show positions and targets
            for agent_id in ['human_0', 'human_1', 'human_2']:
                policy = policies.get(agent_id)
                pos = env.agent_positions.get(agent_id)
                aboard = env.human_aboard.get(agent_id)
                
                if isinstance(policy, HeuristicRoutingHumanPolicy):
                    target = policy.target_nodes
                    pos_str = str(pos) if not isinstance(pos, tuple) else f"on edge {pos[0]}"
                    aboard_str = f" (aboard {aboard})" if aboard else ""
                    print(f"   - {agent_id}: at {pos_str}{aboard_str}, target={target}")
            
            # Show vehicle destinations
            for agent_id in ['vehicle_0', 'vehicle_1']:
                dest = env.vehicle_destinations.get(agent_id)
                pos = env.agent_positions.get(agent_id)
                pos_str = str(pos) if not isinstance(pos, tuple) else f"on edge {pos[0]}"
                print(f"   - {agent_id}: at {pos_str}, destination={dest}")
            
            print()
        
        # Get actions from policies
        actions = {}
        action_justifications = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                action, justification = policy.get_action(obs[agent], action_space_size)
                actions[agent] = action
                action_justifications[agent] = justification
            else:
                actions[agent] = 0  # Pass
                action_justifications[agent] = "Passing (no policy)"
        
        # Print actions with justifications for boarding and departing steps
        if env.step_type in ['boarding', 'departing']:
            print(f"   {env.step_type.capitalize()} actions:")
            for agent in env.agents:
                action_idx = actions[agent]
                justification = action_justifications[agent]
                if action_idx != 0 or 'Boarding' in justification or 'Walking' in justification or 'Waiting' in justification:
                    print(f"   - {agent}: action {action_idx} - {justification}")
            print()
        
        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print("="*70)
    print("Heuristic Routing Human Policy demonstration complete!")
    print("="*70)
    print()
    print("Policy Features Demonstrated:")
    print()
    print("HeuristicRoutingHumanPolicy:")
    print("- Humans have a set of target nodes they want to reach")
    print("- At boarding step, humans consider all vehicles at their current node")
    print("- They compute shortest duration paths for vehicles to their destinations")
    print("- They identify nodes on these paths that are closer to their targets")
    print("- They board the vehicle that gets them to the best intermediate node fastest")
    print("- At departing step, if no good vehicle was available:")
    print("  * With probability p_wait, they wait for more vehicles")
    print("  * With probability (1 - p_wait), they walk toward their target")
    print()
    print("This demonstrates intelligent decision-making where humans:")
    print("- Use vehicles strategically to get closer to their goals")
    print("- Balance between waiting for better options vs. making progress on foot")
    print("- Consider multiple potential targets when making routing decisions")
    print()

if __name__ == "__main__":
    main()
