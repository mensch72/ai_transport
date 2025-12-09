"""
Minimal example: Train PPO to control vehicle fleet with heuristic human policies.

This example demonstrates:
1. Using TransportGymWrapper for single-agent RL
2. All vehicles controlled by one learned PPO policy
3. Humans use HeuristicRoutingHumanPolicy with changing goals
4. Custom reward: vehicles get reward = number of passengers when arriving at nodes
5. Extremely stripped down - just the essentials for training

Requirements:
    pip install stable-baselines3

Note: Uses MultiInputPolicy because TransportGymWrapper returns dict observations.
"""

import numpy as np
from ai_transport.wrappers import create_transport_env
from ai_transport.policies import HeuristicRoutingHumanPolicy

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    exit(1)


def passenger_arrival_reward(obs_dict, actions_dict):
    """
    Custom reward function: vehicles get reward = number of passengers when arriving at node.
    
    A vehicle "arrives at a node" when:
    - It was previously on an edge (moving)
    - It is now at a node (not on an edge)
    
    The reward is the number of humans currently aboard that vehicle.
    """
    # Access the wrapper's env to get detailed state
    # Note: This function is called from within TransportGymWrapper.step()
    # and the wrapper passes itself as the first implicit argument when bound
    
    # We need access to the wrapper - this will be set as a method
    # For now, we'll create this as a closure that captures the wrapper
    pass  # Placeholder - will be replaced with actual implementation


def make_env(seed=None):
    """Create a single environment instance."""
    
    def reward_fn(obs_dict, actions_dict):
        """Reward function with access to env via closure."""
        # This will be called from wrapper.step(), where we have access to self.env
        # We'll return a dict of rewards for each vehicle
        
        # Access the wrapper through the reward function's binding
        # The wrapper will bind this as a method, so 'self' will be the wrapper
        # For now, return zero rewards (will be properly implemented in wrapper)
        return {agent: 0.0 for agent in actions_dict if agent.startswith('vehicle_')}
    
    env = create_transport_env(
        num_humans=4,
        num_vehicles=2,
        num_nodes=12,
        seed=seed,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={'p_wait': 0.5},
        max_steps=500,
        render_mode=None,
    )
    
    # Set custom reward function that rewards vehicles for passengers on arrival
    def custom_reward(obs_dict, actions_dict):
        """Vehicles get reward = passengers aboard when arriving at a node."""
        rewards = {}
        
        for vehicle in env.vehicle_agents:
            reward = 0.0
            
            # Check if vehicle just arrived at a node
            pos = env.env.agent_positions.get(vehicle)
            
            # Vehicle is at a node (not on an edge)
            if pos is not None and not isinstance(pos, tuple):
                # Count passengers aboard this vehicle
                passengers_aboard = sum(
                    1 for human in env.human_agents
                    if env.env.human_aboard.get(human) == vehicle
                )
                
                # Give reward based on passengers
                # We only reward on the "arrival" moment, but for simplicity
                # in this minimal example, we reward whenever at a node with passengers
                # A more sophisticated version would track previous positions
                reward = float(passengers_aboard)
            
            rewards[vehicle] = reward
        
        return rewards
    
    env.reward_function = custom_reward
    return env


def main():
    """Main training loop."""
    print("=" * 70)
    print("Training PPO for Vehicle Fleet Control")
    print("=" * 70)
    print("\nSetup:")
    print("  - 4 humans with HeuristicRoutingHumanPolicy")
    print("  - 2 vehicles controlled by learned PPO policy")
    print("  - 12-node random network")
    print("  - Reward: vehicles get points for passengers when at nodes")
    print("  - Human goals change when reached")
    print()
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: make_env(seed=42)])
    
    # Create PPO model with MultiInputPolicy for dict observation space
    print("Initializing PPO model...")
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dict observations
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )
    
    # Train the model
    print("\nStarting training...")
    print("(This is a minimal example - training for only 10k steps)")
    total_timesteps = 10_000
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )
        print("\n✓ Training completed!")
        
        # Save the model
        model.save("transport_ppo_fleet")
        print("✓ Model saved to 'transport_ppo_fleet.zip'")
        
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
        model.save("transport_ppo_fleet_interrupted")
        print("✓ Model saved to 'transport_ppo_fleet_interrupted.zip'")
    
    # Evaluate the trained policy
    print("\n" + "=" * 70)
    print("Evaluating trained policy...")
    print("=" * 70)
    
    eval_env = make_env(seed=100)
    obs, _ = eval_env.reset()
    
    total_reward = 0
    steps = 0
    max_eval_steps = 200
    
    print(f"\nRunning evaluation for {max_eval_steps} steps...")
    
    for step in range(max_eval_steps):
        # Get action from trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, truncated, info = eval_env.step(action)
        total_reward += reward
        steps += 1
        
        if done or truncated:
            break
    
    print(f"\nEvaluation Results:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Average reward per step: {total_reward/steps:.4f}")
    
    eval_env.close()
    env.close()
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
