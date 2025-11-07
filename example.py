"""Example script demonstrating the AI Transport parallel environment."""

import numpy as np
from ai_transport import parallel_env


def main():
    """Run a simple example of the AI Transport environment."""
    print("Creating AI Transport Parallel Environment...")
    
    # Create the environment
    env = parallel_env(
        num_buses=3,
        num_stops=10,
        max_passengers_per_vehicle=50,
        max_passengers_per_stop=20,
        render_mode="human"
    )
    
    print(f"Possible agents: {env.possible_agents}")
    print(f"Number of stops: {env.num_stops}")
    print(f"Max passengers per vehicle: {env.max_passengers_per_vehicle}")
    
    # Reset the environment
    observations, infos = env.reset(seed=42)
    print(f"\nInitial observations:")
    for agent, obs in observations.items():
        print(f"  {agent}: {obs}")
    
    # Run a few steps
    num_steps = 10
    print(f"\nRunning {num_steps} steps...")
    
    for step in range(num_steps):
        # Sample random actions for all agents
        actions = {
            agent: env.action_space(agent).sample()
            for agent in env.agents
        }
        
        # Execute the step
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render the environment
        env.render()
        
        # Print rewards
        print(f"Rewards: {rewards}")
        
        # Check if episode is done
        if any(terminations.values()) or any(truncations.values()):
            print("\nEpisode finished!")
            break
    
    # Close the environment
    env.close()
    print("\nEnvironment closed successfully!")


if __name__ == "__main__":
    main()
