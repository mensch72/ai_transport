"""
Minimal example: Train PPO for the accessibility-equity transport wrapper.

This file intentionally follows the structure of
`ai_transport/examples/train_ppo_example.py` as closely as possible.
"""

from accessibility_equity.config import DEFAULT_EXPERIMENT_CONFIG
from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.wrappers import create_transport_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    raise SystemExit(1)


EXPERIMENT_CONFIG = DEFAULT_EXPERIMENT_CONFIG


def make_env(seed=None):
    """Create a single environment instance."""
    config = EXPERIMENT_CONFIG
    env = create_transport_env(
        num_humans=config.scenario.num_humans,
        num_vehicles=config.scenario.num_vehicles,
        num_nodes=config.scenario.num_nodes,
        seed=seed,
        human_speeds=[config.mobility.human_walking_speed_kmh] * config.scenario.num_humans,
        vehicle_speeds=[config.mobility.vehicle_speed_kmh] * config.scenario.num_vehicles,
        network_kwargs={"speed_mean": config.mobility.edge_speed_kmh},
        poi_kwargs=config.scenario.poi_kwargs,
        reward_config=config.reward,
        mobility_config=config.mobility,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={"p_wait": 0.5},
        max_steps=config.dqn.max_steps,
        render_mode=None,
    )
    return env


def main():
    """Main training loop."""
    print("=" * 70)
    print("Training PPO for Accessibility-Equity Fleet Control")
    print("=" * 70)

    env = DummyVecEnv([lambda: make_env(seed=EXPERIMENT_CONFIG.scenario.seed)])

    model = PPO(
        "MultiInputPolicy",
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

    total_timesteps = EXPERIMENT_CONFIG.dqn.total_timesteps
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    model.save("accessibility_equity_ppo_fleet")
    print("Model saved to 'accessibility_equity_ppo_fleet.zip'")


if __name__ == "__main__":
    main()
