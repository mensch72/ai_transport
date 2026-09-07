"""
Minimal example: Train PPO for the accessibility-equity transport wrapper.

This file intentionally follows the structure of
`ai_transport/examples/train_ppo_example.py` as closely as possible.
"""

from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))

from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.wrappers import create_transport_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    raise SystemExit(1)


def make_env(cfg: DictConfig, seed=None):
    """Create a single environment instance."""
    env = create_transport_env(
        num_humans=cfg.env.scenario.num_humans,
        num_vehicles=cfg.env.scenario.num_vehicles,
        num_nodes=cfg.env.scenario.num_nodes,
        seed=seed,
        human_speeds=[cfg.env.mobility.human_walking_speed_kmh] * cfg.env.scenario.num_humans,
        vehicle_speeds=[cfg.env.mobility.vehicle_speed_kmh] * cfg.env.scenario.num_vehicles,
        network_kwargs=OmegaConf.to_container(cfg.env.network, resolve=True),
        poi_kwargs=OmegaConf.to_container(cfg.env.poi, resolve=True),
        reward_config=cfg.env.reward,
        mobility_config=cfg.env.mobility,
        human_policy_class=HeuristicRoutingHumanPolicy,
        human_policy_kwargs={"p_wait": 0.5},
        max_steps=cfg.dqn.max_steps,
        render_mode=None,
    )
    return env


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    """Main training loop."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_path = output_dir / "accessibility_equity_ppo_fleet"

    print("=" * 70)
    print("Training PPO for Accessibility-Equity Fleet Control")
    print(f"Output directory: {output_dir}")
    print("=" * 70)

    env = DummyVecEnv([lambda: make_env(cfg, seed=cfg.env.scenario.seed)])

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=cfg.dqn.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=cfg.dqn.total_timesteps, progress_bar=True)
    model.save(model_path)
    print(f"Model saved to '{model_path.with_suffix('.zip')}'")


if __name__ == "__main__":
    main()
