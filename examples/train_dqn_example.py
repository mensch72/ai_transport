"""
Minimal example: Train DQN for single-vehicle accessibility-equity control.

This example intentionally starts with one vehicle because standard DQN expects
a single discrete action. Multi-vehicle DQN needs a separate action design.
"""

from pathlib import Path
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import make_env, plot_training_rewards, save_training_scenario_diagnostics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.wrappers import REWARD_SCALE
from accessibility_equity.rewards.efficient_equity_reward import EquityReward

try:
    from accessibility_equity.algorithms import MaskedDQN as DQN
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    raise SystemExit(1)


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    """
    Run a small DQN training job.
    """
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    model_dir = output_dir / "models"
    train_monitor_path = output_dir / "train.monitor.csv"
    train_curve_path = output_dir / "training_reward_curve.png"
    model_path = model_dir / "accessibility_equity_dqn_single_vehicle"

    print("=" * 70)
    print("Training DQN for Single-Vehicle Accessibility-Equity Control")
    print(f"Decision mode: {cfg.dqn.decision_mode}")
    print(
        f"Scenario: humans={cfg.env.scenario.num_humans}, vehicles={cfg.env.scenario.num_vehicles}, nodes={cfg.env.scenario.num_nodes}"
    )
    print(
        "Scenario config: "
        f"seed={cfg.env.scenario.seed}, "
        f"central_count={cfg.env.scenario.central_count}"
    )
    print(
        "Mobility config: "
        f"human={cfg.env.mobility.human_walking_speed_kmh:g} km/h, "
        f"vehicle={cfg.env.mobility.vehicle_speed_kmh:g} km/h, "
        f"edge={cfg.env.mobility.edge_speed_kmh:g} km/h"
    )
    print(
        "Reward config: "
        f"utility_mode={cfg.env.reward.utility_mode}, "
        f"beta={cfg.env.reward.beta:g}, alpha={cfg.env.reward.alpha:g}, "
        f"xi={cfg.env.reward.xi:g}, eta={cfg.env.reward.eta:g}, "
        f"normalize_utility={cfg.env.reward.normalize_utility}, "
        f"clip_normalized_utility={cfg.env.reward.clip_normalized_utility}, "
        f"normalized_reward_scale={cfg.env.reward.normalized_reward_scale:g}"
    )
    print("DQN output config: " f"save_model={cfg.dqn.save_model}, ")
    print(f"Output directory: {output_dir}")
    print(f"NOTE: DQN rewards are scaled by REWARD_SCALE={REWARD_SCALE:g}.")
    print("      Monitor curves and scaled_reward logs are not raw rewards.")
    print("=" * 70)

    if train_monitor_path.exists():
        train_monitor_path.unlink()

    equity_reward = EquityReward(
        cfg.env.reward.beta,
        cfg.env.reward.alpha,
        cfg.env.reward.xi,
        cfg.env.reward.eta,
        cfg.dqn.gamma,
        cfg.env.mobility.human_walking_speed_kmh,
        cfg.env.mobility.vehicle_speed_kmh,
    )

    env = make_env(
        cfg,
        output_dir=output_dir,
        seed=cfg.env.scenario.seed,
        monitor=True,
        render_mode=None,
        reward_function=equity_reward.reward,
    )

    save_training_scenario_diagnostics(env.env, output_dir)

    equity_reward.initialize(
        env.env.base_env.env.network,
        env.env.base_env.vehicle_agents,
        env.env.base_env.human_agents,
    )

    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=cfg.dqn.learning_rate,
        buffer_size=cfg.dqn.buffer_size,
        learning_starts=cfg.dqn.learning_starts,
        batch_size=cfg.dqn.batch_size,
        gamma=cfg.dqn.gamma,
        train_freq=cfg.dqn.train_freq,
        target_update_interval=cfg.dqn.target_update_interval,
        exploration_fraction=cfg.dqn.exploration_fraction,
        exploration_final_eps=cfg.dqn.exploration_final_eps,
    )

    model.learn(total_timesteps=cfg.dqn.total_timesteps, progress_bar=True)
    if cfg.dqn.save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    env.close()
    plot_training_rewards(cfg, train_monitor_path, train_curve_path)
    if train_monitor_path.exists():
        train_monitor_path.unlink()
    if cfg.dqn.save_model:
        print(f"Model saved to '{model_path.with_suffix('.zip')}'")
    else:
        print("Model file not saved (SAVE_MODEL=False).")


if __name__ == "__main__":
    main()