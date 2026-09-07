from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))

from accessibility_equity.policies import HeuristicRoutingHumanPolicy
from accessibility_equity.wrappers import create_dqn_env
from accessibility_equity.heuristics import TSPVehicleAgent
from accessibility_equity.rewards.efficient_equity_reward import EquityReward


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    equity_reward = EquityReward(
        cfg.env.reward.beta,
        cfg.env.reward.alpha,
        cfg.env.reward.xi,
        cfg.env.reward.eta,
        cfg.dqn.gamma,
        cfg.env.mobility.human_walking_speed_kmh,
        cfg.env.mobility.vehicle_speed_kmh,
        deduct_average_utility=cfg.env.reward.deduct_average_utility,
        reward_potential=cfg.env.reward.reward_potential,
        vehicle_advantage=cfg.env.reward.vehicle_advantage,
    )

    env = create_dqn_env(
        num_humans=cfg.env.scenario.num_humans,
        num_vehicles=cfg.env.scenario.num_vehicles,
        num_nodes=cfg.env.scenario.num_nodes,
        human_speeds=[cfg.env.mobility.human_walking_speed_kmh]
        * cfg.env.scenario.num_humans,
        vehicle_speeds=[cfg.env.mobility.vehicle_speed_kmh]
        * cfg.env.scenario.num_vehicles,
        human_policy_class=HeuristicRoutingHumanPolicy,
        seed=cfg.env.scenario.seed,
        randomize_initial_state=cfg.env.scenario.get("randomize_initial_state", False),
        network_kwargs=OmegaConf.to_container(cfg.env.network, resolve=True),
        poi_kwargs=OmegaConf.to_container(cfg.env.poi, resolve=True),
        reward_config=cfg.env.reward,
        mobility_config=cfg.env.mobility,
        max_steps=cfg.dqn.max_steps,
        reward_function=equity_reward.reward,
        decision_mode=cfg.dqn.decision_mode,
    )

    try:
        equity_reward.initialize(env.base_env)
        obs, _info = env.reset(seed=cfg.env.scenario.seed)
        tsp_agent = TSPVehicleAgent(env)

        for _ in range(1):
            action = tsp_agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"action={action}, reward={reward:.6f}")
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
