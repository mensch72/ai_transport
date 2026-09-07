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
from accessibility_equity.wrappers import create_dqn_env
from accessibility_equity.heuristics import TSPVehicleAgent, GoToHumanVehicleAgent


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    video_path = output_dir / "heuristic_demo.mp4"

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
        render_mode="human",
        decision_mode=cfg.dqn.decision_mode,
    )

    try:
        env.base_env.env.start_video_recording()
        obs, _info = env.reset(seed=cfg.env.scenario.seed)
        env.base_env.env.enable_rendering("graphical")

        gth_agent = GoToHumanVehicleAgent(env)
        _tsp_agent = TSPVehicleAgent(env)

        for _ in range(100):
            action = gth_agent.get_action(obs)
            obs, _reward, terminated, truncated, _info = env.step(action)
            env.base_env.env.render()
            if terminated or truncated:
                break

        env.base_env.env.save_video(str(video_path), fps=5)
        print(f"Heuristic video saved to '{video_path}'")
    finally:
        env.close()


if __name__ == "__main__":
    main()
