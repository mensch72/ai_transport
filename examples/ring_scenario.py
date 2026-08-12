"""
Ring example: Train DQN for single-vehicle accessibility-equity control on a
symmetric ring network.

The scenario is a ring of ``N`` nodes placed on a circle so that every pair of
neighboring nodes is the same distance apart. Each node hosts exactly one point
of interest, all of the same type. One human and one vehicle start on opposing
sides of the ring.

This example uses one vehicle because standard DQN expects a single discrete
action. Multi-vehicle DQN needs a separate action design.
"""

import math
from pathlib import Path
from random import Random
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from omegaconf import open_dict

import matplotlib
import networkx as nx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import make_env, plot_training_rewards, save_training_scenario_diagnostics
from evaluation import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.scenarios import TransportScenario
from accessibility_equity.wrappers import REWARD_SCALE
from accessibility_equity.rewards.efficient_equity_reward import EquityReward
from accessibility_equity.policies.human_policies import RandomHumanPolicy

try:
    from accessibility_equity.algorithms import MaskedDQN as DQN
except ImportError:
    print("ERROR: stable-baselines3 not installed.")
    print("Install with: pip install stable-baselines3")
    raise SystemExit(1)


def build_ring_scenario(
    num_nodes: int,
    poi_type: str = "workplace",
    poi_weight: float = 0.40,
    radius_per_node: float = 1.0,
    edge_speed_kmh: float = 30.0,
    edge_capacity: float = 10.0,
) -> TransportScenario:
    """
    Build a symmetric ring scenario.

    - The network is a ring of ``num_nodes`` nodes evenly spaced on a circle, so
      every pair of neighboring nodes is the same Euclidean distance apart. Each
      neighbor pair is connected by edges in both directions.
    - Each node carries exactly one point of interest, all of the same type.
    - One human and one vehicle start on opposing sides of the ring.

    Node labels are the integers ``0..num_nodes-1`` in ring order. The reward and
    routing code rely on this ordering, since the shortest-path matrices index
    nodes by their position in ``graph.nodes``.
    """
    if num_nodes < 3:
        raise ValueError("A ring scenario needs at least 3 nodes.")

    radius = radius_per_node * num_nodes
    graph = nx.DiGraph()
    for node in range(num_nodes):
        angle = 2.0 * math.pi * node / num_nodes
        graph.add_node(
            node,
            name=f"Node_{node}",
            x=float(radius * math.cos(angle)),
            y=float(radius * math.sin(angle)),
        )

    # The chord length between adjacent vertices of a regular polygon is the
    # same for every neighbor pair, which gives equal distance between all
    # neighbors regardless of where they sit on the ring.
    edge_length = float(2.0 * radius * math.sin(math.pi / num_nodes))
    speed = max(float(edge_speed_kmh), 0.1)
    capacity = max(float(edge_capacity), 1.0)
    for node in range(num_nodes):
        neighbor = (node + 1) % num_nodes
        graph.add_edge(
            node, neighbor, length=edge_length, speed=speed, capacity=capacity
        )
        graph.add_edge(
            neighbor, node, length=edge_length, speed=speed, capacity=capacity
        )

    # Exactly one POI per node, all of the same type.
    poi_records = []
    for node in range(num_nodes):
        poi_records.append(
            {
                "poi_id": f"{poi_type}_{node}",
                "poi_type": poi_type,
                "node": node,
                "nearest_node": node,
                "area_node": node,
                "x": graph.nodes[node]["x"],
                "y": graph.nodes[node]["y"],
                "poi_weight": float(poi_weight),
            }
        )
        graph.nodes[node]["poi_type"] = poi_type
        graph.nodes[node]["poi_types"] = [poi_type]
        graph.nodes[node]["poi_type_counts"] = {poi_type: 1}
        graph.nodes[node]["poi_weight"] = float(poi_weight)
        graph.nodes[node]["area_type"] = "mixed_poi"
    graph.graph["poi_records"] = poi_records
    graph.graph["poi_nodes_by_type"] = {poi_type: list(range(num_nodes))}

    poi_distribution = {
        "poi_records": poi_records,
        "poi_nodes_by_type": {poi_type: list(range(num_nodes))},
        "poi_weights": {poi_type: float(poi_weight)},
        "residential_records": None,
    }

    # Place the human and the vehicle on opposing sides of the ring. With evenly
    # spaced nodes, node ``num_nodes // 2`` sits opposite node ``0``.
    human_node = 0
    vehicle_node = num_nodes // 2
    human_goal_node = vehicle_node

    human_distribution = {
        "human_to_initial_node": {"human_0": human_node},
        "human_to_home_node": {"human_0": human_node},
        "human_to_target_node": {"human_0": human_goal_node},
        "human_goal_nodes": [human_goal_node],
        "human_records": {
            "human_0": {
                "home_node": human_node,
                "initial_node": human_node,
                "target_node": human_goal_node,
                "target_poi": None,
            }
        },
    }
    vehicle_distribution = {
        "vehicle_to_initial_node": {"vehicle_0": vehicle_node},
        "vehicle_records": {"vehicle_0": {"initial_node": vehicle_node}},
    }
    initial_agent_positions = {
        "vehicle_0": vehicle_node,
        "human_0": human_node,
    }

    return TransportScenario(
        network=graph,
        poi_distribution=poi_distribution,
        human_distribution=human_distribution,
        vehicle_distribution=vehicle_distribution,
        initial_agent_positions=initial_agent_positions,
        human_goal_nodes=[human_goal_node],
    )


@hydra.main(config_path="../config", config_name="train")
def main(cfg: DictConfig):
    """
    Run a small DQN training job.
    """
    # The ring scenario defines exactly one human and one vehicle, so keep the
    # environment's agent counts consistent with it.
    cfg.env.scenario.num_humans = 1
    cfg.env.scenario.num_vehicles = 1

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

    scenario = build_ring_scenario(cfg.env.scenario.num_nodes)

    env = make_env(
        cfg,
        output_dir=output_dir,
        seed=cfg.env.scenario.seed,
        monitor=True,
        render_mode=None,
        reward_function=equity_reward.reward,
        scenario=scenario,
    )

    env.env.base_env.human_policy_kwargs = {}
    env.env.base_env.human_policy_class = RandomHumanPolicy

    save_training_scenario_diagnostics(env.env, output_dir)

    equity_reward.initialize(
        env.env.base_env,
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
        tau=cfg.dqn.tau,
        tensorboard_log=output_dir,
    )

    model.learn(total_timesteps=cfg.dqn.total_timesteps, progress_bar=True)
    if cfg.dqn.save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    with open_dict(cfg):
        cfg.save_decision_replay_video = True
        cfg.decision_replay_fps = 2
        cfg.dqn_video_episodes = 1
        cfg.episodes = 20
        cfg.video_fps = 20
    evaluate_model(cfg, output_dir, model, env.env)
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
