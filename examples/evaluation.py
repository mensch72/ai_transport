from collections import Counter
import csv
from pathlib import Path
import sys
from typing import Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import make_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from accessibility_equity.wrappers.dqn_wrapper import DQNTransportWrapper
from accessibility_equity.algorithms import MaskedDQN as DQN
from accessibility_equity.visualization import (
    render_uniform_frames,
    save_scenario_figure,
    save_video,
    start_video_recording,
)
from accessibility_equity.wrappers import REWARD_SCALE
from accessibility_equity.heuristics import TSPVehicleAgent, GoToHumanVehicleAgent
from accessibility_equity.rewards.efficient_equity_reward import EquityReward


def _select_action(policy_name, model, env, obs):
    """
    Choose one action for the requested evaluation policy.
    """
    if policy_name == "dqn":
        action, _state = model.predict(
            obs,
            deterministic=True,
            action_masks=env.action_masks(),
        )
        return int(action)
    if policy_name in ["tsp", "gth"]:
        return model.get_action(obs)
    if policy_name == "random":
        return int(env.action_space.sample())
    if policy_name == "always_pass":
        return 0
    raise ValueError(f"Unknown evaluation policy: {policy_name}")


def _format_boarding_debug_line(policy_name, episode_index, dqn_step, record):
    """
    Format one boarding-phase diagnostic record for console and log output.
    """
    return (
        f"[boarding-debug] policy={policy_name} "
        f"episode={episode_index + 1} "
        f"dqn_step={dqn_step} "
        f"underlying_step={record.get('dqn_underlying_step')} "
        f"env_step={record.get('step_count')} "
        f"time={float(record.get('real_time') or 0.0):.1f} "
        f"human={record.get('human')} "
        f"human_node={record.get('human_node')} "
        f"vehicle_nodes={record.get('vehicle_nodes')} "
        f"vehicle_destinations={record.get('vehicle_destinations')} "
        f"same_node_vehicles={record.get('same_node_vehicles')} "
        f"human_aboard_before={record.get('human_aboard_before')} "
        f"human_aboard_after={record.get('human_aboard_after')} "
        f"human_boarding_action={record.get('human_boarding_action')} "
        f"human_action_space_n={record.get('human_action_space_n')} "
        f"reason={record.get('human_action_reason')}"
    )


def _current_vehicle_node(env):
    position = env.base_env.env.agent_positions.get(env.vehicle_agent)
    return env._position_to_node(position)


def _run_policy_episode(
    cfg,
    policy_name,
    model,
    episode_index,
    seed,
    print_step_details=False,
    detail_steps=8,
    record_video=False,
    video_path=None,
    video_fps=24,
    env: Optional[DQNTransportWrapper] = None,
):
    """
    Run one evaluation episode for a named policy.
    """
    equity_reward = EquityReward(
        cfg.env.reward.beta,
        cfg.env.reward.alpha,
        cfg.env.reward.xi,
        cfg.env.reward.eta,
        cfg.dqn.gamma,
        cfg.env.mobility.human_walking_speed_kmh,
        cfg.env.mobility.vehicle_speed_kmh)
    if env is None:
        env = make_env(
            cfg,
            seed=cfg.env.scenario.seed,
            monitor=False,
            render_mode=None,
            reward_function=equity_reward.reward
        )
    equity_reward.initialize(
        env.base_env,
    )

    if policy_name == 'tsp':
        model = TSPVehicleAgent(env)
    if policy_name == 'gth':
        model = GoToHumanVehicleAgent(env)
    obs, _info = env.reset(seed=cfg.env.scenario.seed)
    raw_env = env.base_env.env
    if record_video:
        if video_path is None:
            raise ValueError("video_path is required when record_video=True.")
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        start_video_recording(raw_env)
    
        def _capture_auto_advance_frame(wrapper, _info):
            render_uniform_frames(wrapper.base_env.env)
        
        env.auto_advance_callback = _capture_auto_advance_frame

    try:
        terminated = False
        truncated = False
        episode_reward = 0.0
        raw_episode_reward = 0.0
        episode_steps = 0
        action_counts = Counter()
        invalid_action_count = 0
        total_action_count = 0
        start_node = _current_vehicle_node(env)
        end_node = start_node
        visited_nodes = set()
        if start_node is not None:
            visited_nodes.add(start_node)
        while not (terminated or truncated):
            action_value = _select_action(policy_name, model, env, obs)
            action_counts[action_value] += 1
            obs, reward, terminated, truncated, info = env.step(action_value)
            end_node = _current_vehicle_node(env)
            if end_node is not None:
                visited_nodes.add(end_node)
            total_action_count += 1
            if not bool(info.get("dqn_action_was_valid", True)):
                invalid_action_count += 1
            episode_reward += float(reward)
            raw_episode_reward += float(info.get("raw_reward", reward))
            episode_steps += 1
            for record in info.get("dqn_boarding_debug_records", []):
                line = _format_boarding_debug_line(
                    policy_name=policy_name,
                    episode_index=episode_index,
                    dqn_step=episode_steps,
                    record=record,
                )
                print(line)

            if print_step_details and episode_steps <= detail_steps:
                print(
                    f"  step={episode_steps:03d} "
                    f"action={action_value:02d} "
                    f"valid_action={bool(info.get('dqn_action_was_valid', True))} "
                    f"scaled_reward={float(reward): .6f} "
                    f"raw_reward={float(info.get('raw_reward', reward)): .6f} "
                    f"current_U={info.get('current_utility')} "
                    f"current_U_norm={info.get('current_normalized_utility')} "
                    f"next_U={info.get('next_utility')} "
                    f"current_A={info.get('current_total_accessibility')} "
                    f"next_A={info.get('next_total_accessibility')}"
                )
        if record_video:
            save_video(raw_env, filename=str(video_path), fps=video_fps)
    finally:
        if record_video:
            env.auto_advance_callback = None
        env.close()

    return {
        "reward": episode_reward,
        "raw_reward": raw_episode_reward,
        "steps": episode_steps,
        "action_counts": action_counts,
        "invalid_action_count": invalid_action_count,
        "total_action_count": total_action_count,
        "start_node": start_node,
        "end_node": end_node,
        "unique_nodes_visited": len(visited_nodes),
        "visited_nodes": sorted(visited_nodes, key=str),
    }


def evaluate_model(cfg, output_dir, model, env = None):
    """
    Run policy comparison after training.
    """
    video_dir = output_dir / "videos"
    log_path = output_dir / "evaluation_summary.txt"
    evaluation_episodes_summary_csv_path = output_dir / "evaluation_episodes_summary.csv"
    decision_replay_video_path = video_dir / "dqn_decision_replay_all_episodes.mp4"

    log_lines = []
    episode_summary_rows = []

    def emit(line=""):
        print(line)
        log_lines.append(line)

    print("\n" + "=" * 70)
    print("Evaluating trained DQN model against baselines")
    print("=" * 70)
    emit("=" * 70)
    emit("Evaluating trained DQN model against baselines")
    emit(f"Rewards reported below are scaled by REWARD_SCALE={REWARD_SCALE:g}.")
    emit("Use raw_reward for the original unscaled accessibility-equity reward.")
    emit("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.dqn_video_episodes:
        video_dir.mkdir(parents=True, exist_ok=True)
    policy_names = ["dqn", "random", "always_pass", "tsp", "gth"]
    for policy_name in policy_names:
        emit("")
        emit("-" * 70)
        emit(f"Policy: {policy_name}")
        emit("-" * 70)

        episode_rewards = []
        episode_raw_rewards = []
        action_counts = Counter()
        invalid_action_count = 0
        total_action_count = 0

        first_video_episode = max(0, cfg.episodes - int(cfg.dqn_video_episodes))
        for episode in range(cfg.episodes):
            record_video = policy_name == "dqn" and episode >= first_video_episode
            video_path = None
            if record_video:
                video_path = video_dir / f"dqn_eval_episode_{episode + 1:03d}.mp4"
            result = _run_policy_episode(
                cfg,
                policy_name=policy_name,
                model=model,
                episode_index=episode,
                seed=cfg.env.scenario.seed,
                print_step_details=(episode == 0),
                detail_steps=8,
                record_video=record_video,
                video_path=video_path,
                video_fps=cfg.video_fps,
                env=env
            )
            episode_rewards.append(result["reward"])
            episode_raw_rewards.append(result["raw_reward"])
            action_counts.update(result["action_counts"])
            invalid_action_count += result["invalid_action_count"]
            total_action_count += result["total_action_count"]

            invalid_rate = (
                result["invalid_action_count"] / result["total_action_count"]
                if result["total_action_count"]
                else 0.0
            )
            emit(
                f"Episode {episode + 1}: "
                f"scaled_reward={result['reward']:.6f}, "
                f"raw_reward={result['raw_reward']:.6f}, "
                f"steps={result['steps']}, "
                f"invalid_actions={result['invalid_action_count']}/"
                f"{result['total_action_count']} ({invalid_rate:.2%})"
            )
            if record_video:
                emit(f"  video={video_path}")

            most_common_action = None
            if result["action_counts"]:
                most_common_action = result["action_counts"].most_common(1)[0][0]
            episode_summary_rows.append(
                {
                    "policy": policy_name,
                    "episode": episode + 1,
                    "reward": result["reward"],
                    "raw_reward": result["raw_reward"],
                    "steps": result["steps"],
                    "invalid_actions": result["invalid_action_count"],
                    "total_actions": result["total_action_count"],
                    "most_common_action": most_common_action,
                    "unique_nodes_visited": result["unique_nodes_visited"],
                    "start_node": result["start_node"],
                    "end_node": result["end_node"],
                    "visited_nodes": " ".join(str(node) for node in result["visited_nodes"]),
                }
            )
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        mean_raw_reward = sum(episode_raw_rewards) / len(episode_raw_rewards)
        invalid_action_rate = (
            invalid_action_count / total_action_count
            if total_action_count
            else 0.0
        )
        emit(f"Mean evaluation scaled reward: {mean_reward:.6f}")
        emit(f"Mean evaluation raw reward: {mean_raw_reward:.6f}")
        emit(
            f"Invalid action rate: {invalid_action_count}/"
            f"{total_action_count} ({invalid_action_rate:.2%})"
        )
        emit(f"Most common actions: {action_counts.most_common(10)}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    with evaluation_episodes_summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "policy",
            "episode",
            "reward",
            "raw_reward",
            "steps",
            "invalid_actions",
            "total_actions",
            "most_common_action",
            "unique_nodes_visited",
            "start_node",
            "end_node",
            "visited_nodes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_summary_rows)
    print(f"Evaluation summary saved to '{log_path}'")
    print(f"Evaluation episode summary CSV saved to '{evaluation_episodes_summary_csv_path}'")

@hydra.main(config_path='../config', config_name='eval')
def main(cfg: DictConfig):
    output_dir = Path(cfg.model_dir)

    # Reuse the exact env/dqn config the model was trained with so the
    # evaluation environment matches the trained model. Hydra saved this
    # config when the training run was executed.
    train_cfg = OmegaConf.load(output_dir / ".hydra" / "config.yaml")
    cfg = OmegaConf.merge(train_cfg, cfg)

    model_path = output_dir / 'models' / f'{cfg.model_filename}.zip'
    model = DQN.load(model_path)
    print(f"Loaded model from '{model_path}'")

    evaluate_model(cfg, output_dir=output_dir, model=model)

if __name__ == "__main__":
    main()
