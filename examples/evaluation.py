from collections import Counter
import csv
from pathlib import Path
import sys

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

from accessibility_equity.algorithms import MaskedDQN as DQN
from accessibility_equity.visualization import render_episode_frame_array
from accessibility_equity.wrappers import REWARD_SCALE
from accessibility_equity.heuristics import TSPVehicleAgent
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
    if policy_name == "tsp":
        return model.get_action(obs)
    if policy_name == "random":
        return int(env.action_space.sample())
    if policy_name == "always_pass":
        return 0
    raise ValueError(f"Unknown evaluation policy: {policy_name}")


def _open_video_writer(video_path, fps=2):
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio

        return imageio.get_writer(str(video_path), fps=fps), video_path
    except Exception as exc:
        gif_path = video_path.with_suffix(".gif")
        try:
            return imageio.get_writer(str(gif_path), mode="I", fps=fps), gif_path
        except Exception as gif_exc:
            print(
                f"Could not open decision replay video writer for '{video_path}' "
                f"or '{gif_path}': {exc}; {gif_exc}"
            )
            return None, None


def _action_description(env, action_value):
    if int(action_value) <= 0:
        return "pass / clear destination"
    target_index = int(action_value) - 1
    if 0 <= target_index < len(env.node_order):
        return f"go to node {env.node_order[target_index]}"
    return f"unknown action {action_value}"


def _format_float(value):
    if value is None:
        return "NA"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "NA"


def _append_decision_replay_frame(
    writer,
    env,
    policy_name,
    episode_index,
    step_index,
    seed,
    action_value=None,
    info=None,
    reward=None,
):
    info = info or {}
    if action_value is None:
        decision_text = "initial state before first DQN decision"
    else:
        decision_text = f"action={action_value} ({_action_description(env, action_value)})"

    title_parts = [
        f"{policy_name.upper()} replay | episode={episode_index + 1} | step={step_index} | seed={seed}",
        decision_text,
    ]
    if reward is not None:
        title_parts.append(
            "impact: "
            f"reward={_format_float(reward)}, "
            f"raw={_format_float(info.get('raw_reward', reward))}, "
            f"U={_format_float(info.get('current_utility'))} -> {_format_float(info.get('next_utility'))}, "
            f"A={_format_float(info.get('current_total_accessibility'))} -> {_format_float(info.get('next_total_accessibility'))}"
        )

    frame = render_episode_frame_array(
        scenario=env.base_env.scenario,
        agent_positions=dict(env.base_env.env.agent_positions),
        human_destinations=dict(env.base_env.env.human_destinations),
        vehicle_destinations=dict(env.base_env.env.vehicle_destinations),
        title="\n".join(title_parts),
    )
    writer.append_data(frame)


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
    replay_writer=None,
):
    """
    Run one evaluation episode for a named policy.
    """
    equity_reward = EquityReward(
        cfg.env.reward.beta,
        cfg.env.reward.alpha,
        cfg.env.reward.xi,
        cfg.env.reward.eta,
        cfg.env.mobility.human_walking_speed_kmh,
        cfg.env.mobility.vehicle_speed_kmh)
    env = make_env(
        cfg,
        seed=seed,
        monitor=False,
        render_mode=None,
        reward_function=equity_reward.reward
    )
    equity_reward.initialize(
        env.base_env.env.network,
        env.base_env.vehicle_agents,
        env.base_env.human_agents
    )

    if policy_name == 'tsp':
        model = TSPVehicleAgent(env)
    obs, _info = env.reset(seed=seed)
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
    if replay_writer is not None:
        _append_decision_replay_frame(
            writer=replay_writer,
            env=env,
            policy_name=policy_name,
            episode_index=episode_index,
            step_index=0,
            seed=seed,
        )

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
        if replay_writer is not None:
            _append_decision_replay_frame(
                writer=replay_writer,
                env=env,
                policy_name=policy_name,
                episode_index=episode_index,
                step_index=episode_steps,
                seed=seed,
                action_value=action_value,
                info=info,
                reward=reward,
            )

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


def evaluate_model(cfg, output_dir, model):
    """
    Run policy comparison after training.
    """
    video_dir = output_dir / "videos"
    scenario_image_path = output_dir / "scenario.png"
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
    policy_names = ["dqn", "random", "always_pass", "tsp"]
    replay_writer = None
    replay_video_path = None
    if cfg.save_decision_replay_video:
        replay_writer, replay_video_path = _open_video_writer(
            decision_replay_video_path,
            fps=cfg.decision_replay_fps,
        )

    try:
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

            for episode in range(cfg.episodes):
                result = _run_policy_episode(
                    cfg=cfg,
                    policy_name=policy_name,
                    model=model,
                    episode_index=episode,
                    seed=cfg.env.scenario.seed,
                    print_step_details=(episode == 0),
                    detail_steps=8,
                    replay_writer=replay_writer if policy_name == "dqn" else None,
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
    finally:
        if replay_writer is not None:
            replay_writer.close()
            emit(f"Decision replay video saved to: {replay_video_path}")

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
