from collections import Counter
import csv
from pathlib import Path
import sys
from typing import Optional
import json
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
matplotlib.use("Agg")
from utils import make_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: str(PROJECT_ROOT))

from accessibility_equity.wrappers.dqn_wrapper import DQNTransportWrapper
from accessibility_equity.algorithms import MaskedDQN as DQN
from accessibility_equity.visualization import (
    render_uniform_frames,
    save_video,
    start_video_recording,
)
from accessibility_equity.wrappers import REWARD_SCALE
from accessibility_equity.policies.vehicle_heuristics import TSPVehicleAgent, GoToHumanVehicleAgent
from accessibility_equity.rewards.efficient_equity_reward import EquityReward
from reward_analysis import plot_reward_analysis


def _project_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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


def build_step_record(env, equity_reward, dqn_step, previous_state):
    raw_env = env.base_env.env
    human_values, vehicle_values = equity_reward.accessibility_values(raw_env._generate_observations())

    record = {
        "dqn_step": int(dqn_step),
        "env_step": int(env.base_env.step_count),
        "completed_phase": (
            previous_state["next_phase"] if previous_state else None
        ),
        "next_phase": raw_env.step_type,
        "sim_time_hours": float(raw_env.real_time),
        "humans": {},
        "vehicles": {},
        "events": [],
        "raw_reward": None,
        "scaled_reward": None,
    }

    for agent in [*raw_env.human_agents, *raw_env.vehicle_agents]:
        position = raw_env.agent_positions[agent]
        on_edge = isinstance(position, tuple)
        data = (
            {"edge": position[0]}
            if on_edge else {"node": position}
        )

        if agent in raw_env.human_agents:
            aboard = raw_env.human_aboard.get(agent)
            data.update(
                state="riding" if aboard is not None else (
                    "walking" if on_edge else "waiting"
                ),
                destination=raw_env.human_destinations.get(agent),
                aboard=aboard,
                xh=float(human_values[agent]),
            )
            record["humans"][agent] = data

            if previous_state is not None:
                previous_human = previous_state["humans"][agent]
                previous_destination = previous_human.get("destination")

                if (
                    previous_destination is not None
                    and not on_edge
                    and position == previous_destination
                    and previous_human.get("node") != previous_destination
                ):
                    record["events"].append({
                        "type": "arrived",
                        "human": agent,
                        "destination": previous_destination,
                    })

                old = previous_human["aboard"]
                if old != aboard:
                    for event, vehicle in [
                        ("unboarding", old), ("boarding", aboard)
                    ]:
                        if vehicle is not None:
                            record["events"].append({
                                "type": event,
                                "human": agent,
                                "vehicle": vehicle,
                            })

        else:
            destination = raw_env.vehicle_destinations.get(agent)
            data.update(
                state="moving" if on_edge else "at_node",
                destination=destination,
                route_remaining=(
                    equity_reward._compute_vehicle_route(position, destination)
                    if destination is not None else []
                ),
                route_basis="reward_model",
                passengers=[
                    h for h in raw_env.human_agents
                    if raw_env.human_aboard.get(h) == agent
                ],
                xh=float(vehicle_values[agent]),
            )
            record["vehicles"][agent] = data

    return record

def json_ready(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(v) for v in value]
    if isinstance(value, float):
        return round(value, 2)
    return value


def write_agent_history(log_dir, records):
    log_dir.mkdir(parents=True, exist_ok=True)

    for group in ("humans", "vehicles"):
        for agent in records[0][group]:
            log_path = log_dir / f"{agent}.jsonl"
            with log_path.open("w", encoding="utf-8") as log_file:
                for record in records:
                    entry = {
                        "agent_type": group[:-1],
                        "agent": agent,
                        "dqn_step": record["dqn_step"],
                        "env_step": record["env_step"],
                        "completed_phase": record["completed_phase"],
                        "next_phase": record["next_phase"],
                        "sim_time_hours": record["sim_time_hours"],
                        **record[group][agent],
                        "events": [
                            (
                                "arrived"
                                if event["type"] == "arrived"
                                else f"{event['type']}_"
                                     f"{event['vehicle'] if group == 'humans' else event['human']}"
                            )
                            for event in record["events"]
                            if event.get(group[:-1]) == agent
                        ],
                        "raw_reward": record["raw_reward"],
                        "scaled_reward": record["scaled_reward"],
                    }
                    log_file.write(
                        json.dumps(
                            json_ready(entry),
                            ensure_ascii=False,
                        ) + "\n"
                    )

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
    raw_env.video_reward_calculator = equity_reward

    step_log_dir = None
    episode_step_records = []
    previous_step_record = None
    pending_step_records = []
    current_dqn_step = 0

    if record_video:
        if video_path is None:
            raise ValueError("video_path is required when record_video=True.")
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        start_video_recording(raw_env)

        step_log_dir = video_path.parent / f"{video_path.stem}_logs"

        previous_step_record = build_step_record(
            env, equity_reward, dqn_step=0, previous_state=None
        )
        episode_step_records.append(previous_step_record)

    
        def _capture_auto_advance_frame(wrapper, _info):
            nonlocal previous_step_record

            render_uniform_frames(wrapper.base_env.env)
            record = build_step_record(
                env,
                equity_reward,
                dqn_step=current_dqn_step,
                previous_state=previous_step_record,
            )
            pending_step_records.append(record)
            previous_step_record = record

        env.auto_advance_callback = _capture_auto_advance_frame

    try:
        terminated = False
        truncated = False
        episode_reward = 0.0
        raw_episode_reward = 0.0
        episode_steps = 0
        reward_records = []  # Rewards for each decision step.

        action_counts = Counter()
        invalid_action_count = 0
        total_action_count = 0
        start_node = _current_vehicle_node(env)
        end_node = start_node
        visited_nodes = set()
        if start_node is not None:
            visited_nodes.add(start_node)

        while not (terminated or truncated):
            current_dqn_step = episode_steps + 1
            pending_step_records.clear()

            action_value = _select_action(policy_name, model, env, obs)
            action_counts[action_value] += 1
            step_start_time = float(raw_env.real_time)
            obs, reward, terminated, truncated, info = env.step(action_value)
            step_end_time = float(raw_env.real_time)
            reward_records.append({
                "step": episode_steps + 1,
                "start_time": step_start_time,
                "end_time": step_end_time,
                "delta_time": step_end_time - step_start_time,
                "raw_reward": float(info["raw_reward"]),
                "scaled_reward": float(reward),
            })

            if record_video and pending_step_records:
                # Reward belongs to the complete DQN decision, so place it on
                # the final underlying step.
                pending_step_records[-1]["raw_reward"] = float(
                    info.get("raw_reward", reward)
                )
                pending_step_records[-1]["scaled_reward"] = float(reward)

                episode_step_records.extend(pending_step_records)
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
            write_agent_history(step_log_dir, episode_step_records)
            print(f"Agent histories saved to '{step_log_dir}'")
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
        "reward_records": reward_records,
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
        policy_step_records = []
        policy_episode_records = []
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
            for record in result["reward_records"]:
                policy_step_records.append({
                    **record,
                    "episode": episode + 1,
                    "global_step": len(policy_step_records) + 1,
                })

            policy_episode_records.append({
                "episode": episode + 1,
                "scaled_reward": result["reward"],
            })
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
        # Plot once after all episodes for this policy, independently of video.
        plot_reward_analysis(
            policy_step_records,
            policy_episode_records,
            output_dir / "reward_analysis" / "evaluation" / policy_name,
            title_prefix=f"{policy_name.upper()} Evaluation",
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
    output_dir = _project_path(cfg.model_dir)

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
