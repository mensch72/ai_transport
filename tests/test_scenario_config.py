import pytest

from accessibility_equity.config import (
    ExperimentConfig,
    MobilityConfig,
    RewardConfig,
    ScenarioConfig,
    load_experiment_config_from_env,
    load_scenario_config_from_env,
)


def test_default_scenario_config_matches_training_defaults(monkeypatch):
    for name in (
        "AE_NUM_NODES",
        "AE_NUM_HUMANS",
        "AE_NUM_VEHICLES",
        "AE_SEED",
        "AE_CENTRAL_COUNT",
    ):
        monkeypatch.delenv(name, raising=False)

    config = load_scenario_config_from_env()

    assert config == ScenarioConfig()
    assert config.poi_kwargs == {"central_count": 1}


def test_scenario_config_can_be_overridden_from_environment(monkeypatch):
    monkeypatch.setenv("AE_NUM_NODES", "7")
    monkeypatch.setenv("AE_NUM_HUMANS", "12")
    monkeypatch.setenv("AE_NUM_VEHICLES", "2")
    monkeypatch.setenv("AE_SEED", "99")
    monkeypatch.setenv("AE_CENTRAL_COUNT", "3")

    config = load_scenario_config_from_env()

    assert config == ScenarioConfig(
        num_nodes=7,
        num_humans=12,
        num_vehicles=2,
        seed=99,
        central_count=3,
    )


def test_scenario_config_rejects_non_integer_environment_values(monkeypatch):
    monkeypatch.setenv("AE_NUM_NODES", "five")

    with pytest.raises(ValueError):
        load_scenario_config_from_env()


def test_experiment_config_collects_reward_mobility_and_dqn(monkeypatch):
    monkeypatch.setenv("AE_HUMAN_WALKING_SPEED_KMH", "3.5")
    monkeypatch.setenv("AE_VEHICLE_SPEED_KMH", "28")
    monkeypatch.setenv("AE_EDGE_SPEED_KMH", "25")
    monkeypatch.setenv("AE_REWARD_BETA", "1.2")
    monkeypatch.setenv("AE_REWARD_ALPHA", "0.7")
    monkeypatch.setenv("AE_REWARD_XI", "1.1")
    monkeypatch.setenv("AE_REWARD_ETA", "2.5")
    monkeypatch.setenv("AE_DQN_GAMMA", "0.95")
    monkeypatch.setenv("AE_DQN_TOTAL_TIMESTEPS", "123")
    monkeypatch.setenv("AE_DQN_SAVE_DECISION_REPLAY_VIDEO", "false")
    monkeypatch.setenv("AE_DQN_DECISION_REPLAY_FPS", "4")

    config = load_experiment_config_from_env()

    assert isinstance(config, ExperimentConfig)
    assert config.mobility == MobilityConfig(
        human_walking_speed_kmh=3.5,
        vehicle_speed_kmh=28.0,
        edge_speed_kmh=25.0,
    )
    assert config.reward == RewardConfig(beta=1.2, alpha=0.7, xi=1.1, eta=2.5)
    assert config.dqn.gamma == 0.95
    assert config.dqn.total_timesteps == 123
    assert config.dqn.save_decision_replay_video is False
    assert config.dqn.decision_replay_fps == 4
