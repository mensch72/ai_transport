from pathlib import Path

from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def test_train_config_contains_environment_and_dqn_defaults():
    cfg = OmegaConf.merge(
        {"env": OmegaConf.load(CONFIG_DIR / "env" / "default.yaml")},
        OmegaConf.load(CONFIG_DIR / "train.yaml"),
    )

    assert cfg.env.scenario.num_nodes == 6
    assert cfg.env.scenario.num_humans == 10
    assert cfg.env.scenario.num_vehicles == 1
    assert cfg.env.scenario.seed == 42
    assert cfg.env.scenario.central_count == 1
    assert cfg.env.mobility.human_walking_speed_kmh == 3.0
    assert cfg.env.mobility.vehicle_speed_kmh == 30.0
    assert cfg.env.reward.utility_mode == "equity"
    assert cfg.env.network.speed_mean == 30.0
    assert cfg.env.network.bidirectional_prob == 0.85
    assert cfg.env.network.central_node_count == 1
    assert cfg.env.poi.residential_ratio == 0.5
    assert cfg.env.poi.a == 0.5
    assert cfg.env.poi.b == 0.5
    assert cfg.env.poi.poi_profiles.workplace.frequency == 0.40
    assert cfg.dqn.decision_mode == "routing"
    assert cfg.dqn.total_timesteps == 20_000
    assert cfg.dqn.profile_enabled is False


def test_eval_config_points_to_a_model_run():
    cfg = OmegaConf.load(CONFIG_DIR / "eval.yaml")

    assert cfg.model_dir
    assert cfg.model_filename == "accessibility_equity_dqn_single_vehicle"
    assert cfg.episodes > 0
