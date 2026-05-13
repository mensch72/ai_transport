"""Tests for masked DQN action selection."""

from accessibility_equity.algorithms import MaskedDQN
from accessibility_equity.wrappers import create_dqn_env


def test_masked_dqn_random_prediction_samples_only_valid_actions():
    env = create_dqn_env(
        num_humans=2,
        num_nodes=8,
        seed=3,
        use_action_masking=True,
        decision_mode="routing_departing",
    )
    obs, _info = env.reset(seed=3)

    # Move from routing to departing, where only outgoing-neighbor node actions
    # should be sampled.
    obs, _reward, _terminated, _truncated, _info = env.step(0)
    assert env.base_env.env.step_type == "departing"
    valid_actions = set(env.action_masks().nonzero()[0])
    assert valid_actions

    model = MaskedDQN(
        "MultiInputPolicy",
        env,
        learning_starts=0,
        buffer_size=16,
        batch_size=4,
        train_freq=1,
        gradient_steps=1,
        verbose=0,
    )
    model.exploration_rate = 1.0

    for _ in range(20):
        action, _state = model.predict(
            obs,
            deterministic=False,
            action_masks=env.action_masks(),
        )
        assert int(action) in valid_actions

    env.close()
