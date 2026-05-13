"""
DQN variant that masks invalid actions during action selection.

This class intentionally only handles action selection:
- epsilon/random exploration samples from currently valid actions
- greedy Q selection sets invalid actions to a very low value before argmax

It does not mask the TD target in ``train()`` yet. That is the next, separate
step once the mask is stored in observations or the replay buffer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.noise import ActionNoise


class MaskedDQN(DQN):
    """DQN with action-mask-aware action selection for discrete actions."""

    def _format_action_masks(
        self,
        action_masks: Optional[np.ndarray],
        n_envs: int,
    ) -> np.ndarray:
        if action_masks is None:
            if self.env is None:
                return np.ones((n_envs, self.action_space.n), dtype=bool)
            try:
                action_masks = np.asarray(
                    self.env.env_method("action_masks"),
                    dtype=bool,
                )
            except AttributeError:
                return np.ones((n_envs, self.action_space.n), dtype=bool)
        else:
            action_masks = np.asarray(action_masks, dtype=bool)

        if action_masks.ndim == 1:
            action_masks = action_masks.reshape(1, -1)

        if action_masks.shape[0] != n_envs:
            if action_masks.shape[0] == 1:
                action_masks = np.repeat(action_masks, n_envs, axis=0)
            else:
                raise ValueError(
                    f"Expected {n_envs} action masks, got shape {action_masks.shape}."
                )

        if action_masks.shape[1] != self.action_space.n:
            raise ValueError(
                f"Expected mask width {self.action_space.n}, got {action_masks.shape[1]}."
            )

        if not action_masks.any(axis=1).all():
            raise ValueError("Every action mask must contain at least one valid action.")

        return action_masks

    def _sample_masked_random_actions(self, action_masks: np.ndarray) -> np.ndarray:
        actions = []
        for mask in action_masks:
            valid_actions = np.flatnonzero(mask)
            actions.append(int(self.action_space.np_random.choice(valid_actions)))
        return np.asarray(actions)

    def _predict_masked_greedy(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)
        with th.no_grad():
            q_values = self.q_net(obs_tensor)
            mask_tensor = th.as_tensor(
                action_masks,
                dtype=th.bool,
                device=q_values.device,
            )
            q_values = q_values.masked_fill(~mask_tensor, -1e9)
            actions = q_values.argmax(dim=1)

        actions_np = actions.cpu().numpy().reshape((-1, *self.action_space.shape))
        return actions_np, vectorized_env

    def _masks_from_observation(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        n_envs: int,
    ) -> Optional[np.ndarray]:
        if not isinstance(observation, dict) or "action_mask" not in observation:
            return None

        action_masks = np.asarray(observation["action_mask"], dtype=bool)
        if action_masks.ndim == 1:
            action_masks = action_masks.reshape(1, -1)
        return self._format_action_masks(action_masks, n_envs)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        if not isinstance(self.action_space, spaces.Discrete):
            return super().predict(observation, state, episode_start, deterministic)

        if self.policy.is_vectorized_observation(observation):
            if isinstance(observation, dict):
                n_envs = observation[next(iter(observation.keys()))].shape[0]
            else:
                n_envs = observation.shape[0]
            vectorized_env = True
        else:
            n_envs = 1
            vectorized_env = False

        if action_masks is None:
            action_masks = self._masks_from_observation(observation, n_envs)
        masks = self._format_action_masks(action_masks, n_envs)

        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._sample_masked_random_actions(masks)
            actions = actions.reshape((-1, *self.action_space.shape))
        else:
            actions, vectorized_env = self._predict_masked_greedy(observation, masks)

        if not vectorized_env:
            actions = actions.squeeze(axis=0)
        return actions, state

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(self.action_space, spaces.Discrete):
            return super()._sample_action(learning_starts, action_noise, n_envs)

        action_masks = self._format_action_masks(None, n_envs)

        if self.num_timesteps < learning_starts:
            unscaled_action = self._sample_masked_random_actions(action_masks)
        elif np.random.rand() < self.exploration_rate:
            unscaled_action = self._sample_masked_random_actions(action_masks)
        else:
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _state = self.predict(
                self._last_obs,
                deterministic=True,
                action_masks=action_masks,
            )
            unscaled_action = np.asarray(unscaled_action)

        buffer_action = unscaled_action
        action = buffer_action
        return action, buffer_action

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        if not (
            isinstance(self.observation_space, spaces.Dict)
            and "action_mask" in self.observation_space.spaces
        ):
            return super().train(gradient_steps, batch_size)

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_action_masks = replay_data.next_observations["action_mask"].bool()
                next_q_values = next_q_values.masked_fill(~next_action_masks, -1e9)
                next_q_values, _ = next_q_values.max(dim=1)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * discounts * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
