"""Device-native PPO core for TensorFlight training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from BalloonPoppingGymEnv.tensorflight.observations import ACTION_SIZE, OBSERVATION_SIZE


Tensor = torch.Tensor


def _orthogonal(layer: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class TanhActorCritic(nn.Module):
    """Separate 256x256 actor/critic with a tanh-squashed Gaussian actor."""

    def __init__(
        self,
        observation_size: int = OBSERVATION_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 256,
        log_std_init: float = -1.0,
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_size),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        for module in (*self.actor, *self.critic):
            if isinstance(module, nn.Linear):
                _orthogonal(module, math.sqrt(2))
        _orthogonal(self.actor[-1], 0.01)  # type: ignore[arg-type]
        _orthogonal(self.critic[-1], 1.0)  # type: ignore[arg-type]
        self.log_std = nn.Parameter(torch.full((action_size,), log_std_init))

    def actor_mean(self, observation: Tensor) -> Tensor:
        return self.actor(observation)

    def value(self, observation: Tensor) -> Tensor:
        return self.critic(observation).squeeze(-1)

    @staticmethod
    def _log_prob(mean: Tensor, log_std: Tensor, latent: Tensor) -> Tensor:
        inverse_variance = torch.exp(-2 * log_std)
        gaussian = (
            -0.5 * (latent - mean).square() * inverse_variance
            - log_std
            - 0.5 * math.log(2 * math.pi)
        ).sum(-1)
        action = torch.tanh(latent)
        correction = torch.log(1 - action.square() + 1e-6).sum(-1)
        return gaussian - correction

    @torch.no_grad()
    def act(
        self, observation: Tensor, *, generator: torch.Generator
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean = self.actor_mean(observation)
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        latent = mean + torch.exp(self.log_std) * noise
        action = torch.tanh(latent)
        log_prob = self._log_prob(mean, self.log_std, latent)
        return action, latent, log_prob, self.value(observation)

    def evaluate(
        self, observation: Tensor, latent: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean = self.actor_mean(observation)
        log_prob = self._log_prob(mean, self.log_std, latent)
        entropy = (self.log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
        return log_prob, entropy.expand_as(log_prob), self.value(observation)


@dataclass(frozen=True)
class PPOHyperparameters:
    gamma: float = 1.0 - 0.01 / 15.0
    gae_lambda: float = 0.97
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    entropy_coefficient: float = 5e-3
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 65_536
    target_kl: float | None = 0.02


class DeviceRolloutBuffer:
    """Fixed-shape rollout storage that never leaves the selected device."""

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if horizon < 1 or num_envs < 1:
            raise ValueError("horizon and num_envs must be positive")
        shape = (horizon, num_envs)
        self.horizon = horizon
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.observations = torch.empty(
            (*shape, OBSERVATION_SIZE), device=self.device, dtype=dtype
        )
        self.latent_actions = torch.empty(
            (*shape, ACTION_SIZE), device=self.device, dtype=dtype
        )
        self.log_probabilities = torch.empty(shape, device=self.device, dtype=dtype)
        self.values = torch.empty_like(self.log_probabilities)
        self.rewards = torch.empty_like(self.log_probabilities)
        self.bootstrap_values = torch.empty_like(self.log_probabilities)
        self.terminated = torch.empty(shape, device=self.device, dtype=torch.bool)
        self.truncated = torch.empty_like(self.terminated)
        self.policy_active = torch.empty_like(self.terminated)
        self.advantages = torch.empty_like(self.log_probabilities)
        self.returns = torch.empty_like(self.log_probabilities)

    @property
    def transition_count(self) -> int:
        return self.horizon * self.num_envs

    @property
    def allocated_bytes(self) -> int:
        tensors = (
            self.observations,
            self.latent_actions,
            self.log_probabilities,
            self.values,
            self.rewards,
            self.bootstrap_values,
            self.terminated,
            self.truncated,
            self.policy_active,
            self.advantages,
            self.returns,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def compute_advantages(self, parameters: PPOHyperparameters) -> None:
        advantages, returns = compute_gae(
            self.rewards,
            self.values,
            self.bootstrap_values,
            self.terminated,
            self.truncated,
            gamma=parameters.gamma,
            gae_lambda=parameters.gae_lambda,
        )
        self.advantages.copy_(advantages)
        self.returns.copy_(returns)


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[Tensor, Tensor]:
    """GAE with truncation bootstrap but no advantage flow across reset."""
    if not (
        rewards.shape
        == values.shape
        == bootstrap_values.shape
        == terminated.shape
        == truncated.shape
    ):
        raise ValueError("all GAE tensors must have the same [H, B] shape")
    advantages = torch.empty_like(rewards)
    following_advantage = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        bootstrap_mask = (~terminated[index]).to(rewards.dtype)
        continuation_mask = (~(terminated[index] | truncated[index])).to(rewards.dtype)
        delta = (
            rewards[index]
            + gamma * bootstrap_values[index] * bootstrap_mask
            - values[index]
        )
        following_advantage = (
            delta + gamma * gae_lambda * continuation_mask * following_advantage
        )
        advantages[index] = following_advantage
    return advantages, advantages + values


@dataclass(frozen=True)
class PPOUpdateMetrics:
    samples: int
    epochs_completed: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float


def update_ppo(
    model: TanhActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: DeviceRolloutBuffer,
    parameters: PPOHyperparameters,
    *,
    shuffle_generator: torch.Generator,
) -> PPOUpdateMetrics:
    rollout.compute_advantages(parameters)
    active_flat = rollout.policy_active.reshape(-1)
    active_indices = torch.nonzero(active_flat, as_tuple=False).squeeze(-1)
    sample_count = active_indices.numel()
    if sample_count == 0:
        return PPOUpdateMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    observations = rollout.observations.reshape(-1, OBSERVATION_SIZE)
    latent_actions = rollout.latent_actions.reshape(-1, ACTION_SIZE)
    old_log_probabilities = rollout.log_probabilities.reshape(-1)
    old_values = rollout.values.reshape(-1)
    returns = rollout.returns.reshape(-1)
    advantages = rollout.advantages.reshape(-1)
    selected_advantages = advantages[active_indices]
    advantage_mean = selected_advantages.mean()
    advantage_std = selected_advantages.std(unbiased=False).clamp_min(1e-8)
    advantages = (advantages - advantage_mean) / advantage_std

    totals = torch.zeros(6, device=rollout.device, dtype=torch.float64)
    minibatches = 0
    epochs_completed = 0
    stop = False
    for _ in range(parameters.epochs):
        order = active_indices[
            torch.randperm(
                sample_count, generator=shuffle_generator, device=rollout.device
            )
        ]
        for start in range(0, sample_count, parameters.minibatch_size):
            indices = order[start : start + parameters.minibatch_size]
            log_probability, entropy, value = model.evaluate(
                observations[indices], latent_actions[indices]
            )
            log_ratio = log_probability - old_log_probabilities[indices]
            ratio = torch.exp(log_ratio)
            minibatch_advantage = advantages[indices]
            unclipped = ratio * minibatch_advantage
            clipped = (
                ratio.clamp(1 - parameters.clip_range, 1 + parameters.clip_range)
                * minibatch_advantage
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()

            value_delta = value - old_values[indices]
            clipped_value = old_values[indices] + value_delta.clamp(
                -parameters.value_clip_range, parameters.value_clip_range
            )
            value_loss = (
                0.5
                * torch.maximum(
                    (value - returns[indices]).square(),
                    (clipped_value - returns[indices]).square(),
                ).mean()
            )
            entropy_mean = entropy.mean()
            loss = (
                policy_loss
                + parameters.value_coefficient * value_loss
                - parameters.entropy_coefficient * entropy_mean
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(), parameters.max_gradient_norm
            )
            optimizer.step()

            with torch.no_grad():
                approximate_kl = ((ratio - 1) - log_ratio).mean()
                clip_fraction = (
                    (torch.abs(ratio - 1) > parameters.clip_range)
                    .to(rollout.dtype)
                    .mean()
                )
                totals += torch.stack(
                    (
                        policy_loss.detach().to(torch.float64),
                        value_loss.detach().to(torch.float64),
                        entropy_mean.detach().to(torch.float64),
                        approximate_kl.detach().to(torch.float64),
                        clip_fraction.detach().to(torch.float64),
                        gradient_norm.detach().to(torch.float64),
                    )
                )
                minibatches += 1
                if (
                    parameters.target_kl is not None
                    and float(approximate_kl) > parameters.target_kl
                ):
                    stop = True
                    break
        epochs_completed += 1
        if stop:
            break
    averages = (totals / max(minibatches, 1)).cpu().tolist()
    return PPOUpdateMetrics(
        samples=sample_count,
        epochs_completed=epochs_completed,
        policy_loss=averages[0],
        value_loss=averages[1],
        entropy=averages[2],
        approximate_kl=averages[3],
        clip_fraction=averages[4],
        gradient_norm=averages[5],
    )
