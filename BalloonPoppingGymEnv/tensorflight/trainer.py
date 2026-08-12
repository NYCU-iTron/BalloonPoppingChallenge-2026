"""GPU-native TensorFlight PPO trainer, checkpoints, and deployment export."""

from __future__ import annotations

import argparse
import base64
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from BalloonPoppingGymEnv.tensorflight.balloons import (
    BalloonBatchParameters,
    quaternion_to_matrix,
)
from BalloonPoppingGymEnv.tensorflight.environment import TensorFlightEnvironment
from BalloonPoppingGymEnv.tensorflight.factory import (
    TensorFlightSource,
    build_scenario1_source,
    make_tensorflight_environment,
)
from BalloonPoppingGymEnv.tensorflight.observations import TensorAgentAdapter
from BalloonPoppingGymEnv.tensorflight.ppo import (
    DeviceRolloutBuffer,
    PPOHyperparameters,
    PPOUpdateMetrics,
    TanhActorCritic,
    update_ppo,
)


Tensor = torch.Tensor
CHECKPOINT_SCHEMA = 1
DEPLOYMENT_SCHEMA = 1


@dataclass(frozen=True)
class TensorFlightTrainingConfig:
    num_envs: int = 4096
    horizon: int = 64
    updates: int = 10
    hidden_size: int = 256
    handoff_altitude_agl: float = 40.0
    seed: int = 2121
    device: str = "cuda"
    score_threshold: float = 1.0
    ppo: PPOHyperparameters = field(default_factory=PPOHyperparameters)

    def validate(self) -> None:
        if self.num_envs < 1 or self.horizon < 1 or self.updates < 1:
            raise ValueError("num_envs, horizon, and updates must be positive")
        if self.hidden_size < 1 or self.handoff_altitude_agl < 0:
            raise ValueError("hidden size and handoff altitude must be non-negative")


@dataclass(frozen=True)
class RolloutMetrics:
    seconds: float
    transitions_per_second: float
    completed_episodes: int
    mean_official_score: float | None
    mean_shaped_return: float | None
    mean_episode_length: float | None
    mean_policy_shaped_reward: float | None
    official_pops_per_transition: float


@dataclass(frozen=True)
class TrainingUpdateMetrics:
    update: int
    global_transitions: int
    rollout: RolloutMetrics
    ppo: PPOUpdateMetrics
    update_seconds: float
    end_to_end_transitions_per_second: float
    reward_wallclock_auc: float
    shaped_return_wallclock_auc: float
    policy_reward_wallclock_auc: float
    time_to_score_seconds: float | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cpu_tree(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _copy(target: Tensor, value: object) -> None:
    target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))


def environment_state_dict(environment: TensorFlightEnvironment) -> dict[str, object]:
    balloon_parameters = {
        field.name: getattr(environment.balloons.parameters, field.name).clone()
        for field in fields(BalloonBatchParameters)
    }
    sampler = environment.balloons.reset_sampler
    sampler_state = (
        sampler.generator.get_state() if hasattr(sampler, "generator") else None
    )
    return {
        "rocket_state": environment.rocket_state.clone(),
        "rocket_elapsed": environment.rocket_elapsed.clone(),
        "terminated": environment.terminated.clone(),
        "truncated": environment.truncated.clone(),
        "impact_countdown": environment.impact_countdown.clone(),
        "reward": environment.reward.clone(),
        "closest_distance": environment.closest_distance.clone(),
        "hit_mask": environment.hit_mask.clone(),
        "cached_rhs": environment.cached_rhs.clone(),
        "rocket_sensors": environment.rocket_sensors.clone(),
        "actuator_output": environment.actuators.output.clone(),
        "balloons": {
            "previous_state": environment.balloons.previous_state.clone(),
            "current_state": environment.balloons.current_state.clone(),
            "next_state": environment.balloons.next_state.clone(),
            "status": environment.balloons.status.clone(),
            "current_step": environment.balloons.current_step.clone(),
            "parameters": balloon_parameters,
            "sampler_rng": sampler_state,
        },
        "sensors": {
            "gyro_drift": environment.sensor_suite.gyro_drift.clone(),
            "accel_drift": environment.sensor_suite.accel_drift.clone(),
            "rng": environment.sensor_suite.generator.get_state(),
        },
    }


def load_environment_state(
    environment: TensorFlightEnvironment, state: Mapping[str, object]
) -> None:
    for name in (
        "rocket_state",
        "rocket_elapsed",
        "terminated",
        "truncated",
        "impact_countdown",
        "reward",
        "closest_distance",
        "hit_mask",
        "cached_rhs",
        "rocket_sensors",
    ):
        _copy(getattr(environment, name), state[name])
    _copy(environment.actuators.output, state["actuator_output"])
    balloon_state = state["balloons"]
    if not isinstance(balloon_state, Mapping):
        raise TypeError("balloon checkpoint must be a mapping")
    for name in (
        "previous_state",
        "current_state",
        "next_state",
        "status",
        "current_step",
    ):
        _copy(getattr(environment.balloons, name), balloon_state[name])
    parameters = balloon_state["parameters"]
    if not isinstance(parameters, Mapping):
        raise TypeError("balloon parameter checkpoint must be a mapping")
    for item in fields(BalloonBatchParameters):
        _copy(
            getattr(environment.balloons.parameters, item.name), parameters[item.name]
        )
    rotation = quaternion_to_matrix(environment.balloons.parameters.quaternion)
    environment.balloons.dynamics.rotation.copy_(rotation)
    environment.balloons.dynamics.body_z.copy_(rotation[..., :, 2])
    sampler = environment.balloons.reset_sampler
    if balloon_state["sampler_rng"] is not None and hasattr(sampler, "generator"):
        sampler.generator.set_state(  # type: ignore[union-attr]
            torch.as_tensor(balloon_state["sampler_rng"], device="cpu")
        )
    sensors = state["sensors"]
    if not isinstance(sensors, Mapping):
        raise TypeError("sensor checkpoint must be a mapping")
    _copy(environment.sensor_suite.gyro_drift, sensors["gyro_drift"])
    _copy(environment.sensor_suite.accel_drift, sensors["accel_drift"])
    environment.sensor_suite.generator.set_state(
        torch.as_tensor(sensors["rng"], device="cpu")
    )


class TensorFlightTrainer:
    """Own the complete deterministic state of a native TensorFlight PPO run."""

    def __init__(
        self, source: TensorFlightSource, config: TensorFlightTrainingConfig
    ) -> None:
        config.validate()
        self.source = source
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but torch.cuda is unavailable")
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        self.environment = make_tensorflight_environment(
            source,
            config.num_envs,
            seed=config.seed + 100,
            device=self.device,
        )
        self.adapter = TensorAgentAdapter(
            config.num_envs,
            sampling_rate=source.sensor_config.sampling_rate,
            ground_elevation=source.elevation,
            handoff_altitude_agl=config.handoff_altitude_agl,
            device=self.device,
            dtype=torch.float32,
        )
        self.adapter.prepare(self.environment.observation())
        self.model = TanhActorCritic(hidden_size=config.hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.ppo.learning_rate, eps=1e-5
        )
        self.rollout = DeviceRolloutBuffer(
            config.horizon, config.num_envs, device=self.device
        )
        self.action_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 200
        )
        self.shuffle_generator = torch.Generator(device=self.device).manual_seed(
            config.seed + 300
        )
        self.update_index = 0
        self.global_transitions = 0
        self.episode_counter = torch.zeros(
            config.num_envs, device=self.device, dtype=torch.int64
        )
        self.episode_shaped_return = torch.zeros(config.num_envs, device=self.device)
        self.episode_official_score = torch.zeros_like(self.episode_shaped_return)
        self.episode_length = torch.zeros_like(self.episode_counter)
        self.history: list[dict[str, float | int | None]] = []
        self.reward_wallclock_auc = 0.0
        self.shaped_return_wallclock_auc = 0.0
        self.policy_reward_wallclock_auc = 0.0
        self.time_to_score_seconds: float | None = None
        self._last_metric_time = 0.0
        self._last_mean_score = 0.0
        self._last_mean_shaped_return = 0.0
        self._last_policy_reward_time = 0.0
        self._last_mean_policy_reward = 0.0
        self._started = time.perf_counter()
        self._elapsed_before_resume = 0.0

    @property
    def elapsed(self) -> float:
        return self._elapsed_before_resume + (time.perf_counter() - self._started)

    def _record_completed(
        self, done: Tensor
    ) -> tuple[int, float | None, float | None, float | None]:
        count = int(done.sum().item())
        if count == 0:
            return 0, None, None, None
        mean_score = float(self.episode_official_score[done].mean().item())
        mean_shaped = float(self.episode_shaped_return[done].mean().item())
        mean_length = float(self.episode_length[done].to(torch.float32).mean().item())
        now = self.elapsed
        delta_time = now - self._last_metric_time
        self.reward_wallclock_auc += (
            (self._last_mean_score + mean_score) * 0.5 * delta_time
        )
        self.shaped_return_wallclock_auc += (
            (self._last_mean_shaped_return + mean_shaped) * 0.5 * delta_time
        )
        self._last_metric_time = now
        self._last_mean_score = mean_score
        self._last_mean_shaped_return = mean_shaped
        if (
            self.time_to_score_seconds is None
            and mean_score >= self.config.score_threshold
        ):
            self.time_to_score_seconds = now
        self.history.append(
            {
                "wall_seconds": now,
                "episodes": count,
                "mean_official_score": mean_score,
                "mean_shaped_return": mean_shaped,
                "mean_episode_length": mean_length,
            }
        )
        return count, mean_score, mean_shaped, mean_length

    def collect_rollout(self) -> RolloutMetrics:
        completed = 0
        score_total = 0.0
        shaped_total = 0.0
        length_total = 0.0
        rollout_official_pop_total = torch.zeros((), device=self.device)
        self.model.eval()
        _synchronize(self.device)
        started = time.perf_counter()
        for index in range(self.config.horizon):
            prepared = self.adapter.current
            if prepared is None:
                raise RuntimeError("agent adapter has no current observation")
            normalized_action, latent, log_probability, value = self.model.act(
                prepared.normalized, generator=self.action_generator
            )
            policy_control = self.adapter.physical_action(
                normalized_action,
                max_roll_torque=self.source.max_roll_torque,
                max_gimbal_angle=self.source.max_gimbal_angle,
                throttle_low=self.source.throttle_low,
                throttle_high=self.source.throttle_high,
            )
            applied_control = self.adapter.select_control(policy_control)
            self.rollout.observations[index].copy_(prepared.normalized)
            self.rollout.latent_actions[index].copy_(latent)
            self.rollout.log_probabilities[index].copy_(log_probability)
            self.rollout.values[index].copy_(value)
            self.rollout.policy_active[index].copy_(prepared.controller_active)

            transition = self.environment.step(applied_control)
            self.adapter.previous_action.copy_(applied_control)
            next_prepared = self.adapter.prepare(transition.observation)
            shaped_reward = self.adapter.reward_model.compute(
                transition.reward,
                self.adapter.estimator.features,
                next_prepared.target_index,
                next_prepared.target_available,
                next_prepared.target_state,
                transition.terminated,
                normalized_action,
                next_prepared.sin_alpha,
                next_prepared.sin_beta,
            )
            with torch.no_grad():
                bootstrap_value = self.model.value(next_prepared.normalized)
            self.rollout.rewards[index].copy_(shaped_reward)
            self.rollout.bootstrap_values[index].copy_(bootstrap_value)
            self.rollout.terminated[index].copy_(transition.terminated)
            self.rollout.truncated[index].copy_(transition.truncated)

            self.episode_shaped_return += shaped_reward
            self.episode_official_score += transition.reward
            rollout_official_pop_total += transition.reward.sum()
            self.episode_length += 1
            done = transition.terminated | transition.truncated
            event_count, mean_score, mean_shaped, mean_length = self._record_completed(
                done
            )
            if event_count:
                completed += event_count
                score_total += float(mean_score) * event_count
                shaped_total += float(mean_shaped) * event_count
                length_total += float(mean_length) * event_count
                self.episode_counter[done] += 1
                self.episode_shaped_return[done] = 0
                self.episode_official_score[done] = 0
                self.episode_length[done] = 0
                reset_observation = self.environment.reset(done)
                self.adapter.reset(done)
                self.adapter.prepare(reset_observation)
        _synchronize(self.device)
        seconds = time.perf_counter() - started
        transitions = self.rollout.transition_count
        policy_count = int(self.rollout.policy_active.sum().item())
        mean_policy_reward = (
            float(self.rollout.rewards[self.rollout.policy_active].mean().item())
            if policy_count
            else None
        )
        if mean_policy_reward is not None:
            now = self.elapsed
            delta_time = now - self._last_policy_reward_time
            self.policy_reward_wallclock_auc += (
                (self._last_mean_policy_reward + mean_policy_reward) * 0.5 * delta_time
            )
            self._last_policy_reward_time = now
            self._last_mean_policy_reward = mean_policy_reward
        return RolloutMetrics(
            seconds=seconds,
            transitions_per_second=transitions / seconds,
            completed_episodes=completed,
            mean_official_score=(score_total / completed if completed else None),
            mean_shaped_return=(shaped_total / completed if completed else None),
            mean_episode_length=(length_total / completed if completed else None),
            mean_policy_shaped_reward=mean_policy_reward,
            official_pops_per_transition=(
                float(rollout_official_pop_total.item()) / transitions
            ),
        )

    def train_update(self) -> TrainingUpdateMetrics:
        _synchronize(self.device)
        started = time.perf_counter()
        rollout_metrics = self.collect_rollout()
        self.model.train()
        ppo_metrics = update_ppo(
            self.model,
            self.optimizer,
            self.rollout,
            self.config.ppo,
            shuffle_generator=self.shuffle_generator,
        )
        _synchronize(self.device)
        update_seconds = time.perf_counter() - started
        self.update_index += 1
        self.global_transitions += self.rollout.transition_count
        result = TrainingUpdateMetrics(
            update=self.update_index,
            global_transitions=self.global_transitions,
            rollout=rollout_metrics,
            ppo=ppo_metrics,
            update_seconds=update_seconds,
            end_to_end_transitions_per_second=(
                self.rollout.transition_count / update_seconds
            ),
            reward_wallclock_auc=self.reward_wallclock_auc,
            shaped_return_wallclock_auc=self.shaped_return_wallclock_auc,
            policy_reward_wallclock_auc=self.policy_reward_wallclock_auc,
            time_to_score_seconds=self.time_to_score_seconds,
        )
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "config": asdict(self.config),
            "source_hashes": self.source.source_hashes,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment": environment_state_dict(self.environment),
            "adapter": self.adapter.state_dict(),
            "progress": {
                "update_index": self.update_index,
                "global_transitions": self.global_transitions,
                "episode_counter": self.episode_counter.clone(),
                "episode_shaped_return": self.episode_shaped_return.clone(),
                "episode_official_score": self.episode_official_score.clone(),
                "episode_length": self.episode_length.clone(),
                "history": self.history,
                "reward_wallclock_auc": self.reward_wallclock_auc,
                "shaped_return_wallclock_auc": self.shaped_return_wallclock_auc,
                "policy_reward_wallclock_auc": self.policy_reward_wallclock_auc,
                "time_to_score_seconds": self.time_to_score_seconds,
                "last_metric_time": self._last_metric_time,
                "last_mean_score": self._last_mean_score,
                "last_mean_shaped_return": self._last_mean_shaped_return,
                "last_policy_reward_time": self._last_policy_reward_time,
                "last_mean_policy_reward": self._last_mean_policy_reward,
                "elapsed": self.elapsed,
            },
            "rng": {
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
                "action": self.action_generator.get_state(),
                "shuffle": self.shuffle_generator.get_state(),
            },
        }

    def save_checkpoint(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(_cpu_tree(self.state_dict()), temporary)
        temporary.replace(destination)

    def load_checkpoint(self, path: str | Path, *, strict_hashes: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported TensorFlight checkpoint schema")
        if strict_hashes and checkpoint["source_hashes"] != self.source.source_hashes:
            raise ValueError("checkpoint source hashes do not match this checkout")
        saved_config = dict(checkpoint["config"])
        current_config = asdict(self.config)
        # ``updates`` is only the requested stopping point and may be extended
        # on resume. Every value that changes trajectories or PPO math must be
        # identical for deterministic continuation.
        saved_config.pop("updates", None)
        current_config.pop("updates", None)
        if saved_config != current_config:
            raise ValueError("checkpoint training config does not match this run")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        load_environment_state(self.environment, checkpoint["environment"])
        self.adapter.load_state_dict(checkpoint["adapter"])
        progress = checkpoint["progress"]
        self.update_index = int(progress["update_index"])
        self.global_transitions = int(progress["global_transitions"])
        for target, name in (
            (self.episode_counter, "episode_counter"),
            (self.episode_shaped_return, "episode_shaped_return"),
            (self.episode_official_score, "episode_official_score"),
            (self.episode_length, "episode_length"),
        ):
            _copy(target, progress[name])
        self.history = list(progress["history"])
        self.reward_wallclock_auc = float(progress["reward_wallclock_auc"])
        self.shaped_return_wallclock_auc = float(
            progress["shaped_return_wallclock_auc"]
        )
        self.policy_reward_wallclock_auc = float(
            progress["policy_reward_wallclock_auc"]
        )
        self.time_to_score_seconds = progress["time_to_score_seconds"]
        self._last_metric_time = float(progress["last_metric_time"])
        self._last_mean_score = float(progress["last_mean_score"])
        self._last_mean_shaped_return = float(progress["last_mean_shaped_return"])
        self._last_policy_reward_time = float(progress["last_policy_reward_time"])
        self._last_mean_policy_reward = float(progress["last_mean_policy_reward"])
        self._elapsed_before_resume = float(progress["elapsed"])
        self._started = time.perf_counter()
        rng = checkpoint["rng"]
        self.action_generator.set_state(rng["action"])
        self.shuffle_generator.set_state(rng["shuffle"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    def export_deployment(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        actor_layers = [
            module for module in self.model.actor if isinstance(module, torch.nn.Linear)
        ]
        variance = self.adapter.normalizer.variance.detach().cpu().numpy()
        arrays: dict[str, np.ndarray] = {
            "schema_version": np.asarray(DEPLOYMENT_SCHEMA, dtype=np.int64),
            "normalizer_mean": self.adapter.normalizer.mean.detach().cpu().numpy(),
            "normalizer_variance": variance,
            "normalizer_epsilon": np.asarray(
                self.adapter.normalizer.epsilon, dtype=np.float64
            ),
            "normalizer_clip": np.asarray(
                self.adapter.normalizer.clip, dtype=np.float64
            ),
            "launch_attitude": np.asarray((90.0, 0.0), dtype=np.float64),
            "launch_time": np.asarray(0.01, dtype=np.float64),
            "handoff_altitude_agl": np.asarray(
                self.config.handoff_altitude_agl, dtype=np.float64
            ),
            "max_roll_torque": np.asarray(
                self.source.max_roll_torque, dtype=np.float64
            ),
            "max_gimbal_angle": np.asarray(
                self.source.max_gimbal_angle, dtype=np.float64
            ),
            "throttle_range": np.asarray(
                (self.source.throttle_low, self.source.throttle_high),
                dtype=np.float64,
            ),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "schema_version": DEPLOYMENT_SCHEMA,
                        "observation_size": self.model.observation_size,
                        "action_size": self.model.action_size,
                        "hidden_size": self.model.hidden_size,
                        "training_update": self.update_index,
                        "global_transitions": self.global_transitions,
                        "training_seed": self.config.seed,
                        "training_num_envs": self.config.num_envs,
                        "training_horizon": self.config.horizon,
                        "source_hashes": self.source.source_hashes,
                    },
                    sort_keys=True,
                )
            ),
        }
        for index, layer in enumerate(actor_layers):
            arrays[f"actor_weight_{index}"] = layer.weight.detach().cpu().numpy()
            arrays[f"actor_bias_{index}"] = layer.bias.detach().cpu().numpy()
        np.savez(destination, **arrays)

    def export_self_contained_agent(
        self, path: str | Path, *, deployment_path: str | Path
    ) -> None:
        """Embed a deployment NPZ in one NumPy-only submission source file."""
        artifact = Path(deployment_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"deployment artifact does not exist: {artifact}")
        template = (
            Path(__file__).resolve().parents[1]
            / "agents"
            / "numpy_tensorflight_agent.py"
        )
        source = template.read_text(encoding="utf-8")
        marker = 'EMBEDDED_DEPLOYMENT_BASE64 = ""'
        if source.count(marker) != 1:
            raise RuntimeError("deployment agent embed marker changed")
        encoded = base64.b64encode(artifact.read_bytes()).decode("ascii")
        chunks = [encoded[index : index + 88] for index in range(0, len(encoded), 88)]
        replacement = "EMBEDDED_DEPLOYMENT_BASE64 = (\n"
        replacement += "".join(f"    {chunk!r}\n" for chunk in chunks)
        replacement += ")"
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            source.replace(marker, replacement), encoding="utf-8", newline="\n"
        )


def _metrics_dict(metrics: TrainingUpdateMetrics) -> dict[str, object]:
    return asdict(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=65_536)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2121)
    parser.add_argument("--checkpoint", type=Path, default=Path("training.ckpt"))
    parser.add_argument("--deployment", type=Path, default=Path("deployment.npz"))
    parser.add_argument(
        "--deployment-agent",
        type=Path,
        help="optional self-contained NumPy agent source for leaderboard packing",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    args = parser.parse_args()
    ppo = PPOHyperparameters(epochs=args.epochs, minibatch_size=args.minibatch_size)
    config = TensorFlightTrainingConfig(
        num_envs=args.num_envs,
        horizon=args.horizon,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
        ppo=ppo,
    )
    source = build_scenario1_source(seed=args.seed)
    trainer = TensorFlightTrainer(source, config)
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    remaining = max(config.updates - trainer.update_index, 0)
    for _ in range(remaining):
        metrics = trainer.train_update()
        print(json.dumps(_metrics_dict(metrics), sort_keys=True))
        if trainer.update_index % args.checkpoint_every == 0:
            trainer.save_checkpoint(args.checkpoint)
    trainer.save_checkpoint(args.checkpoint)
    trainer.export_deployment(args.deployment)
    if args.deployment_agent is not None:
        trainer.export_self_contained_agent(
            args.deployment_agent, deployment_path=args.deployment
        )


if __name__ == "__main__":
    main()
