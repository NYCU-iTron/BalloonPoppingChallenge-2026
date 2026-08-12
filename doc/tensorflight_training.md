# TensorFlight GPU training

TensorFlight is the repository's batched training backend for Scenario 1. It
runs the rocket, 100 online balloons, observation transform, PPO rollout, and
policy update on one PyTorch device. ActiveRocketPy remains the source of truth:
always validate exported policies in the official environment.

## Install and verify CUDA

```shell
uv sync --extra tensorflight
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU')"
```

If `torch.cuda.is_available()` is false, install the CUDA build recommended by
the PyTorch installation selector for the machine's driver, then run the check
again. A CPU PyTorch wheel can train correctly, but `--device cuda` cannot use
it.

## Train

The command below creates one self-contained run directory. Checkpoints,
deployment weights, and the submission agent all appear there; no file needs to
be copied out of an experiment directory.

```shell
uv run balloon-tensorflight train \
  --output-dir runs/tensorflight/scenario1 \
  --num-envs 4096 \
  --horizon 64 \
  --updates 100 \
  --device cuda
```

On PowerShell, replace each trailing `\` with a backtick. The outputs are:

- `training.ckpt`: deterministic-resume checkpoint, including optimizer,
  normalizer, environment, progress, RNG, config, and source hashes.
- `checkpoints/update_XXXXXX.ckpt`: retained candidates. PPO can regress late
  in a run, so do not assume the final update is the best policy.
- `deployment.npz`: NumPy inference weights.
- `submission_agent.py`: one-file agent with the deployment embedded.

Resume by increasing `--updates` and passing the checkpoint:

```shell
uv run balloon-tensorflight train \
  --output-dir runs/tensorflight/scenario1 \
  --updates 200 \
  --resume runs/tensorflight/scenario1/training.ckpt \
  --device cuda
```

All trajectory-defining options except the stopping update must match the
checkpoint. A mismatch is rejected instead of silently changing the run.

Select a retained checkpoint with official holdout scores, then export the
winner in one command:

```shell
uv run balloon-tensorflight select \
  runs/tensorflight/scenario1/checkpoints/update_000080.ckpt \
  runs/tensorflight/scenario1/checkpoints/update_000090.ckpt \
  runs/tensorflight/scenario1/training.ckpt \
  --scenario 1 \
  --seeds 3201,3203,3207 \
  --output-dir runs/tensorflight/scenario1/best
```

Selection ranks mean official score, then maximum score, then prefers the
earlier update on a tie. It writes `selection.json`, `deployment.npz`, and the
self-contained agent. Training-environment reward is not used for this choice.

## Python API

```python
from BalloonPoppingGymEnv.tensorflight import (
    PPOHyperparameters,
    TensorFlightTrainer,
    TensorFlightTrainingConfig,
    build_scenario1_source,
)

config = TensorFlightTrainingConfig(
    num_envs=4096,
    horizon=64,
    updates=100,
    device="cuda",
    ppo=PPOHyperparameters(epochs=4, minibatch_size=65_536),
)
source = build_scenario1_source(seed=config.seed)
trainer = TensorFlightTrainer(source, config)

for _ in range(config.updates):
    metrics = trainer.train_update()
    print(metrics.rollout.transitions_per_second)

trainer.save_checkpoint("runs/tensorflight/training.ckpt")
trainer.export_deployment("runs/tensorflight/deployment.npz")
trainer.export_self_contained_agent(
    "runs/tensorflight/submission_agent.py",
    deployment_path="runs/tensorflight/deployment.npz",
)
```

The public policy path only receives official observation fields. True
`rocket_states` remain internal to the physics backend and are excluded from
the estimator, target selector, handoff, observation builder, and deployment
agent.

## Validate in ActiveRocketPy

`evaluate` uses the unmodified official environment and NumPy inference; no
TensorFlight dynamics participate:

```shell
uv run balloon-tensorflight evaluate \
  runs/tensorflight/scenario1/deployment.npz \
  --scenario 1 \
  --seeds 3201,3203,3207
```

Add `--require-hit` for a CI smoke gate. A non-zero result proves the deployment
can hit a balloon in ActiveRocketPy, but it is not a leaderboard-performance
claim. Use held-out Scenario 1 seeds and report their complete score
distribution for competitive conclusions.

The repository includes a small pretrained smoke artifact. It was trained on
the GPU TensorFlight backend and scores 2 in the unmodified Scenario 0 seed 0
environment. Run:

```shell
uv run balloon-tensorflight evaluate --scenario 0 --seeds 0 --require-hit
```

This artifact demonstrates the training/export/official-simulator path. It is
not presented as a Scenario 1 agent; Scenario 1 seed 3201 scored 0 in the
release check.

## Use the deployment agent directly

```python
from BalloonPoppingGymEnv.agents.numpy_tensorflight_agent import TensorFlightAgent

agent = TensorFlightAgent(given_parameters, artifact_path="deployment.npz")
action = agent.get_action(observation)
```

Omit `artifact_path` to use the packaged smoke artifact. Inference uses NumPy
only, so the official evaluation machine does not need PyTorch or CUDA.

## Current scope

- TensorFlight is parameterized for future sensor effects, gust, and actuator
  lag, but only published Scenario 1 parameters are claimed as validated.
- Scenario 4 parameters are not guessed. When its YAML is published, add the
  mapping and rerun rocket, balloon, event, dtype, and official-transfer gates.
- The fallback order remains GPU TensorFlight, CPU TensorFlight, then official
  ActiveRocketPy actors with a separate learner.
