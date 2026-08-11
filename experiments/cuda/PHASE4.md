# Phase 4: GPU-native PPO and NumPy deployment

Phase 4 connects the validated Phase 3 TensorFlight backend to a device-native
PPO loop. It is a training implementation and benchmark, not a claim that the
resulting short-run policy transfers to the official simulator. ActiveRocketPy
remains the scoring oracle for Phase 5.

## Agent boundary

The training path preserves the Phase 1 control split:

```text
LaunchPlanner (official deployment only)
  -> launch + inclination + heading
BootstrapController
  -> fixed historical angular-rate control
ControlHandoff
  -> latched from sensor-estimated altitude AGL
PPO PostLaunchController
  -> [roll, tvc_x, tvc_y, throttle]
```

`phase4_observation.py` consumes only `TensorAgentObservation`: simulation
time, balloon status/state, and the 12 official rocket sensor values. The
selector uses estimated position, and the handoff uses estimated GNSS altitude.
Neither accepts `TensorOracleState` or the internal 13-state rocket tensor.

The historical E2E estimator and 29-feature transform are ported to batched
PyTorch operations. Running normalization updates only rows for which
`controller_active & sensors_finite`; NaN launch sensors therefore never enter
its statistics. In Scenario 1 the historical 40 m AGL handoff occurred near
step 583 (about 5.86 s) in the measured bootstrap trace. This is an observed
boundary, not a hard-coded step: activation remains sensor-derived.

The shaped distance/ZEM/stability reward is training-only. Canonical balloon
pop reward is accumulated separately as official score so shaping cannot be
mistaken for competition performance.

## PPO semantics

`phase4_ppo.py` provides a separate 256x256 tanh actor and critic, a
tanh-squashed Gaussian policy, fixed device-side rollout storage, and clipped
PPO updates. Policy sampling and minibatch shuffling use dedicated device RNGs.

GAE deliberately uses two masks:

```text
bootstrap_mask  = not terminated
continuation    = not (terminated or truncated)
```

Thus a true terminal state does not bootstrap. A truncation includes the value
of its terminal observation, but its advantage does not flow into the reset
episode. A numerical unit test locks this distinction.

All observations, actions, rewards, values, advantages, environment state, and
policy parameters remain on the selected device. The Python loop checks one
done flag per environment step so it can trigger stochastic masked resets and
record episode metrics; it does not copy batched rollout data through NumPy or
subprocess IPC.

## Checkpoint and deployment artifacts

`training.ckpt` contains:

- actor, critic, log standard deviation, and optimizer state;
- observation-normalization and estimator/controller recurrent state;
- complete rocket, actuator, sensor, balloon, and stochastic-reset state;
- update/global-transition and per-environment episode counters;
- global PyTorch CPU/CUDA RNG plus dedicated action/shuffle/environment RNGs;
- config, scenario/source SHA-256 hashes, metrics, and elapsed progress.

Checkpoints are written atomically at PPO update boundaries. CPU and CUDA tests
save a checkpoint, restore a new trainer, run the next update, and require exact
model equality. Strict loading rejects a different environment count, horizon,
hidden size, seed/device, or source hashes.

`deployment.npz` is intentionally smaller: actor weights, normalization
statistics, launch/handoff/action metadata, and source hashes. The accompanying
`NumpyTensorFlightAgent` performs estimator, selector, 29-feature transform,
normalization, and MLP inference with NumPy only. Its forward output is tested
against the deterministic PyTorch actor. Official evaluation therefore does
not need PyTorch or CUDA.

The current leaderboard packer embeds only the configured Python agent source
in the submission JSON; it does not copy sibling files. For that path,
`--deployment-agent` generates a second, self-contained NumPy-only `.py` file
whose base64 payload is the exact same NPZ. The local/finals source bundle may
keep the auditable external NPZ, while leaderboard packing points
`agent_module_path` at the generated single file. The test suite dynamically
imports that generated file without the NPZ present and checks its forward
output against PyTorch.

## Measured 4x4 grid

The RTX 5070 Ti benchmark used one seed, one PPO epoch, a 65,536-sample maximum
minibatch, and 1,024 steps per environment. The run crossed the sensor-derived
handoff and exercised PPO, but its 10.24 seconds of simulated time produced no
complete episodes and no official score. Values below are end-to-end
transitions/s; parentheses are peak allocated CUDA MiB.

| Environments | H=16 | H=32 | H=64 | H=128 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 10,823 (137) | 10,767 (196) | 10,830 (303) | 10,845 (486) |
| 1,024 | 21,648 (206) | 21,652 (313) | 21,677 (496) | 21,645 (507) |
| 2,048 | 42,786 (333) | 42,756 (517) | 42,733 (529) | 42,744 (552) |
| 4,096 | 82,884 (556) | 83,308 (569) | 82,893 (593) | 83,198 (638) |

At 4,096 environments the complete environment plus one-epoch PPO loop stays
near 83k transitions/s. Horizon changes throughput by less than one percent in
this short sample, while memory grows with rollout/update batch size. This is
strong evidence that the full GPU loop is viable, but it is deliberately **not
a horizon selection**.

The benchmark records environment and end-to-end throughput, PPO samples/s,
official score, completed shaped return, active-policy shaped reward and their
wall-clock AUCs, time-to-score, and peak memory. If no episode completes, its
decision is `learning_signal_inconclusive`; neither TPS nor one short shaped
reward trace is allowed to select the PPO configuration. A real selection
requires longer, repeated seeds and a score/time-to-score signal.

## Reproduce

Run the tests:

```powershell
.venv\Scripts\python -m pytest tests\test_cuda_phase4_training.py -q
```

Run the specified environment/horizon grid:

```powershell
.venv\Scripts\python -m experiments.cuda.benchmark_phase4 `
  --num-envs 512,1024,2048,4096 --horizons 16,32,64,128 `
  --steps-per-env 1024 --epochs 1 --minibatch-size 65536 `
  --output .artifacts\cuda\phase4\grid.json
```

Train, checkpoint, and export a deployment artifact:

```powershell
.venv\Scripts\python -m experiments.cuda.phase4_training `
  --num-envs 4096 --horizon 64 --updates 100 `
  --checkpoint .artifacts\cuda\phase4\training.ckpt `
  --deployment .artifacts\cuda\phase4\deployment.npz `
  --deployment-agent .artifacts\cuda\phase4\submission_agent.py
```

Resume deterministically at the next update boundary:

```powershell
.venv\Scripts\python -m experiments.cuda.phase4_training `
  --num-envs 4096 --horizon 64 --updates 200 `
  --resume .artifacts\cuda\phase4\training.ckpt `
  --checkpoint .artifacts\cuda\phase4\training.ckpt `
  --deployment .artifacts\cuda\phase4\deployment.npz `
  --deployment-agent .artifacts\cuda\phase4\submission_agent.py
```

Use `phase4_eval_config.yaml` as a local official-evaluator template after the
artifact exists. Keep `leaderboard_submission: false` during development. For
an actual leaderboard pack, copy that template and change `agent_module_path`
to the generated `submission_agent.py`; no `artifact_path` kwarg is then
needed.

## Phase 4 decision

Phase 4 is **GO for review as a training stack**. The GPU-native observation,
selector, handoff, rollout, GAE/PPO, deterministic resume, and NumPy deployment
boundaries are implemented and tested. The tested machine supports the
requested 4,096 x 128 case with ample memory headroom.

It is **not yet GO as a competition policy**: this short engineering benchmark
did not complete an episode, improve official score, or validate transfer.
Those are explicitly Phase 5 gates. Until a repeated long-run benchmark yields
time-to-score, 4,096 x 64 remains only a neutral working default rather than a
selected optimum.
