# Phase 3: online 100-balloon tensor environment

Phase 3 combines the Phase 2 rocket with an online batched balloon world,
official-shaped observations, parameter-driven effects, swept collision
geometry, rewards, and episode lifecycle. ActiveRocketPy remains the oracle;
this backend is for training and is not substituted into official evaluation.

## Boundary and storage model

The hot path in `phase3_balloon.py`, `phase3_effects.py`, and
`phase3_environment.py` imports PyTorch only. Canonical capture and comparison
remain isolated in `phase3_oracle.py`.

`TensorAgentObservation` exposes only the official fields:

- simulation time;
- balloon status;
- balloon six-state;
- 12 rocket sensor values.

The 13-state rocket value is available only through the separate diagnostic
`TensorOracleState`; it is not included in a training step. Phase 1's typed
estimator/selector/handoff contracts remain the agent-facing boundary.

`TensorBalloonWorld` stores only `previous_state`, `current_state`, and
`next_state`, each `[B, N, 6]`, plus status, release schedule, realized
stochastic parameters, and per-environment time. It never allocates a
`[B,N,T,6]` history. At B=4,096 and N=100, the three float32 state buffers use
about 28.1 MiB; an official-style 15,001-sample history would use about
137.3 GiB before other state. `Scenario1BalloonSampler` keeps a device RNG and
can replace realized parameters on masked episode resets.

## Balloon-only fidelity gate

Seed 2071 compared all 100 realized canonical balloons over all 14,999
available transitions (149.99 seconds) with one RK4 step per 0.01 seconds.

| Metric | float64 | float32 |
| --- | ---: | ---: |
| position p99 | 0.01759 m | 0.04897 m |
| position maximum | 0.01940 m | 0.09679 m |
| velocity p99 | 0.00450 m/s | 0.00444 m/s |
| release-time maximum | 0 s | 0 s |
| status mismatches | 0 | 0 |

The isolated velocity maxima (0.761 m/s in both dtypes) occur at the canonical
rail/free phase discontinuity; full-horizon p99 stays below 0.005 m/s. The
float32 position p99 passes the 0.05 m production gate, but with less margin
than float64. Phase 5 must therefore repeat this gate on holdout seeds.

## Combined transfer gate

Seed 2072 used 100 balloons and a deterministic changing open-loop controller
until canonical impact/termination. The oracle-compatibility column uses the
Phase 2 cached-FSAL path in float64. The production candidate uses one-step
RK4 in float32 on CUDA.

| p99 error | float64 cached FSAL | float32 RK4 |
| --- | ---: | ---: |
| rocket position | 0.000047 m | 0.01673 m |
| rocket velocity | 0.000064 m/s | 0.01298 m/s |
| rocket attitude | 0.000248 deg | 0.05562 deg |
| balloon position | 0.01521 m | 0.01533 m |
| closest distance | 0.00527 m | 0.01247 m |
| sensor absolute component | 0.0000085 | 0.01453 |

Both candidates had zero reward, balloon-status, and termination mismatches.
For float32 the measured p99.9 closest-distance error is 0.01832 m, so the
specified ambiguity rule gives:

```text
max(0.05 m, 0.01832 m + 0.01 m) = 0.05 m
```

Pure float32 therefore passes the current closest-distance gate. Computing
only collision geometry in float64 remains a switchable option, but it did not
improve the state-originated error enough to justify selecting it.

The transfer test also locks a pinned ActiveRocketPy lifecycle detail. Impact
is root-located and the rocket state freezes, but `terminated=True` appears
only after two subsequent phase-advance calls. TensorFlight models this
countdown explicitly instead of ending on the first below-ground endpoint.

## Scenario 4 readiness hooks

No unpublished Scenario 4 values are assumed. The backend accepts:

- per-environment altitude gust profiles shared by rocket and balloons;
- gyro and accelerometer white noise, random walk, bias, and mount position;
- GNSS position/altitude/velocity accuracy and mount position;
- optional actuator low-pass constants, followed by rate limits and clamps;
- independent stochastic balloon parameters on reset.

The sensor path intentionally reproduces the pinned ActiveRocketPy
accelerometer gravity lookup, including its current use of `u[3]`. If the
official submodule changes, the oracle fixture and compatibility behavior must
be reviewed together rather than silently changing training semantics.

## End-to-end throughput gate

The benchmark includes rocket RK4, 100 online balloons, actuators, sensors,
swept distance, reward, impact lifecycle, and observation assembly. Results
are medians of three eager runs on the RTX 5070 Ti / Ryzen 9800X3D machine.

| Environments | CPU float32 | CUDA float32 | CUDA mixed-distance | CUDA float64 |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3,563 | 1,437 | 1,430 | 1,466 |
| 512 | 14,862 | 11,643 | 11,665 | 11,655 |
| 2,048 | 24,034 | 46,174 | 45,867 | 44,986 |
| 4,096 | 25,373 | 88,523 | 87,112 | 79,297 |

Units are complete environment transitions per second. CUDA is still slower
at small batches and reaches only 1.92x at B=2,048. At B=4,096, float32 is
3.49x the matching tensor CPU baseline and passes the 2x gate. Its measured
peak allocated CUDA memory was 248.6 MiB for this short rollout; Phase 4 must
add PPO rollout-buffer memory before selecting horizon.

Decision:

- production candidate: pure float32, B=4,096;
- B <= 2,048 on this eager Windows build: retain the tensor CPU fallback until
  a Phase 4 learning-throughput result justifies CUDA;
- fidelity failure on holdout seeds: fall back to float64/mixed or canonical
  ActiveRocketPy actors rather than weakening the gate.

## Reproduce

```powershell
.venv\Scripts\python -m pytest tests\test_cuda_phase3_environment.py -q

.venv\Scripts\python -m experiments.cuda.phase3_oracle `
  --seed 2071 --num-balloons 100 --substeps 1 --dtype float32 `
  --output .artifacts\cuda\phase3\balloons_seed2071_n100_f32.json

.venv\Scripts\python -m experiments.cuda.phase3_oracle `
  --seed 2072 --num-balloons 100 --steps 512 --full-environment `
  --dtype float32 --device cuda --integrator rk4 `
  --output .artifacts\cuda\phase3\combined_seed2072_n100_f32_rk4_cuda.json

.venv\Scripts\python -m experiments.cuda.benchmark_phase3 `
  --batch-sizes 64,512,2048 --num-balloons 100 `
  --steps 5 --warmup-steps 2 --repeats 3 `
  --output .artifacts\cuda\phase3\throughput_seed2081.json

.venv\Scripts\python -m experiments.cuda.benchmark_phase3 `
  --batch-sizes 4096 --num-balloons 100 `
  --steps 8 --warmup-steps 3 --repeats 3 `
  --output .artifacts\cuda\phase3\throughput_seed2081_b4096.json
```

## Phase 3 decision

Phase 3 is **GO for review**. Balloon-only fidelity, combined
rocket/balloon/event fidelity, float32 production gating, stochastic reset,
and the B=4,096 CUDA throughput gate pass. The backend is not yet an RL
training loop: rollout storage, GAE, deterministic checkpoints, and NumPy
deployment export remain Phase 4.
