# CUDA flight-environment experiment

This directory answers a narrow question: can the E2E flight environment gain
useful training throughput from CUDA while remaining transferable to the
official ActiveRocketPy simulator?

The answer from the first experiment is **not by moving one RocketPy flight to
the GPU**.  The useful architecture is a single process advancing a large
tensor batch of independent flights.  Among the sampled batch sizes on the
tested Windows machine, eager CUDA did not overtake tensorized CPU execution
until batch 4,096.

The official simulator remains canonical. Phase 3 now provides a validated
competition-specific training surrogate for Scenario 1, but its policies must
still pass holdout transfer in the unmodified official simulator before any
competition-performance claim.

## Why one balloon still runs on the CPU

The historical `e2e` and `improve-rl-navigator` branches both launch 20
`SubprocVecEnv` workers and set Stable-Baselines3 PPO to `device="cpu"`.  The
E2E wrapper does reduce the scene to one balloon, but each worker still owns a
separate Python/NumPy/SciPy ActiveRocketPy `Flight`.  Changing PPO to
`device="cuda"` would move policy evaluation and optimization; it would not
move those 20 flight simulations.

There is also a separate setup issue in the historical E2E branch: its Windows
lock resolves the ordinary PyPI PyTorch wheel, while both training scripts
explicitly request `device="cpu"`.  For this experiment, a CUDA 13.0 PyTorch
wheel was installed separately and `torch.cuda.is_available()` plus an actual
GPU matrix operation were verified.  Fixing those two items enables a CUDA
policy, but still does not make ActiveRocketPy a CUDA environment.

Official v0.1.1 advances a 13-state flight with SciPy RK45 at a 0.01-second
control interval.  Its RHS still traverses RocketPy's Python objects,
interpolators, aerodynamics, motor properties, controllers, sensors, and event
bookkeeping.  SciPy's current `solve_ivp` does not accept a CUDA Array API
backend, so this graph cannot be moved merely by changing an array device.

## Measurements

Hardware and software:

- AMD Ryzen 7 9800X3D, 8 cores / 16 logical processors
- NVIDIA GeForce RTX 5070 Ti 16 GB
- Windows 11, Python 3.12
- PyTorch 2.12.0+cu130, CUDA runtime 13.0
- repository `main`/`cuda` base `55d8d73`, ActiveRocketPy `473447d`

### Canonical one-balloon profile

Three 500-step samples from `profile_official_env.py`:

| Measurement | Median |
| --- | ---: |
| reset | 0.001233 s |
| launch / flight initialization | 0.020081 s |
| in-flight control steps | 434.8 steps/s |
| RK RHS evaluations | 6.02 per control step |

`cProfile` places almost all in-flight time below
`Flight.step_simulation`, principally the 6-DoF RHS, SciPy RK stepping, and
RocketPy scalar interpolation/evaluation.  This confirms that the one-balloon
path is flight-bound rather than balloon-reset-bound.

### Physics-only subprocess scaling

A representative 300-step steady-state sweep, with every child advancing all
steps internally:

| Processes | Aggregate canonical steps/s |
| ---: | ---: |
| 1 | 477 |
| 2 | 931 |
| 4 | 1,871 |
| 8 | 3,166 |
| 12 | 3,437 |
| 16 | 4,083 |
| 20 | 3,527 |

Two additional 500-step comparisons kept 16 processes near 4,000 steps/s;
20 processes varied between about 3,300 and 3,900 steps/s.  Thus 20 did not
beat 16 in this physics-only sample.  This benchmark omits `SubprocVecEnv`'s
per-control-step action/observation IPC and synchronization barrier, so the
actual PPO worker count still requires an end-to-end sweep.

### Non-canonical batched tensor surrogate

This result includes one fixed RK4 transition (`dt=0.01`), simplified 6-DoF,
four **post-launch** actions `[roll, tvc_x, tvc_y, throttle]`, actuator lag,
official-style independent swept-segment hit geometry for one moving target,
reward, done masks, and masked reset.  State and outputs remain on the selected
device.

| Batch | Tensor CPU transitions/s | Eager CUDA transitions/s |
| ---: | ---: | ---: |
| 1 | 1,113 | 247 |
| 20 | 20,485 | 4,910 |
| 64 | 62,866 | 15,713 |
| 256 | 205,994 | 62,687 |
| 1,024 | 489,199 | 250,419 |
| 4,096 | 638,261 | 1,002,237 |

At batch 20, CUDA is about four times slower because many small kernels pay
launch overhead.  At the sampled batch 4,096, CUDA is about 1.5 times faster
than tensor CPU.  These numbers show a batching crossover region for this
simplified model; they are **not a speedup comparison against a
RocketPy-equivalent model**.  The table contains medians from three seeded
repeats; the benchmark also prints each combination's minimum and maximum.

`TensorFlightBatch` now supports device-side masked reset and protects state
from non-finite policy actions. Phase 1 adds an isolated historical 29-element
E2E observation fixture and agent/oracle data contracts, but they are not yet
wired into `TensorFlightBatch`. There is still no PPO rollout buffer/adapter or
canonical dynamics and reward parity, so it is not a drop-in trainable
environment.

Launch is deliberately not part of the four-action controller. A
`LaunchPlanner` owns only launch/inclination/heading, a post-launch bootstrap
controller owns the historical climb, and a sensor-estimated handoff activates
PPO. See [PHASE1.md](PHASE1.md) for the canonical trace schema, measured launch
boundary, comparison report, and synchronous IPC baseline.

Phase 2 adds a separate float64 `Scenario1TensorRocket` that ports the
official generalized 6-DoF RHS, matches Scenario 1 actuator saturation/rate
limits, and compares RK4 1/2/4 with a Dormand--Prince reference. It also makes
the official RK45 stale-FSAL behavior at control discontinuities explicit.
Phase 3 wires that exact rocket into an online 100-balloon environment with
official-shaped observations, stochastic reset, gust/sensor/actuator hooks,
swept pops, rewards, and canonical impact lifecycle. A 4,096-environment
float32 run reached 88,523 transitions/s, 3.49x the matching tensor CPU
baseline, while its combined closest-distance p99 error was 0.0125 m. See
[PHASE2.md](PHASE2.md) for rocket parity and [PHASE3.md](PHASE3.md) for the
balloon, dtype, transfer, and throughput gates.

`torch.compile(mode="reduce-overhead")` safely fell back to eager execution on
this native Windows installation with `TritonMissing`.  A Linux/WSL2 run is a
separate experiment and may move the crossover through kernel fusion and CUDA
graphs.

## Reproduce

Verify that the environment has a CUDA-enabled PyTorch build:

```powershell
.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.12.0
.venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Run the canonical single-balloon profile:

```powershell
.venv\Scripts\python -m experiments.cuda.profile_official_env --steps 500 --repeats 3
.venv\Scripts\python -m experiments.cuda.profile_official_env --steps 300 --repeats 1 --profile
```

Run the current subprocess baseline:

```powershell
.venv\Scripts\python -m experiments.cuda.benchmark_cpu_parallel --steps 300 --workers 1,2,4,8,12,16,20
```

Record and compare a canonical Phase 1 trace, then run the synchronous
rollout-collection baseline:

```powershell
.venv\Scripts\python -m experiments.cuda.canonical_oracle --case launch_control --output .artifacts\cuda\oracle\launch_control.npz
.venv\Scripts\python -m experiments.cuda.compare_canonical_trace .artifacts\cuda\oracle\launch_control.npz --output .artifacts\cuda\oracle\launch_control_report.json
.venv\Scripts\python -m experiments.cuda.benchmark_cpu_training_loop --workers 1,4,8,16,20 --steps 300 --warmup-steps 10 --repeats 3
```

Run the isolated tensor benchmark and its tests:

```powershell
.venv\Scripts\python -m experiments.cuda.benchmark_tensor_flight
.venv\Scripts\python -m experiments.cuda.benchmark_tensor_flight --device cuda --batch-sizes 4096 --compile
.venv\Scripts\python -m pytest tests/test_tensor_flight_cuda_experiment.py -q
```

Run the Phase 2 rocket fidelity gates and generate the formal report:

```powershell
.venv\Scripts\python -m pytest tests/test_cuda_phase2_fidelity.py -q
.venv\Scripts\python -m experiments.cuda.phase2_oracle --steps 256 --seed 2031 --output .artifacts\cuda\phase2\report_seed2031.json
```

Run the Phase 3 online-world gates and formal reports:

```powershell
.venv\Scripts\python -m pytest tests\test_cuda_phase3_environment.py -q
.venv\Scripts\python -m experiments.cuda.phase3_oracle --seed 2071 --num-balloons 100 --dtype float32 --output .artifacts\cuda\phase3\balloons_seed2071_n100_f32.json
.venv\Scripts\python -m experiments.cuda.benchmark_phase3 --batch-sizes 4096 --steps 8 --warmup-steps 3 --repeats 3 --output .artifacts\cuda\phase3\throughput_seed2081_b4096.json
```

## Recommended implementation path

1. Keep the official ActiveRocketPy environment as the scoring oracle.  For
   the existing SB3 path, benchmark 8, 12, 16, and 20 CPU workers end-to-end
   instead of assuming the physics-only result determines the optimum.
2. Extend `TensorFlightBatch` into a competition-specific training backend,
   not a full PyTorch rewrite of RocketPy.  Port only the actual motor and
   mass/inertia curves, atmosphere/wind, aero surfaces, TVC/roll/throttle
   dynamics, sensors, launch/burnout/impact phases, and official reward/pop
   semantics used by the challenge.
3. Validate each layer in float64 against ActiveRocketPy: RHS snapshots,
   one-step results, fixed open-loop action traces, phase/event times, sensor
   outputs, swept pop identity, reward, and termination.  Test RK4 with one,
   two, and four substeps; trajectory and event error must stay comfortably
   below the 1.5 m pop-radius decision scale, with an explicit ambiguity band
   for boundary cases.
4. Only then connect a device-native vectorized PPO loop.  Hundreds to
   thousands of states, actions, observations, rewards, and rollout entries
   should remain on one GPU.  A `SubprocVecEnv` that returns NumPy every 0.01 s
   defeats this design.
5. Pretrain on the fast backend, then fine-tune and evaluate on canonical CPU
   ActiveRocketPy with Scenario 4 randomization.  Expand the port only if warm
   end-to-end rollout plus PPO update is at least roughly 2x faster and the
   canonical score distribution does not materially degrade.

If fixed-step convergence cannot meet the fidelity gate, compare a
batch-independent adaptive method such as `torchode` before attempting a full
RocketPy rewrite.

## Primary references

- [SciPy Array API support for integrate](https://docs.scipy.org/doc/scipy-1.17.0/dev/api-dev/array_api_modules_tables/integrate.html)
- [MATLAB `ode45` extended capabilities](https://www.mathworks.com/help/matlab/ref/ode45.html)
- [MATLAB `dlode45` GPU arrays](https://www.mathworks.com/help/deeplearning/ref/dlarray.dlode45.html)
- [SciML: the two forms of GPU ODE acceleration](https://docs.sciml.ai/DiffEqGPU/stable/getting_started/)
- [SciML ensemble simulations](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
- [Diffrax solver interface](https://docs.kidger.site/diffrax/api/solvers/abstract_solvers/)
- [`torchode`: batch-parallel ODE solver](https://torchode.readthedocs.io/en/latest/)
- [Stable-Baselines3 PPO CPU guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile)
