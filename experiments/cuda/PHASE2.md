# Phase 2: Scenario 1 rocket fidelity

Phase 2 ports the competition-specific Scenario 1 generalized 6-DoF rocket
equations to batched PyTorch and validates them in float64 against the
unmodified ActiveRocketPy simulator. It does not yet add the 100-balloon GPU
world, observations, sensors, rewards, events, or a PPO adapter; those remain
Phase 3 and Phase 4 work.

The team reports that the organizer approved using a custom GPU-vectorized
surrogate for training while retaining the unmodified official simulator for
evaluation and using only allowed observations. This closes the Phase 0 rules
gate recorded in Phase 1.

## Boundary and implementation

`phase2_oracle.py` is the only Phase 2 module that imports the official
environment. It creates a canonical Scenario 1 flight and extracts immutable
tables and constants. `phase2_rocket.py` then runs independently using only
PyTorch operations:

```text
ActiveRocketPy Scenario 1
  -> one-time table/constant extraction
  -> Scenario1TensorRocket
       -> components(): diagnostic force/moment/RHS decomposition
       -> rhs(): batched integration hot path
       -> rk4_step() / dopri5_step(): separate solver layer
```

The port follows `Flight.u_dot_generalized`, which is the equation selected by
the official Scenario 1 environment. It includes:

- time-varying total mass, mass flow, center of mass, and full inertia tensor;
- density, pressure, viscosity, speed of sound, wind, and gravity tables;
- motor thrust and pressure thrust, throttle, TVC, and roll control;
- power-on/off drag, nose and trapezoidal-fin lift and moments;
- moving-center-of-mass coupling, buoyancy/weight, and Coriolis acceleration;
- quaternion and angular dynamics.

The training-side module has no NumPy, SciPy, RocketPy, Gymnasium, or official
environment import. Its RHS accepts leading batch dimensions, preserves the
selected device and dtype, and avoids a data-dependent host read in the solver
hot path. Finite-action validation remains available at the actuator boundary;
a future device-native environment can instead mask invalid rows without a
CUDA synchronization.

## Actuator parity

`Scenario1ActuatorBank` reproduces the official update order:

```text
optional low-pass filter -> rate limit -> saturation
```

All Scenario 1 time constants are null, but rate limiting remains active. At
100 Hz, the maximum change per command is 0.2 N m for roll, 0.6 degrees for
each TVC axis, and 0.02 for throttle. Tests compare a sequence of extreme and
ordinary commands directly with the four canonical actuator objects, including
reset and masked-reset behavior.

## RHS parity result

The formal report used seed 2031, a 0.00125 s time-property table, and an
open-loop command trace until impact. It evaluated 204 canonical RHS states
and 203 complete 0.01 s integration intervals. One impact-root interval was
shorter and was reported separately rather than mixed into regular-step error.

Selected float64 component errors:

| Component norm error | p99 | maximum |
| --- | ---: | ---: |
| atmosphere vector | 5.36e-9 | 5.36e-9 |
| aerodynamic force | 3.96e-15 | 6.40e-15 |
| aerodynamic moment | 4.00e-15 | 4.44e-15 |
| angular acceleration | 8.91e-16 | 1.34e-15 |
| center-of-mass second derivative | 6.73e-6 | 1.04e-4 |
| complete 13-state RHS | 6.73e-6 | 1.04e-4 |

Mass, mass flow, thrust, control moment, and quaternion derivative are exact at
reported precision. The limiting term is the canonical center-of-mass second
derivative: ActiveRocketPy obtains it through a finite second difference of a
composite `Function`. The tensor port records that value on the time table, so
the interpolation reproduces the same small numerical noise rather than hiding
it by changing the physical equation.

## Integrator parity and the FSAL boundary

RHS parity was established before testing solvers. Classical RK4 with one,
two, and four substeps was compared with a fresh-RHS Dormand--Prince 5 step.
All three pass the Phase 2 one-step gates. Selected p99 errors are:

| RK4 substeps | position (m) | velocity (m/s) | attitude (deg) | angular rate (rad/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6.80e-9 | 1.30e-6 | 1.71e-6 | 1.76e-7 |
| 2 | 2.85e-9 | 4.10e-7 | 2.40e-6 | 2.32e-7 |
| 4 | 3.74e-9 | 7.99e-7 | 2.40e-6 | 3.28e-7 |

The tiny non-monotonic residue at two versus four substeps comes from the
sampled canonical mass-property derivative, not an incorrect force equation.
There is no evidence in this result that an adaptive GPU ODE library is needed
for Scenario 1.

The comparison also uncovered a canonical compatibility detail. SciPy RK45 is
an FSAL method: its endpoint derivative is cached as the next step's first
stage. The official control loop changes actuator outputs at each 0.01 s node
without invalidating that cache. Therefore the first derivative after a command
jump was computed using the preceding actuator output. A mathematically fresh
RK4 step cannot converge to that behavior merely by adding substeps; its p99
angular-rate difference from the official step remains about 2.70e-4 rad/s.

`dopri5_step(initial_rhs=...)` makes this boundary explicit. With the official
cached first stage, it reproduces the unmodified evaluator with these p99
errors:

| State group | p99 |
| --- | ---: |
| position | 5.74e-9 m |
| velocity | 1.34e-6 m/s |
| attitude | 1.71e-6 deg |
| angular rate | 8.78e-9 rad/s |

This is an oracle-compatibility mode, not a recommendation to reproduce stale
derivatives in every future backend. Phase 3 transfer tests can compare both:
fresh-RHS integration for intended continuous dynamics, and stale-FSAL mode
when bit-level behavioral agreement with the current official evaluator is
important.

## What was borrowed from SciML and Diffrax

No Julia or JAX implementation was copied, and neither ecosystem was added as
a dependency. Their designs still provide useful architecture lessons:

- SciML's `EnsembleProblem` separates trajectory construction, per-trajectory
  randomness, output selection, reductions, and batch scheduling. Phase 3
  should follow the same separation for balloon/world RNG and should not save
  full histories on the GPU.
- DiffEqGPU distinguishes fused whole-solve kernels from array-based batched
  solvers with synchronized adaptivity. That reinforces using fixed 0.01 s
  batched stepping first, while benchmarking batch size rather than assuming a
  single small ODE benefits from CUDA.
- Both ecosystems make discontinuities part of the solver contract. SciML
  invalidates FSAL/Jacobian caches when a callback changes state, and Diffrax
  exposes a `made_jump` flag in solver steps. TensorFlight therefore exposes
  the first-stage/cached derivative explicitly instead of burying it inside
  mutable solver state.
- Diffrax separates vector fields (`Term`), solver state, manual stepping,
  events, and `SaveAt`. The Phase 2 tensor RHS and solver functions follow that
  same separation, and later phases should save only rollout data and event
  summaries required by training.
- Per-sample result masks are preferable to one batch-wide exception. That
  belongs in the Phase 3 asynchronous reset/event implementation.

Primary design references:

- [SciML ensemble simulations](https://docs.sciml.ai/DiffEqDocs/stable/features/ensemble/)
- [DiffEqGPU trajectory-count guidance](https://docs.sciml.ai/DiffEqGPU/stable/manual/optimal_trajectories/)
- [DiffEqGPU ensembler selection](https://docs.sciml.ai/DiffEqGPU/stable/manual/choosing_ensembler/)
- [DiffEqGPU callbacks on GPUs](https://docs.sciml.ai/DiffEqGPU/stable/tutorials/parallel_callbacks/)
- [SciML integrator interface and cache invalidation](https://docs.sciml.ai/DiffEqDocs/stable/basics/integrator/)
- [Diffrax terms](https://docs.kidger.site/diffrax/api/terms/)
- [Diffrax solver interface and `made_jump`](https://docs.kidger.site/diffrax/api/solvers/abstract_solvers/)
- [Diffrax manual stepping](https://docs.kidger.site/diffrax/usage/manual-stepping/)
- [Diffrax output selection](https://docs.kidger.site/diffrax/api/saveat/)
- [Diffrax events](https://docs.kidger.site/diffrax/api/events/)

## Reproduce

Run the Phase 2 gates, including a CUDA device-residency test when CUDA is
available:

```powershell
.venv\Scripts\python -m pytest tests\test_cuda_phase2_fidelity.py -q
```

Generate the full report:

```powershell
.venv\Scripts\python -m experiments.cuda.phase2_oracle `
  --steps 256 --seed 2031 `
  --output .artifacts\cuda\phase2\report_seed2031.json
```

## Phase 2 decision

Phase 2 is **GO** for review:

- generalized RHS component parity passes in float64;
- Scenario 1 saturation and rate-limit parity passes;
- RK4 1/2/4 all pass the fresh-RHS one-step gates;
- current-evaluator stale-FSAL behavior is understood and reproduced;
- the same batched RHS and both solvers execute and remain on CUDA.

Torchode/Diffrax is therefore not introduced at this gate. Phase 3 should use
the smallest fixed-step configuration that passes full-trajectory and event
transfer, while retaining the explicit solver boundary for later replacement.
