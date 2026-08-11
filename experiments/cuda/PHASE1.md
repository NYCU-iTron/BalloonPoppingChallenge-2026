# Phase 1: canonical oracle and agent boundary

Phase 1 establishes measurement infrastructure.  It does **not** claim that
the current `TensorFlightBatch` matches ActiveRocketPy.  ActiveRocketPy v0.1.1
remains the canonical oracle throughout later phases.

## Control ownership

The official action contains launch planning as well as post-launch control.
The training architecture keeps those responsibilities separate:

```text
LaunchPlanner: launch + inclination + heading
                       |
                       v
BootstrapController: fixed historical post-launch control
                       |
                       v
ControlHandoff: sensor-estimated condition (historically about 40 m AGL)
                       |
                       v
PPO PostLaunchController: [roll, tvc_x, tvc_y, throttle]
```

`LaunchPlanner` does not own the 40 m climb.  The handoff receives an
`EstimatedRocketFeatures` value reconstructed from the official 12-element
sensor observation.  It cannot accept `OracleRocketState`.

The official launch boundary was measured and is covered by regression tests:

| Frame | Rocket sensors |
| --- | --- |
| reset | all NaN |
| pre-launch step | all NaN |
| launch-command step | all NaN |
| first step after launch | 12/12 finite |

The historical 29-D E2E builder therefore runs only after sensor validity and
controller activation. `MaskedRunningNormalizer` updates its statistics only
for rows satisfying both conditions; inactive or non-finite rows cannot poison
the statistics.

Repository inspection also confirmed that the old `e2e` and
`improve-rl-navigator` policy paths used a sensor-derived estimator.  Their
selectors/controllers did not consume `info["rocket_states"]`.  True state was
used only inside the simulator, result export, and diagnostic scripts.

## Canonical trace schema

`canonical_oracle.py` records a compressed, versioned NPZ containing:

- official action and observation fields;
- true 13-state and active 13-D RHS for oracle diagnostics;
- rate-limited/saturated actuator outputs;
- balloon states/status, per-balloon release transitions, and swept closest
  distance;
- reward, popped count, phase index, terminated and truncated;
- explicit reset/pre-launch/launch/first-post-launch/burnout/hit/impact/timeout
  event flags;
- scenario hash plus repository and ActiveRocketPy revisions;
- independent moving-segment hit and miss geometry fixtures.

The fixed corpus names five regimes: `launch_control`, `burnout`, `impact`,
`timeout_without_launch`, and `moving_balloon`.  Generated traces belong under
`.artifacts/` and are not source-controlled.

A complete corpus smoke run verified that the burnout case crosses the motor
burn time and records a pop, the cutoff case reaches impact, the never-launch
case ends with `truncated=True` and no launch event, and the Scenario 1 case
contains an online-moving balloon trajectory from the canonical precomputation.

```powershell
.venv\Scripts\python -m experiments.cuda.canonical_oracle `
  --case launch_control `
  --output .artifacts\cuda\oracle\launch_control.npz

.venv\Scripts\python -m experiments.cuda.canonical_oracle `
  --corpus-dir .artifacts\cuda\oracle
```

## Current TensorFlight baseline

The comparator initializes the non-canonical tensor surrogate from the first
valid post-launch canonical state, converts the same physical commands back to
normalized post-launch actions, and reports error by state group.

```powershell
.venv\Scripts\python -m experiments.cuda.compare_canonical_trace `
  .artifacts\cuda\oracle\launch_control.npz `
  --output .artifacts\cuda\oracle\launch_control_report.json
```

On the Phase 1 machine, the intentionally aggressive control trace impacted
after 142 recorded frames, leaving 138 aligned post-launch transitions.  The
current simplified surrogate produced:

| Metric | p99 |
| --- | ---: |
| position L2 error | 8.708 m |
| velocity L2 error | 33.447 m/s |
| attitude angular error | 81.907 deg |
| angular-rate L2 error | 2.734 rad/s |
| roll actuator absolute error | 4.324 N m |
| TVC actuator L2 error | 20.599 deg |
| throttle absolute error | 0.452 |

This is expected and is the reason Phase 2 begins with equation/RHS parity.
The result must not be presented as RocketPy-equivalent performance.

## Synchronous CPU rollout baseline

`benchmark_cpu_training_loop.py` includes the boundary omitted by the original
physics-only process benchmark.  At every 0.01 s barrier the parent runs a
29→256→256→4 NumPy MLP, sends an action to each process, receives the next 29-D
observation, and waits for every worker. Before timing, each worker uses the
sensor estimator plus bootstrap controller to reach the estimated 40 m AGL
handoff; no oracle altitude participates.

Three 300-barrier repeats on the Phase 1 Windows machine:

| Workers | Median transitions/s | Range | Median barriers/s |
| ---: | ---: | ---: | ---: |
| 1 | 390.0 | 377.3–391.0 | 390.0 |
| 4 | 1,434.8 | 1,428.3–1,435.6 | 358.7 |
| 8 | 1,928.9 | 1,928.2–1,935.8 | 241.1 |
| 16 | 3,294.9 | 3,191.2–3,311.2 | 205.9 |
| 20 | 2,447.5 | 2,447.4–2,449.7 | 122.4 |

This measures synchronous rollout collection, including IPC and inference, but
does not include PPO optimization.  Sixteen workers beat twenty on this
machine; later end-to-end learning benchmarks must still remeasure rather than
hard-code that count.

```powershell
.venv\Scripts\python -m experiments.cuda.benchmark_cpu_training_loop `
  --workers 1,4,8,16,20 --steps 300 --warmup-steps 10 --repeats 3
```

## Phase 0 organizer decision

The team reports that the organizer approved training with a custom
GPU-vectorized surrogate, provided evaluation continues to use the unmodified
official simulator and the submitted agent consumes only allowed observations
and given parameters. Phase 0 is therefore closed; the technical fidelity and
transfer gates remain mandatory.
