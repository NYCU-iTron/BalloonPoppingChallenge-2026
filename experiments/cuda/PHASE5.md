# Phase 5: official transfer holdouts

Phase 5 compares the float32 CUDA training backend with the unmodified
Scenario 1 / ActiveRocketPy v0.1.1 evaluator. It separates three questions:

1. do online balloon trajectories remain faithful for the complete horizon;
2. does the rocket/event model remain faithful under identical actions; and
3. does one policy make equivalent decisions from independently generated
   official and Tensor observations?

The answer is currently **GO for numerical and lifecycle fidelity, but NO-GO
for a release policy**. The tested smoke policies did not pop a balloon, so
hit identity, last-pop time, and non-zero score transfer do not yet have
closed-loop coverage.

## Full-horizon balloon holdouts

Seeds 2201, 2203, and 2207 each compared 100 balloons for all 14,999 available
0.01-second transitions. TensorFlight kept only online previous/current/next
state; the canonical side retained the precomputed ActiveRocketPy trajectories
as the oracle.

| Seed | position p99 | velocity p99 | release max | status mismatches |
| ---: | ---: | ---: | ---: | ---: |
| 2201 | 0.04475 m | 0.002813 m/s | 0 s | 0 |
| 2203 | 0.04899 m | 0.003857 m/s | 0 s | 0 |
| 2207 | 0.04765 m | 0.004206 m/s | 0 s | 0 |

All pass the 0.05 m position and 0.02 m/s velocity gates. Isolated maxima at
the canonical rail/free discontinuity remain visible in the report and are not
hidden by the p99 gate.

## Open-loop rocket and event holdouts

The same three seeds used 100 balloons, float32 CUDA RK4, and a deterministic
changing action trace until impact. All seeds had zero reward, status, and
termination mismatches.

| Seed | closest-distance p99 | rocket-position p99 | sensor-component p99 |
| ---: | ---: | ---: | ---: |
| 2201 | 0.01249 m | 0.01673 m | 0.01453 |
| 2203 | 0.01254 m | 0.01673 m | 0.01453 |
| 2207 | 0.01265 m | 0.01673 m | 0.01453 |

This is the physics comparison: both backends receive the same action at each
step. It therefore does not hide model error behind policy feedback.

## Closed-loop independent-observation holdouts

A deterministic, explicitly untrained Phase 4 NumPy policy ran separately on
official and Tensor observations for seeds 2201 and 2203. It launched,
crossed the sensor-derived 40 m handoff, and ran through impact.

| Seed | action p99 | closest-distance p99 | official end | Tensor end |
| ---: | ---: | ---: | ---: | ---: |
| 2201 | 0.00144 | 0.03805 m | 13.71 s | 13.71000004 s |
| 2203 | 0.00117 | 0.04893 m | 14.78 s | 14.77999973 s |

Both sides reported `terminated=True`, `truncated=False`, score zero, and no
hits. Numerical and lifecycle gates pass. The report deliberately marks the
overall transfer gate incomplete because two matching empty hit lists do not
prove pop-event equivalence.

## Alfonso branch compatibility result

`origin/alfonso` cannot be merged or copied as a release example:

- it diverges from the v0.1.1 main history;
- its evaluator config references `RLAgent`, which the branch deleted;
- `RLNavigatorEnv` calls `select_target`, while its current selector only
  defines `select_targets`;
- the controller reads the obsolete `gimbal_range` key rather than
  `max_gimbal_angle`;
- the original agent does not reset its target cursor between episodes.

`alfonso_compat_agent.py` is a non-release, sensor-only compatibility fixture
that repairs those API/lifecycle defects without reading the oracle state. On
official Scenario 1 seed 2231 it completed a full run but scored zero. In the
independent-observation transfer run it executed on both backends, but its
high-gain feedback amplified small model differences:

| Metric | result |
| --- | ---: |
| action max-component p99 | 30.0 |
| rocket-position p99 | 312.64 m |
| closest-distance p99 | 266.18 m |
| official / Tensor end time | 106.96 / 111.96 s |
| official / Tensor score | 0 / 0 |

That is a compatibility **NO-GO**, not an environment crash. The old agent can
be made to call the new API, but it cannot be advertised as a TensorFlight
training/deployment example until its controller is tensorized and retrained
or fine-tuned against this backend, followed by a scoring official transfer
test. The public release wrapper and teammate-facing example are therefore
intentionally deferred instead of shipping a misleading zero-score example.

## Reproduce

Run the decision-gate tests:

```powershell
.venv\Scripts\python -m pytest tests\test_cuda_phase5_transfer.py -q
```

Generate the complete holdout report:

```powershell
.venv\Scripts\python -m experiments.cuda.phase5_transfer `
  --seeds 2201,2203,2207 --closed-loop-seeds 2201,2203 `
  --num-balloons 100 --dtype float32 --device cuda `
  --output .artifacts\cuda\phase5\holdout_scenario1_f32_cuda.json
```

The `passed` field requires all numerical/lifecycle gates **and** at least one
closed-loop hit. `numerical_and_lifecycle_passed` is reported separately so a
missing event-coverage case cannot be confused with a model regression.

## Decision

Phase 5 remains active. The next gate is a policy or controlled scenario that
produces at least one official pop and the same hit identity, score, and
last-pop time in TensorFlight outside the measured ambiguity band. Only after
that gate should the training CLI, example agent, and user guide be promoted
from `experiments/cuda` into a release-facing package surface.
