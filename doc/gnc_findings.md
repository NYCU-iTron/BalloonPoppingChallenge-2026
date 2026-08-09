# iTron GNC — measurements, findings and dead ends

Working record for the scenario 1 agent. Everything here is measured, not
argued: figures come from `scripts/benchmark_itron.py` over 24 seeds unless
stated otherwise. Written so that directions already ruled out are not retried.

---

## 1. Current state

```
24 seeds:  popped 4.38   std 0.65   min 3   max 5
           planned 5.04  completion 87%
```

Progression, each step confirmed on 24 seeds:

| Change | Score |
|---|---|
| Original proportional navigation | 0 |
| Guidance/autopilot rewrite, reachability-constrained selection | 1.92 |
| Cruise ceiling 25 → 15 m/s | 2.00 |
| Launch decided rather than fixed at t=70 | 3.00 |
| Legs priced by climb angle | 3.58 |
| `min_launch_chain` 3 → 5 | 4.00 |
| Drift-aware chain planning | 4.17 |
| Beam search in the velocity frame | 4.29 |
| Legs priced by turn radius | **4.38** |

### Measuring anything

```
python scripts/benchmark_itron.py --seeds 24 --jobs 8            # ~165 s
python scripts/benchmark_itron.py --seeds 24 --compare base.json # paired diff
python scripts/benchmark_itron.py --seeds 24 --env montecarlo    # real scenario env
```

A single flight is noise: seed-to-seed spread is 0.4–1.1 balloons, the same size
as most effects. An 8-seed sweep once picked `cruise_speed=30` at 4.62; at 24
seeds it scored 3.92. **The paired better/worse/unchanged count is far more
sensitive than comparing means** — a 0.3 difference in mean can be 8-worse-0-better.

Balloon flights come from `scripts/pool_scenario_1.npy` by default, which is the
same Monte Carlo the scenario env runs, done once. It saves ~30 s per run and
agrees with the scenario env (3.50 vs 3.58 on the same configuration).

---

## 2. The vehicle, measured

```
m0 = 90.9 kg    T = 1080 N    T/W = 1.21    burn = 30 s
mass flow 0.587 kg/s          tvc lever 0.80 m
gimbal 15 deg, rate limit 60 deg/s -> 0.25 s to cross its range
angular acceleration at full gimbal: 8.52 rad/s^2
```

| t (s) | mass | a_max | T/W | lateral while level | min climbable inclination |
|---|---|---|---|---|---|
| 0 | 90.9 | 11.88 | 1.21 | 6.70 | 65.5° |
| 15 | 82.1 | 13.16 | 1.34 | 8.77 | 55.0° |
| 30 | 73.3 | — | — | — | burnout |

**The 30 s burn is a hard, unmanageable window.** Propellant flow is a function
of time only — throttling scales thrust and never consumption
([flight.py:2041](../ActiveRocketPy/rocketpy/simulation/flight.py#L2041)), and the tank's
`flux_time` starts at launch. There is no way to save fuel for later.

### Scenario

100 balloons, released every 0.5 s from t=0 to t=49.5 from points scattered over
about ±110 m. They rise ~6 m/s and drift ~5 m/s downwind. Capture radius 1.5 m,
detected on the swept path between steps. Simulation runs at 100 Hz to t=150.

---

## 3. What the flight actually does

### Along-track acceleration is zero

```
gravity along track  -5.00 m/s^2
thrust  along track  +5.05 m/s^2
net                  +0.05 m/s^2
thrust-vs-velocity   59.2 deg
mean climb angle     32.8 deg
```

The rocket chases a downwind plume at a shallow 32.8°, but thrust must point
near vertical to hold the vehicle up. The angle between them is `90 − 32.8 =
57.2°`, so only `cos(59°) ≈ 0.51` of the thrust reaches the speed. **Essentially
all the speed is won on the first, steep leg and merely carried afterwards.**

### Corners cost speed, and the corners are at the cloud edge

```
mean turn inside the field   21.1 deg  (n=25)
mean turn at the edge        44.3 deg  (n=8)
correlation turn ~ edge      +0.66
correlation speed lost ~ turn +0.44
mean speed lost per leg      8.1 m/s   (entering at ~21)
```

24% of legs happen beyond the 95th percentile of the field. The rocket runs out
of balloons and has to reverse.

### The autopilot never strains, and never catches up

```
thrust axis vs commanded direction:  24.0 deg mean, 33.8 deg terminal, p90 81.9
gimbal deflection used:              1.43 of 15 deg, on its stop 1% of steps
navigator acceleration saturation:   36-43% of steps
```

Note these are two different saturations: the navigator's acceleration command
exceeds `T/m` often, while the *gimbal* almost never saturates.

---

## 4. Bugs found and fixed

| Bug | Effect |
|---|---|
| Attitude integrator started at identity, not the launch attitude | World→body rotation off by the launch heading (10.68°) for the whole flight |
| Launch heading used `arctan2(y, x)` where the simulator wants a compass bearing `arctan2(x, y)` | 248.6° error; harmless only because inclination was pinned at 90° |
| Selector's origin at sea level, not the pad | 20 m error in the launch-angle cost, weighted ×80 |
| Estimator's pre-launch position at sea level | Same family; one frame of bad geometry |
| Target lead applied twice — estimator extrapolated, then ZEM extrapolated again | Terminal miss 2.85 m → 1.34 m when fixed |
| Gimbal clipped per axis, so the pair could reach 15·√2 = 21.2° | Past what the nozzle can do, and it skewed the thrust direction |
| `max_body_rate` 1.5 rad/s sat *above* the gimbal's saturation threshold of 1.28 | Every large attitude change drove the actuator onto its stop |
| `valid_mask` used `isnan`, but the env never NaNs a balloon | Grounded and popped balloons counted as targets; only mattered once launching early |
| `K` guard required ≥8 candidates *ahead of the origin* | Mid-flight replans returned nothing |
| Chain planned from a static snapshot while balloons drift | Rocket flew 2.31× the planned leg length |
| RocketPy leaves `balloon_sim_<pid>.*.txt` in the temp dir forever | Filled the filesystem during sweeps; benchmark now cleans up |

---

## 5. Architecture

```
Selector   picks the chain, decides when to launch
Estimator  rocket state from IMU/GNSS, current target state
Navigator  guidance: owns the whole acceleration vector
Controller attitude autopilot only
Vehicle    shared mass / thrust / authority model
```

**`Vehicle` exists so guidance and selection cannot disagree about the rocket.**
The original split let the navigator command 30 m/s² of lateral acceleration
from a vehicle with 6.7 m/s² available.

**Navigator owns the entire acceleration vector** — the manoeuvre, gravity
cancellation, saturation against the real envelope, and the split into a thrust
direction and a throttle. Previously the navigator produced only a lateral
command, the controller silently added gravity, and the throttle came from an
unrelated heuristic; nothing ever reconciled the achieved acceleration with the
commanded one.

Guidance is zero-effort-miss, `a = 3·ZEM/t_go²`, with `t_go` from
`R = Vc·t + ½at²` rather than `range/closing_speed` so it stays finite from rest.
Proportional navigation's assumptions (high speed, T/W ≫ 1, negligible gravity)
are all false here, and its `max(v_closing, 0)` gate held the command at exactly
zero for the first nine seconds of every flight.

**Controller gains derive from actuator limits**, not hand-picked constants:
`rate_time_constant = gimbal_slew_time()`, attitude loop 2× slower, rate command
capped at what the gimbal can build within one time constant.

**Selection is a beam search over (position, heading, elapsed)** — no plume axis.
The axis bought a DAG to run a DP over, but that is computational convenience;
of its three jobs, the off-axis penalty measured harmful and the "ahead of the
origin" filter broke mid-flight replanning. Only acyclicity was load-bearing,
and the beam gets that free because every leg costs time.

---

## 6. Dead ends

Every row measured on 24 seeds unless noted.

### Planning ambition is neutral — four independent mechanisms

| Change | planned | completion | score |
|---|---|---|---|
| cruise 15→30, launching at t=70 | 2.17→2.83 | 92%→65% | 2.00→1.83 |
| cruise 15→30, launching at t=10 | 4.04→4.42 | 99%→89% | 4.00→3.92 |
| corner speed retention | 4.04→5.25 | 99%→76% | 4.00→4.00 |
| planning speed 15→22 (measured cruise) | 5.00→5.92 | 83%→70% | 4.17→4.12 |

**Every gain in planned targets is cancelled by execution failures.** The reason
became clear later: a miss at the end of the chain is *free* — the leftover burn
has no other use — so over-planning costs nothing and under-planning costs
balloons. The current setting already sits on that boundary.

### Everything else

| Idea | Result |
|---|---|
| Replan after every hit (axis DP) | 3.88 vs 4.17; 8 worse, 0 better |
| Replan the tail only, keeping the committed target | 3.50 on 4 seeds vs 4.00 |
| Replan after every hit (beam, no axis filter) | 3.92 vs 4.29; 8 worse, 0 better. Big misses fall 14→2, which is exactly the loss |
| Replan once the chain is exhausted | Never fires — a missed target is never retired |
| Shorter chains so they complete (`time_budget_fraction`) | 0.65→3.00, 0.75→3.50, 0.85→4.00, 0.95→**4.17** |
| Traverse the plume upwind instead of downwind | −0.67 to −1.17 at every launch time; the pad is at the upwind end |
| Climb to the top and work down | −0.25 to −1.00 |
| Single un-steered straight burn | ≤1 balloon over 468 directions × 6 seeds × 6 launch times |
| Raise the off-axis distance weight | 10→**4.17**, 25→4.04, 50→3.96, 100→3.92 |
| Prefer straighter chains (raise turn price) | Mean turn 39°→12.7° but score 2.38→2.00: straight means long legs |
| `nav_constant` above 3 | 3→**4.17**, 4→3.92, 5→3.83, 6→3.67 |
| ZEV terminal-direction constraint (threading) | 0 balloons full-blend, 2 with a commit phase, vs 3 |
| Ease off the aim once the miss is inside the balloon | 4.04 both as a switch and as a continuous blend, vs 4.38 |
| Inner loop faster | 0.05→3.08, 0.10→3.42, 0.15→3.75, **0.25→4.38** |
| Inner loop slower | 0.35→3.88, 0.50→2.83, 0.70→1.58 |
| Limit how fast the guidance command may turn | 0.5→3.12, 1.0→3.50, 1.5→3.67, 3.0→3.92, **off→4.38** |
| Loop separation 3× instead of 2× | 2 of 3 targets vs 3 of 3 (single seed) |
| Beam `angle_weight` | 5, 10 and 20 give bit-identical results; the turn is already priced through time |

### Two bounds that turned out to measure the wrong thing

**"Corners could be taken at 54 m/s"** — true if all the lateral authority went
into turning. The rocket also has to aim, and aiming is what spends it. Easing
off the aim to keep speed lost 0.34 balloons with 0 seeds improving. **The speed
given up in a corner is the price of the hit, not waste.**

**Offline trajectory optimisation** to bound the fastest possible two-balloon
leg pair failed: cross-entropy search found no feasible control history even at
the horizon the guidance itself achieved, where one provably exists. Hitting a
1.5 m ball at 20 m/s is too fine a target for open-loop random search. No bound
was obtained.

### Three hypotheses about the 24° tracking error, all wrong

The thrust axis sits 24° off the commanded direction on average. Faster inner
loop: worse. Slower inner loop: worse. Rate-limiting the command: worse. Both
the 0.25 s loop and the unlimited command are sharp optima. The remaining
explanation is that the command genuinely needs to swing that fast and the
vehicle arrives at each balloon still turning toward it — a normal operating
state for this vehicle on this task, not a defect.

---

## 7. Where the budget goes

```
28.5 s usable (0.95 of the burn)
~8 s   reaching the first balloon
~6.6 s per balloon after that
       -> about 4 balloons, which is what happens
```

Eight balloons needs ~2.9 s a leg, which needs ~26 m/s sustained. The vehicle
cruises at 20–25 m/s and its along-track acceleration is +0.05 m/s², so it
cannot build more. Chains of 45 m legs at 15 m/s take 4.2 s at best plus 1.2 s
settling — about 5.9 targets even with a free first leg.

---

## 8. Open

Nothing left that is tuning. Two options:

1. **Reinforcement learning for one leg pair.** Keep the selector and the
   autopilot, replace `navigator.compute` with a learned policy that sees the
   next target as well as the current one. The structural argument is that ZEM
   aims at one balloon and cannot, by construction, prepare for the following
   one; a policy can trade a little terminal accuracy for exit speed and find
   the exchange rate itself. The counter-argument is that both hand-built
   versions of that trade measured worse, so the current operating point may
   already be on the right side of it. Train on a single leg pair first: if a
   policy cannot beat ZEM over two balloons, the full task is not worth it.

2. **Stop at 4.38.** Best single flight was 6.

Untuned parameters, expected value low: `turn_time_per_radian`, `terminal_time`,
`min_launch_chain` (swept only under the old geometry), `earliest/latest_launch_time`,
`max_tilt`, `terminal_control_time`, `rate_command_margin`, `integral_gain`,
`roll_time_constant`.
