import argparse
import copy
from pathlib import Path
import random

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = REPO_ROOT / "BalloonPoppingGymEnv" / "envs" / "scenario_parameters"


def _load_yaml(path):
    with path.open("r", encoding="utf-8-sig") as file:
        return yaml.safe_load(file)


def _write_yaml(path, data):
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def _lerp(a, b, t):
    return a + (b - a) * t


def _jitter(value, amount, rng, min_value=None):
    if amount <= 0:
        out = value
    else:
        out = value * (1.0 + rng.uniform(-amount, amount))
    if min_value is not None:
        out = max(min_value, out)
    return out


def _axis_scales(axis_weights, axis_bias_strength):
    total = sum(axis_weights)
    if total <= 0:
        raise ValueError("--axis-weights must contain at least one positive value")

    normalized = [weight / total for weight in axis_weights]
    # Average normalized axis weight is 1/3. Scale around 1 so neutral is [1, 1, 1].
    return [max(0.05, 1.0 + axis_bias_strength * (weight * 3.0 - 1.0)) for weight in normalized]


def build_scenario(template, scenario_number, args, rng):
    scenario = copy.deepcopy(template)
    difficulty_span = max(args.end - args.start, 1)
    progress = (scenario_number - args.start) / difficulty_span
    difficulty = _lerp(args.difficulty_start, args.difficulty_end, progress)
    difficulty = max(0.0, min(1.0, difficulty))

    x_scale, y_scale, z_scale = _axis_scales(args.axis_weights, args.axis_bias_strength)

    scenario["scenario"]["number"] = scenario_number
    scenario["scenario"]["random_seed"] = None if args.random_seed is None else args.random_seed + scenario_number

    simulation = scenario["simulation"]
    balloon = scenario["balloon"]
    stochastic = balloon["stochastic"]
    environment = scenario["environment"]
    gust = environment["gust"]
    sensors = scenario["rocket"]["sensors"]

    simulation["max_time"] = round(
        _jitter(_lerp(args.max_time_easy, args.max_time_hard, difficulty), args.randomness * 0.08, rng, 10.0),
        3,
    )
    simulation["time_step"] = args.time_step

    balloon["num"] = int(round(_jitter(_lerp(args.num_easy, args.num_hard, difficulty), args.randomness * 0.15, rng, 1)))
    balloon["release_interval"] = round(
        _jitter(_lerp(args.release_interval_easy, args.release_interval_hard, difficulty), args.randomness * 0.20, rng, args.time_step),
        4,
    )
    balloon["radius"] = round(
        _jitter(_lerp(args.radius_easy, args.radius_hard, difficulty), args.randomness * 0.05, rng, 0.1),
        4,
    )

    # In this environment, horizontal initial spread is controlled by latitude/longitude std.
    # X is local east, mostly longitude. Y is local north, mostly latitude.
    spread = _jitter(args.point_std * _lerp(args.spread_easy_scale, args.spread_hard_scale, difficulty), args.randomness, rng, 1e-8)
    stochastic["longitude_std"] = float(round(spread * x_scale, 8))
    stochastic["latitude_std"] = float(round(spread * y_scale, 8))

    # There is no direct z-position std in the current scenario schema.
    # These fields influence balloon ascent/trajectory dispersion, so z bias maps here.
    stochastic["mass_std"] = round(
        _jitter(_lerp(args.mass_std_easy, args.mass_std_hard, difficulty) * z_scale, args.randomness * 0.4, rng, 0.0),
        5,
    )
    stochastic["volume_std"] = round(
        _jitter(_lerp(args.volume_std_easy, args.volume_std_hard, difficulty) * z_scale, args.randomness * 0.4, rng, 0.0),
        5,
    )
    stochastic["inertia_std"] = round(
        _jitter(_lerp(args.inertia_std_easy, args.inertia_std_hard, difficulty) * z_scale, args.randomness * 0.4, rng, 0.0),
        5,
    )

    gust["enable"] = args.enable_gust
    gust["max_gust_speed"] = round(
        _jitter(_lerp(args.gust_easy, args.gust_hard, difficulty), args.randomness * 0.35, rng, 0.0),
        4,
    )
    gust["altitude_spacing"] = args.gust_altitude_spacing
    gust["gust_decay_height"] = args.gust_decay_height

    sensor_noise = _lerp(args.sensor_noise_easy, args.sensor_noise_hard, difficulty)
    sensors["gyro_noise_density"] = round(_jitter(sensor_noise * 0.01, args.randomness * 0.3, rng, 0.0), 8)
    sensors["gyro_random_walk_density"] = round(_jitter(sensor_noise * 0.001, args.randomness * 0.3, rng, 0.0), 8)
    sensors["accelerometer_noise_density"] = round(_jitter(sensor_noise * 0.1, args.randomness * 0.3, rng, 0.0), 8)
    sensors["accelerometer_random_walk_density"] = round(_jitter(sensor_noise * 0.01, args.randomness * 0.3, rng, 0.0), 8)
    sensors["gnss_position_accuracy"] = round(_jitter(sensor_noise * 2.0, args.randomness * 0.3, rng, 0.0), 5)
    sensors["gnss_altitude_accuracy"] = round(_jitter(sensor_noise * 2.0 * z_scale, args.randomness * 0.3, rng, 0.0), 5)
    sensors["gnss_velocity_accuracy"] = round(_jitter(sensor_noise * 0.2, args.randomness * 0.3, rng, 0.0), 5)

    return scenario


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate practice scenario_x_parameters.yaml and "
            "scenario_x_given_parameters.yaml files for x >= 100."
        )
    )
    parser.add_argument("--start", type=int, default=100, help="First scenario number to generate, inclusive.")
    parser.add_argument("--end", type=int, default=120, help="Last scenario number to generate, inclusive.")
    parser.add_argument("--template-scenario", type=int, default=1, help="Scenario number used as the base template.")
    parser.add_argument("--output-dir", type=Path, default=SCENARIO_DIR, help="Directory for generated YAML files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated scenario files.")
    parser.add_argument("--random-seed", default="1000", help="Base seed. Scenario seed becomes base + scenario number. Use --random-seed none for null.")

    parser.add_argument("--difficulty-start", type=float, default=0.2, help="Difficulty at --start, 0.0 to 1.0.")
    parser.add_argument("--difficulty-end", type=float, default=0.9, help="Difficulty at --end, 0.0 to 1.0.")
    parser.add_argument("--randomness", type=float, default=0.25, help="Per-scenario random jitter, usually 0.0 to 1.0.")
    parser.add_argument("--point-std", type=float, default=0.001, help="Base horizontal balloon origin std in degrees.")
    parser.add_argument(
        "--axis-weights",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(1.0, 1.0, 1.0),
        help="Relative emphasis for x/east, y/north, z/ascent dispersion.",
    )
    parser.add_argument("--axis-bias-strength", type=float, default=0.5, help="How strongly --axis-weights reshape dispersion.")

    parser.add_argument("--time-step", type=float, default=0.01)
    parser.add_argument("--max-time-easy", type=float, default=100.0)
    parser.add_argument("--max-time-hard", type=float, default=150.0)
    parser.add_argument("--num-easy", type=float, default=20.0)
    parser.add_argument("--num-hard", type=float, default=100.0)
    parser.add_argument("--release-interval-easy", type=float, default=1.0)
    parser.add_argument("--release-interval-hard", type=float, default=0.35)
    parser.add_argument("--radius-easy", type=float, default=1.8)
    parser.add_argument("--radius-hard", type=float, default=1.2)
    parser.add_argument("--spread-easy-scale", type=float, default=0.5)
    parser.add_argument("--spread-hard-scale", type=float, default=2.0)
    parser.add_argument("--mass-std-easy", type=float, default=0.05)
    parser.add_argument("--mass-std-hard", type=float, default=0.35)
    parser.add_argument("--volume-std-easy", type=float, default=0.15)
    parser.add_argument("--volume-std-hard", type=float, default=0.8)
    parser.add_argument("--inertia-std-easy", type=float, default=0.03)
    parser.add_argument("--inertia-std-hard", type=float, default=0.2)
    parser.add_argument("--enable-gust", action="store_true", help="Enable synthetic wind gusts in generated scenarios.")
    parser.add_argument("--gust-easy", type=float, default=0.0)
    parser.add_argument("--gust-hard", type=float, default=5.0)
    parser.add_argument("--gust-altitude-spacing", type=float, default=10.0)
    parser.add_argument("--gust-decay-height", type=float, default=150.0)
    parser.add_argument("--sensor-noise-easy", type=float, default=0.0)
    parser.add_argument("--sensor-noise-hard", type=float, default=1.0)

    args = parser.parse_args()
    if str(args.random_seed).lower() == "none":
        args.random_seed = None
    else:
        args.random_seed = int(args.random_seed)
    if args.start < 100:
        raise ValueError("--start must be >= 100 to keep generated scenarios separate from official scenarios")
    if args.end < args.start:
        raise ValueError("--end must be >= --start")
    return args


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    template_parameters = _load_yaml(SCENARIO_DIR / f"scenario_{args.template_scenario}_parameters.yaml")
    template_given = _load_yaml(SCENARIO_DIR / f"scenario_{args.template_scenario}_given_parameters.yaml")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for scenario_number in range(args.start, args.end + 1):
        parameters_path = args.output_dir / f"scenario_{scenario_number}_parameters.yaml"
        given_path = args.output_dir / f"scenario_{scenario_number}_given_parameters.yaml"
        if not args.overwrite and (parameters_path.exists() or given_path.exists()):
            raise FileExistsError(f"{parameters_path.name} or {given_path.name} already exists. Pass --overwrite to replace.")

        scenario = build_scenario(template_parameters, scenario_number, args, rng)
        _write_yaml(parameters_path, scenario)
        _write_yaml(given_path, template_given)
        generated.append(scenario_number)

    print(f"Generated {len(generated)} training scenarios in {args.output_dir}")
    print(f"Range: scenario_{generated[0]} to scenario_{generated[-1]}")


if __name__ == "__main__":
    main()
