import copy
import os
import glob
import tempfile
import numpy as np
import pymap3d as pm
from rocketpy import (
    Environment, Flight, LinearGenericSurface, MonteCarlo,
    Rocket, SolidMotor, StochasticEnvironment, StochasticFlight, StochasticRocket
)

def build_environment(env_cfg, rng=None):
    """Build the RocketPy Environment exactly as BalloonPoppingEnv does.

    Mirrors ``BalloonPoppingEnv.__create_environment`` (atmosphere selection and
    optional wind gust). Any divergence here makes the generated pool describe a
    different world than the scenario it is supposed to stand in for -- notably
    the atmosphere, which carries the wind field that drives the balloons'
    horizontal drift.
    """
    py_env = Environment(
        date=env_cfg["date"], latitude=env_cfg["latitude"], longitude=env_cfg["longitude"],
        elevation=env_cfg["elevation"], datum="WGS84", timezone="UTC"
    )

    if env_cfg.get("atmosphere_data_filename") is None:
        py_env.set_atmospheric_model(type="standard_atmosphere")
    else:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", env_cfg["atmosphere_data_filename"]
        )
        py_env.set_atmospheric_model(type="Ensemble", file=path, dictionary="ECMWF")

    gust_cfg = env_cfg.get("gust", {})
    if gust_cfg.get("enable"):
        rng = rng if rng is not None else np.random.default_rng()
        altitude_nodes = np.arange(
            0.0, py_env.max_expected_height + gust_cfg["altitude_spacing"], gust_cfg["altitude_spacing"]
        )
        x_gust_nodes = rng.uniform(-gust_cfg["max_gust_speed"], gust_cfg["max_gust_speed"], size=len(altitude_nodes))
        y_gust_nodes = rng.uniform(-gust_cfg["max_gust_speed"], gust_cfg["max_gust_speed"], size=len(altitude_nodes))
        gust_decay = np.exp(-altitude_nodes / gust_cfg["gust_decay_height"])

        def gust_x(height_asl): return np.interp(height_asl, altitude_nodes, x_gust_nodes * gust_decay)
        def gust_y(height_asl): return np.interp(height_asl, altitude_nodes, y_gust_nodes * gust_decay)
        py_env.add_wind_gust(gust_x, gust_y)

    return py_env


def generate_balloon_chunk(config, num_simulations, random_seed=None):
    """Internal helper to simulate a single localized RocketPy Monte Carlo chunk."""
    env_cfg = config["environment"]
    sim_cfg = config["simulation"]
    bal_cfg = config["balloon"]

    py_env = build_environment(env_cfg, rng=np.random.default_rng(random_seed))

    # StochasticEnvironment mutates the environment it is given (each Monte Carlo
    # draw overwrites latitude/longitude), so hand it a copy and keep ``py_env``
    # pristine as the ENU reference. Sharing one object silently shifts every
    # track in the chunk by the last random draw's offset.
    monte_carlo_environment = copy.deepcopy(py_env)

    lat, lon = env_cfg["latitude"], env_cfg["longitude"]
    lat_std, lon_std = bal_cfg["stochastic"]["latitude_std"], bal_cfg["stochastic"]["longitude_std"]
    stochastic_env = StochasticEnvironment(
        environment=monte_carlo_environment,
        latitude=(lat - lat_std, lat + lat_std, "uniform"),
        longitude=(lon - lon_std, lon + lon_std, "uniform")
    )

    motor = SolidMotor(
        thrust_source=50, burn_time=0.2, grain_number=1, grain_density=100,
        grain_initial_inner_radius=0.01, grain_outer_radius=0.035, grain_initial_height=0.1,
        nozzle_radius=0.0335, nozzle_position=0, throat_radius=0.0114, grain_separation=0.00,
        grains_center_of_mass_position=0.2, dry_inertia=(0, 0, 0), center_of_dry_mass_position=0, dry_mass=0
    )

    aero_surface = LinearGenericSurface(
        reference_area=np.pi * bal_cfg["radius"] ** 2, reference_length=1,
        coefficient_constants=[
            bal_cfg["aero_coefficients"]["cL"], 0, 0, 0, 0, 0,
            bal_cfg["aero_coefficients"]["cQ"], 0, 0, 0, 0, 0,
            bal_cfg["aero_coefficients"]["cD"], 0, 0, 0, 0, 0,
            0, 0, 0, bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"],
            0, 0, 0, bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"],
            0, 0, 0, bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"], bal_cfg["aero_coefficients"]["moment_damping"]
        ],
        center_of_pressure=(0, 0, 0), name="Balloon Aero Model"
    )

    balloon_base = Rocket(
        volume=4 / 3 * np.pi * bal_cfg["radius"] ** 3, radius=0.05,
        mass=bal_cfg["mass"], inertia=tuple(bal_cfg["inertia"]),
        center_of_mass_without_motor=0.2, power_off_drag=0, power_on_drag=0, coordinate_system_orientation="tail_to_nose"
    )
    balloon_base.add_motor(motor, position=0)
    balloon_base.add_surfaces(aero_surface, positions=(0, 0, 0.2))

    stochastic_balloon = StochasticRocket(
        rocket=balloon_base, mass=bal_cfg["stochastic"]["mass_std"], volume=bal_cfg["stochastic"]["volume_std"],
        inertia_11=bal_cfg["stochastic"]["inertia_std"], inertia_22=bal_cfg["stochastic"]["inertia_std"], inertia_33=bal_cfg["stochastic"]["inertia_std"],
        center_of_mass_without_motor=0
    )
    stochastic_balloon.add_motor(motor, position=0)
    stochastic_balloon.add_linear_generic_surface(aero_surface)

    flight = Flight(
        rocket=balloon_base, environment=monte_carlo_environment, inclination=90, heading=180, rail_length=0.1,
        max_time=sim_cfg["max_time"], verbose=False, run_simulation=False, ode_solver="RK45"
    )
    stochastic_flight = StochasticFlight(flight=flight, inclination=5, heading=90)

    temp_base_path = os.path.join(tempfile.gettempdir(), f"generator_{os.getpid()}")

    time_array = np.arange(0, sim_cfg["max_time"], sim_cfg["time_step"])

    mc_sim = MonteCarlo(
        filename=temp_base_path,
        environment=stochastic_env, rocket=stochastic_balloon, flight=stochastic_flight,
        export_list=["t_final"],
        data_collector={
            "x": lambda f: f.x(time_array), "y": lambda f: f.y(time_array), "z": lambda f: f.z(time_array),
            "vx": lambda f: f.vx(time_array), "vy": lambda f: f.vy(time_array), "vz": lambda f: f.vz(time_array),
            "lat0": lambda f: f.latitude(0), "lon0": lambda f: f.longitude(0)
        }
    )

    results = mc_sim.simulate(
        number_of_simulations=num_simulations,
        append=False, include_function_data=False, random_seed=random_seed, parallel=False
    )

    for temp_file in glob.glob(f"{temp_base_path}*"):
        try:
            os.remove(temp_file)
        except OSError:
            pass

    east0, north0, up0 = pm.geodetic2enu(
        results["lat0"], results["lon0"], py_env.elevation,
        py_env.latitude, py_env.longitude, py_env.elevation
    )
    east0, north0, up0 = np.array(east0)[:, None], np.array(north0)[:, None], np.array(up0)[:, None]

    raw_flights = np.stack([
        np.array(results["x"]) + east0, np.array(results["y"]) + north0, np.array(results["z"]) + up0,
        np.array(results["vx"]), np.array(results["vy"]), np.array(results["vz"])
    ], axis=1)

    return raw_flights.astype(np.float32)

def generate_balloon_pool(config, total_tracks=5000, chunk_size=1000, output_path="single_balloons_pool.npy",
                          seed=0):
    """Compile a trajectory pool to disk.

    ``config`` must be a full scenario parameter dict so the tracks obey the same
    environment, balloon model and stochastic spread as that scenario. Tracks are
    stored UNSHIFTED; PoolEnv applies the release schedule when it injects them.
    """
    state_dims = 6

    max_time = config["simulation"]["max_time"]
    time_step = config["simulation"]["time_step"]
    num_timesteps = int(max_time / time_step)

    fp = np.lib.format.open_memmap(
        output_path,
        dtype='float32',
        mode='w+',
        shape=(total_tracks, state_dims, num_timesteps)
    )

    num_chunks = total_tracks // chunk_size
    if total_tracks % chunk_size != 0:
        raise ValueError("total_tracks must be perfectly divisible by chunk_size to maintain structural alignment.")

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        # Distinct per-chunk seed keeps the pool reproducible without repeating draws.
        chunk_data = generate_balloon_chunk(config, chunk_size, random_seed=seed + chunk_idx)

        fp[start_idx:end_idx] = chunk_data
        fp.flush()

    return fp
