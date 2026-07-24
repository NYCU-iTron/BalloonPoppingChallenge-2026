from pathlib import Path
import tempfile
import numpy as np
import pymap3d as pm
from rocketpy import (
    Environment, Flight, LinearGenericSurface, MonteCarlo,
    Rocket, SolidMotor, StochasticEnvironment, StochasticFlight, StochasticRocket
)

def generate_balloon_chunk(config, chunk_size):
    env_cfg = config["environment"]
    sim_cfg = config["simulation"]
    balloon_cfg = config["balloon"]

    # -------------------------- Stochastic Environment -------------------------- #
    # Initialize base environment
    py_env = Environment(
        date=env_cfg["date"],
        latitude=env_cfg["latitude"],
        longitude=env_cfg["longitude"],
        elevation=env_cfg["elevation"],
        datum="WGS84",
        timezone="UTC"
    )

    # Set atmospheric model
    atmosphere_filename = env_cfg.get("atmosphere_data_filename")
    if atmosphere_filename is None:
        py_env.set_atmospheric_model(type="standard_atmosphere")
    else:
        data_path = Path(__file__).resolve().parent.parent / "envs" / "data" / atmosphere_filename
        py_env.set_atmospheric_model(type="Ensemble", file=str(data_path), dictionary="ECMWF")

    # Set stochastic environment
    lat = env_cfg["latitude"]
    lon = env_cfg["longitude"]
    lat_std = balloon_cfg["stochastic"]["latitude_std"]
    lon_std = balloon_cfg["stochastic"]["longitude_std"]
    stochastic_env = StochasticEnvironment(
        environment=py_env,
        latitude=(lat - lat_std, lat + lat_std, "uniform"),
        longitude=(lon - lon_std, lon + lon_std, "uniform")
    )

    # ---------------------------- Stochastic Balloon ---------------------------- #
    # Set propulsion motor for launching balloon
    motor = SolidMotor(
        thrust_source=50,
        burn_time=0.2,
        grain_number=1,
        grain_density=100,
        grain_initial_inner_radius=0.01,
        grain_outer_radius=0.035,
        grain_initial_height=0.1,
        nozzle_radius=0.0335,
        nozzle_position=0,
        throat_radius=0.0114,
        grain_separation=0.00,
        grains_center_of_mass_position=0.2,
        dry_inertia=(0, 0, 0),
        center_of_dry_mass_position=0,
        dry_mass=0
    )

    # Set aerodynamic surface
    aero_surface = LinearGenericSurface(
        reference_area=np.pi * balloon_cfg["radius"] ** 2, reference_length=1,
        coefficient_constants=[
            balloon_cfg["aero_coefficients"]["cL"], 0, 0, 0, 0, 0,
            balloon_cfg["aero_coefficients"]["cQ"], 0, 0, 0, 0, 0,
            balloon_cfg["aero_coefficients"]["cD"], 0, 0, 0, 0, 0,
            0, 0, 0, balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"],
            0, 0, 0, balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"],
            0, 0, 0, balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"], balloon_cfg["aero_coefficients"]["moment_damping"]
        ],
        center_of_pressure=(0, 0, 0),
        name="Balloon Aero Model"
    )

    # Set base balloon body
    balloon_base = Rocket(
        volume=4 / 3 * np.pi * balloon_cfg["radius"] ** 3,
        radius=0.05,
        mass=balloon_cfg["mass"],
        inertia=tuple(balloon_cfg["inertia"]),
        center_of_mass_without_motor=0.2,
        power_off_drag=0,
        power_on_drag=0,
        coordinate_system_orientation="tail_to_nose"
    )
    balloon_base.add_motor(motor, position=0)
    balloon_base.add_surfaces(aero_surface, positions=(0, 0, 0.2))

    # Set stochastic balloon
    stochastic_balloon = StochasticRocket(
        rocket=balloon_base,
        mass=balloon_cfg["stochastic"]["mass_std"],
        volume=balloon_cfg["stochastic"]["volume_std"],
        inertia_11=balloon_cfg["stochastic"]["inertia_std"],
        inertia_22=balloon_cfg["stochastic"]["inertia_std"],
        inertia_33=balloon_cfg["stochastic"]["inertia_std"],
        center_of_mass_without_motor=0
    )
    stochastic_balloon.add_motor(motor, position=0)
    stochastic_balloon.add_linear_generic_surface(aero_surface)

    # ----------------------------- Stochastic Flight ---------------------------- #
    flight = Flight(
        rocket=balloon_base,
        environment=py_env,
        inclination=90,
        heading=180,
        rail_length=0.1,
        max_time=sim_cfg["max_time"],
        verbose=False,
        run_simulation=False,
        ode_solver="RK45"
    )
    stochastic_flight = StochasticFlight(flight=flight, inclination=5, heading=90)


    # -------------------------------- Monte Carlo ------------------------------- #
    with tempfile.TemporaryDirectory(prefix="generator_") as temp_dir:
        temp_base_path = str(Path(temp_dir) / "balloon_sim")

        max_time = config["simulation"]["max_time"]
        time_step = config["simulation"]["time_step"]
        time_array = np.arange(0, max_time, time_step)

        mc_sim = MonteCarlo(
            filename=temp_base_path,
            environment=stochastic_env,
            rocket=stochastic_balloon,
            flight=stochastic_flight,
            export_list=["t_final"],
            data_collector={
                "x": lambda f: f.x(time_array), "y": lambda f: f.y(time_array), "z": lambda f: f.z(time_array),
                "vx": lambda f: f.vx(time_array), "vy": lambda f: f.vy(time_array), "vz": lambda f: f.vz(time_array),
                "lat0": lambda f: f.latitude(0), "lon0": lambda f: f.longitude(0)
            }
        )

        results = mc_sim.simulate(
            number_of_simulations=chunk_size,
            append=False,
            include_function_data=False,
            parallel=False
        )

    # Convert Raw Coordinates to Local ENU Frame Metrics
    east0, north0, up0 = pm.geodetic2enu(
        results["lat0"], results["lon0"], py_env.elevation,
        py_env.latitude, py_env.longitude, py_env.elevation
    )
    east0 = np.array(east0)[:, None]
    north0 = np.array(north0)[:, None]
    up0 = np.array(up0)[:, None]

    raw_flights = np.stack([
        np.array(results["x"]) + east0,
        np.array(results["y"]) + north0,
        np.array(results["z"]) + up0,
        np.array(results["vx"]),
        np.array(results["vy"]),
        np.array(results["vz"])
    ], axis=1).astype(np.float32)

    return raw_flights

def generate_balloon_pool(config, total_tracks, output_path="balloons_pool.npy"):
    chunk_size = 1000
    assert total_tracks % chunk_size == 0

    state_dims = 6 # [x, y, z, vx, vy, vz]

    max_time = config["simulation"]["max_time"]
    time_step = config["simulation"]["time_step"]
    num_timesteps = len(np.arange(0, max_time, time_step))

    # Allocate memory-mapped array on disk
    fp = np.lib.format.open_memmap(
        output_path,
        dtype='float32',
        mode='w+',
        shape=(total_tracks, state_dims, num_timesteps)
    )

    # Generate tracks by chunk
    num_chunks = total_tracks // chunk_size
    for chunk_idx in range(num_chunks):
        chunk_data = generate_balloon_chunk(config, chunk_size)

        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        fp[start_idx:end_idx] = chunk_data
        fp.flush()

    print(f"[SUCCESS] Saved pool at: '{output_path}'\n")
