import os
import glob
import tempfile
import numpy as np
import pymap3d as pm
from rocketpy import (
    Environment, Flight, LinearGenericSurface, MonteCarlo,
    Rocket, SolidMotor, StochasticEnvironment, StochasticFlight, StochasticRocket
)

def generate_balloon_chunk(config, num_simulations):
    """Internal helper to simulate a single localized RocketPy Monte Carlo chunk."""
    env_cfg = config["environment"]
    sim_cfg = config["simulation"]
    bal_cfg = config["balloon"]

    # Initialize Base Aero Environment
    py_env = Environment(
        date=env_cfg["date"], latitude=env_cfg["latitude"], longitude=env_cfg["longitude"],
        elevation=env_cfg["elevation"], datum="WGS84", timezone="UTC"
    )
    py_env.set_atmospheric_model(type="standard_atmosphere")

    # Set Stochastic Geographical Boundaries
    lat, lon = env_cfg["latitude"], env_cfg["longitude"]
    lat_std, lon_std = bal_cfg["stochastic"]["latitude_std"], bal_cfg["stochastic"]["longitude_std"]
    stochastic_env = StochasticEnvironment(
        environment=py_env,
        latitude=(lat - lat_std, lat + lat_std, "uniform"),
        longitude=(lon - lon_std, lon + lon_std, "uniform")
    )

    # Define Micro Solid Motor Propulsion
    motor = SolidMotor(
        thrust_source=50, burn_time=0.2, grain_number=1, grain_density=100,
        grain_initial_inner_radius=0.01, grain_outer_radius=0.035, grain_initial_height=0.1,
        nozzle_radius=0.0335, nozzle_position=0, throat_radius=0.0114, grain_separation=0.00,
        grains_center_of_mass_position=0.2, dry_inertia=(0, 0, 0), center_of_dry_mass_position=0, dry_mass=0
    )

    # Define Linear Aerodynamics Profile
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

    # Assemble Structural Blueprint
    balloon_base = Rocket(
        volume=4 / 3 * np.pi * bal_cfg["radius"] ** 3, radius=0.05,
        mass=bal_cfg["mass"], inertia=tuple(bal_cfg["inertia"]),
        center_of_mass_without_motor=0.2, power_off_drag=0, power_on_drag=0, coordinate_system_orientation="tail_to_nose"
    )
    balloon_base.add_motor(motor, position=0)
    balloon_base.add_surfaces(aero_surface, positions=(0, 0, 0.2))

    # Inject Stochastic Uncertainty Metrics
    stochastic_balloon = StochasticRocket(
        rocket=balloon_base, mass=bal_cfg["stochastic"]["mass_std"], volume=bal_cfg["stochastic"]["volume_std"],
        inertia_11=bal_cfg["stochastic"]["inertia_std"], inertia_22=bal_cfg["stochastic"]["inertia_std"], inertia_33=bal_cfg["stochastic"]["inertia_std"],
        center_of_mass_without_motor=0
    )
    stochastic_balloon.add_motor(motor, position=0)
    stochastic_balloon.add_linear_generic_surface(aero_surface)

    # Set Flight Solver
    flight = Flight(
        rocket=balloon_base, environment=py_env, inclination=90, heading=180, rail_length=0.1,
        max_time=sim_cfg["max_time"], verbose=False, run_simulation=False, ode_solver="RK45"
    )
    stochastic_flight = StochasticFlight(flight=flight, inclination=5, heading=90)

    temp_base_path = os.path.join(tempfile.gettempdir(), f"generator_{os.getpid()}")

    # Initialize Timestep Matrix Collection
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
        append=False, include_function_data=False, parallel=False
    )

    for temp_file in glob.glob(f"{temp_base_path}*"):
        try:
            os.remove(temp_file)
            print(f"[CLEANUP] Removed temporary file: {temp_file}")
        except OSError:
            pass

    # Convert Raw Coordinates to Local ENU Frame Metrics
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

def generate_balloon_pool(config, total_tracks=5000, chunk_size=1000, output_path="single_balloons_pool.npy"):
    """
    Main external API generator.
    Accepts arbitrary configuration dicts to compile standalone trajectory matrices onto disk.
    """
    state_dims = 6  # Fixed kinematic properties: [x, y, z, vx, vy, vz]

    # Dynamically compute total timesteps from current configuration parameters
    max_time = config["simulation"]["max_time"]
    time_step = config["simulation"]["time_step"]
    num_timesteps = int(max_time / time_step)

    print(f"\n[INIT] Allocation sequence started for target map: '{output_path}'")
    print(f"[INFO] Array Layout dimensions: ({total_tracks}, {state_dims}, {num_timesteps})")

    # Safe disk-mapping initialization
    fp = np.lib.format.open_memmap(
        output_path,
        dtype='float32',
        mode='w+',
        shape=(total_tracks, state_dims, num_timesteps)
    )

    num_chunks = total_tracks // chunk_size
    if total_tracks % chunk_size != 0:
        raise ValueError("total_tracks must be perfectly divisible by chunk_size to maintain structural alignment.")

    # Execute operational loop using integrated progress visualization
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size

        # Invoke inner worker function
        chunk_data = generate_balloon_chunk(config, chunk_size)

        # Map slices back onto disk sectors smoothly
        fp[start_idx:end_idx] = chunk_data
        fp.flush()

    print(f"[SUCCESS] Compiled tracking pool registry at: '{output_path}'\n")
    return fp
