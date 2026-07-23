import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from rocketpy import (
    Environment,
    Flight,
    Function,
    Rocket,
)
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.sensors.gyroscope import Gyroscope
from rocketpy.tools import euler313_to_quaternions


class PoolEnv(gym.Env):
    metadata = {"render_modes": ["vpython", "matplotlib"]}

    def __init__(self, render_mode, parameters):
        self.scenario_parameters = parameters["scenario"]
        self.environment_parameters = parameters["environment"]
        self.simulation_parameters = parameters["simulation"]
        self.balloon_parameters = parameters["balloon"]
        self.rocket_parameters = parameters["rocket"]

        # ActiveRocketPy flight classes for rocket agent simulation
        self._rocket_flight = None
        self._balloon_flights = None

        # State and logging placeholders
        self.initial_solution = None
        self._balloon_status = np.zeros((self.balloon_parameters["num"], 1), dtype=int)
        self._balloon_states = np.array(np.zeros((self.balloon_parameters["num"], 6)))
        self._rocket_sensors = np.full(12, np.nan)
        self._rocket_states = np.full(13, np.nan)
        self.trajectories = None

        # Step tracking metrics
        self.rocket_launched = False
        self.current_step = 0
        self.num_timesteps = 0
        self._popped_count = 0
        self._balloon_release_at_step = None
        self._rocketpy_env = None

        # Wind-field rotation state (see set_wind_rotation / __create_environment):
        # the base wind-vs-altitude table is captured once from the atmosphere
        # so per-episode rotations are pure array math, no file reload.
        self._wind_rotation = 0.0
        self._base_wind_profile = None

        # --- DECOUPLED TRACK CONFIGURATION ---
        self.state_dims = 6
        self.num_pool_timesteps = 15000
        # Holds the raw un-shifted trajectories passed from the outside script
        self._raw_source_trajectories = None

        # Gym space allocations (Now perfectly bound to immutable initialized num)
        self.observation_space = spaces.Dict({
            "simulation_time": spaces.Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
            "balloon_status": spaces.MultiDiscrete(3 * np.ones((self.balloon_parameters["num"], 1), dtype=int)),
            "balloon_states": spaces.Box(
                low=-np.inf * np.ones((self.balloon_parameters["num"], 6)),
                high=np.inf * np.ones((self.balloon_parameters["num"], 6)),
                dtype=np.float64
            ),
            "rocket_sensors": spaces.Box(low=-np.inf * np.ones(12), high=np.inf * np.ones(12), dtype=np.float64),
        })

        self.action_space = spaces.Dict({
            "launch": spaces.Box(low=0, high=1, shape=(), dtype=bool),
            "launch_inclination_heading": spaces.Box(low=np.array([0, 0]), high=np.array([90, 360]), shape=(2,), dtype=np.float64),
            "tvc": spaces.Box(
                low=-self.rocket_parameters["control"]["gimbal_range"] * np.ones(2),
                high=self.rocket_parameters["control"]["gimbal_range"] * np.ones(2),
                dtype=np.float64
            ),
            "throttle": spaces.Box(
                low=self.rocket_parameters["control"]["throttle_range"][0],
                high=self.rocket_parameters["control"]["throttle_range"][1],
                shape=(),
                dtype=np.float64
            ),
            "roll": spaces.Box(
                low=-self.rocket_parameters["control"]["max_roll_torque"],
                high=self.rocket_parameters["control"]["max_roll_torque"],
                shape=(),
                dtype=np.float64
            ),
        })

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None

    def update_source_trajectories(self, raw_trajectories):
        """
        External API to inject pre-extracted trajectories before reset.
        Expected shape: (num_balloons, 6, 15000)
        """
        expected_shape = (self.balloon_parameters["num"], self.state_dims, self.num_pool_timesteps)
        if raw_trajectories.shape != expected_shape:
            raise ValueError(
                f"Trajectory shape mismatch! Expected {expected_shape}, "
                f"but received {raw_trajectories.shape}."
            )
        self._raw_source_trajectories = raw_trajectories

    def _get_obs(self):
        sim_time = self.current_step * self.simulation_parameters["time_step"]
        return {
            "simulation_time": sim_time,
            "balloon_status": self._balloon_status,
            "balloon_states": self._balloon_states,
            "rocket_sensors": self._rocket_sensors,
        }

    def _get_info(self):
        return {
            "rocket_states": self._rocket_states,
            "popped_count": self._popped_count,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.__create_environment()
        self._rocket_flight = None
        self._balloon_flights = None
        self.initial_solution = None

        self.__reset_balloon_release_sequence()

        # Scenario 0 remains a hardcoded static debugging corridor
        if self.scenario_parameters["number"] == 0:
            self.__generate_static_balloon_flights()
            self._balloon_status = np.ones((self.balloon_parameters["num"], 1), dtype=int)
        else:
            # For Scenario 1, 2, and all higher levels:
            # Positions are 100% governed by the externally injected pool tracks.
            # Any preset parameter positions are completely ignored and bypassed.
            self.__process_injected_balloon_flights()
            self._balloon_status = np.zeros((self.balloon_parameters["num"], 1), dtype=int)

        self._balloon_states = self._balloon_flights[:, :, 0]
        self._rocket_sensors = np.full(12, np.nan)
        self._rocket_states = np.full(13, np.nan)
        self.trajectories = None

        self.rocket_launched = False
        self.current_step = 0
        self.num_timesteps = self._balloon_flights.shape[2]
        self._popped_count = 0

        observation = self._get_obs()
        info = self._get_info()

        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None
        self._render_frame()

        return observation, info

    def __num_episode_timesteps(self):
        """Episode timeline length in steps, governed solely by ``simulation.max_time``."""
        return len(np.arange(0, self.simulation_parameters["max_time"], self.simulation_parameters["time_step"]))

    def __process_injected_balloon_flights(self):
        """
        Processes the clean trajectories passed from outside via sequential time masks.
        Forces the simulation to rely SOLELY on the injected pool data.

        The episode timeline length is governed by ``simulation.max_time``, so the
        balloon tracks are truncated to fit the scenario independently of the
        (typically longer) source pool length.
        """
        num_balloons = self.balloon_parameters["num"]

        if self._raw_source_trajectories is None:
            raise RuntimeError(
                "CRITICAL RUNTIME ERROR: No trajectories found! Please call "
                "'env.update_source_trajectories(raw_tracks)' prior to executing env.reset()."
            )

        # Directly reference the RAM slice sampled externally from the npy file
        raw_sampled_flights = self._raw_source_trajectories

        # Source pool length (what was injected) vs. episode length (truncated by max_time)
        num_source_timesteps = raw_sampled_flights.shape[2]
        num_episode_timesteps = self.__num_episode_timesteps()

        release_steps = np.asarray(self._balloon_release_at_step, dtype=int)
        release_steps = np.clip(release_steps, 0, num_episode_timesteps)

        time_idx = np.arange(num_episode_timesteps)
        source_idx = np.clip(
            time_idx[np.newaxis, :] - release_steps[:, np.newaxis], 0, num_source_timesteps - 1
        )

        balloon_idx = np.arange(num_balloons)[:, None, None]
        state_idx = np.arange(self.state_dims)[None, :, None]
        shifted = raw_sampled_flights[balloon_idx, state_idx, source_idx[:, None, :]]

        pre_release_mask = time_idx < release_steps[:, np.newaxis]
        initial_states = raw_sampled_flights[:, :, 0:1]

        # Final active flights are filled cleanly and exclusively using pool data
        self._balloon_flights = np.where(pre_release_mask[:, np.newaxis, :], initial_states, shifted)

    def __generate_static_balloon_flights(self):
        num_balloons = self.balloon_parameters["num"]
        num_timesteps = self.__num_episode_timesteps()
        self._balloon_flights = np.zeros((num_balloons, 6, num_timesteps))
        z_values = 10 + self._rocketpy_env.elevation + np.arange(num_balloons) * 40
        self._balloon_flights[:, 2, :] = z_values[:, None]

    def step(self, action):
        previous_balloon_positions = self._balloon_states[:, :3].copy()
        previous_rocket_position = self._rocket_states[:3].copy()
        self.current_step += 1

        self._balloon_states = self._balloon_flights[:, :, self.current_step]

        released_mask = self.current_step >= self._balloon_release_at_step
        ground_mask = self._balloon_status[:, 0] == 0
        self._balloon_status[released_mask & ground_mask, 0] = 1

        if not self.rocket_launched:
            _rocket_finished = False
            if action["launch"]:
                self.rocket_launched = True
                self.__get_init_rocket_states(action["launch_inclination_heading"][0], action["launch_inclination_heading"][1])
                self.initial_solution[0] = self.current_step * self.simulation_parameters["time_step"]
                self.__init_rocket_simulation()
        else:
            self._rocket_flight.rocket.roll_control.roll_torque = action["roll"]
            self._rocket_flight.rocket.tvc.gimbal_angle_x = action["tvc"][0]
            self._rocket_flight.rocket.tvc.gimbal_angle_y = action["tvc"][1]
            self._rocket_flight.rocket.throttle_control.throttle = action["throttle"]
            self._rocket_flight.step_simulation()

            _sensor = self._rocket_flight.sensors
            self._rocket_sensors[:3] = _sensor[0].measurement
            self._rocket_sensors[3:6] = _sensor[1].measurement
            self._rocket_sensors[6:12] = _sensor[2].measurement
            self._rocket_states = self._rocket_flight.y_sol[:]
            _rocket_finished = self._rocket_flight._step_state["finished"]

            self._detect_pops(previous_balloon_positions, previous_rocket_position)

        step_record = {
            "time": self.current_step * self.simulation_parameters["time_step"],
            "rocket_states": self._rocket_states.copy().tolist(),
            "balloon_states": self._balloon_states.copy().tolist(),
            "balloon_status": self._balloon_status[:, 0].tolist(),
        }
        if self.trajectories is None:
            self.trajectories = [step_record]
        else:
            self.trajectories.append(step_record)

        _timeout = self.current_step >= self.num_timesteps - 1
        if _timeout:
            # print("Truncated: Reached max time")
            # Post-processing only applies when the rocket was actually launched;
            # reaching max_time before launch leaves _rocket_flight as None.
            if self._rocket_flight is not None:
                self._rocket_flight.post_process_simulation()
                self._rocket_flight.initialize_prints_plots()
        # elif _rocket_finished:
        #     print("Terminated: Rocket flight finished")
        # The rocket flight ending is a true MDP terminal (no value bootstrap);
        # hitting max_time is a time-limit truncation (bootstrap V(s_T)). Keep
        # them mutually exclusive with terminated taking priority.
        terminated = _rocket_finished
        truncated = _timeout and not terminated

        new_count = np.sum(self._balloon_status[:, 0] == 2)
        reward = new_count - self._popped_count
        self._popped_count = new_count

        observation = self._get_obs()
        info = self._get_info()

        _remainder = np.remainder(self.current_step, 0.1 / self.simulation_parameters["time_step"])
        if _remainder == 0 or terminated or truncated:
            self._render_frame()

        return observation, reward, terminated, truncated, info

    @staticmethod
    def _segment_distance_squared_batch(segment_start_a, segment_end_a, segment_start_b, segment_end_b):
        segment_start_a = np.asarray(segment_start_a, dtype=float)
        segment_end_a = np.asarray(segment_end_a, dtype=float)
        segment_start_b = np.asarray(segment_start_b, dtype=float)
        segment_end_b = np.asarray(segment_end_b, dtype=float)

        epsilon = 1e-12
        direction_a = segment_end_a - segment_start_a
        direction_b = segment_end_b - segment_start_b
        offset = segment_start_a - segment_start_b

        n_segments = segment_start_b.shape[0]
        s_param = np.zeros(n_segments)
        t_param = np.zeros(n_segments)

        a_coeff = float(np.dot(direction_a, direction_a))
        e_coeff = np.einsum("ij,ij->i", direction_b, direction_b)
        f_coeff = np.einsum("ij,ij->i", direction_b, offset)

        if a_coeff <= epsilon:
            valid_e = e_coeff > epsilon
            t_param[valid_e] = np.clip(f_coeff[valid_e] / e_coeff[valid_e], 0.0, 1.0)
        else:
            c_coeff = np.einsum("j,ij->i", direction_a, offset)
            degenerate_b = e_coeff <= epsilon
            s_param[degenerate_b] = np.clip(-c_coeff[degenerate_b] / a_coeff, 0.0, 1.0)

            regular = ~degenerate_b
            if np.any(regular):
                b_coeff = np.einsum("j,ij->i", direction_a, direction_b)
                denominator = a_coeff * e_coeff - b_coeff * b_coeff

                non_parallel = regular & (np.abs(denominator) > epsilon)
                s_param[non_parallel] = np.clip(
                    (b_coeff[non_parallel] * f_coeff[non_parallel] - c_coeff[non_parallel] * e_coeff[non_parallel]) / denominator[non_parallel],
                    0.0, 1.0
                )
                t_param[regular] = (b_coeff[regular] * s_param[regular] + f_coeff[regular]) / e_coeff[regular]

                t_too_low = regular & (t_param < 0.0)
                t_param[t_too_low] = 0.0
                s_param[t_too_low] = np.clip(-c_coeff[t_too_low] / a_coeff, 0.0, 1.0)

                t_too_high = regular & (t_param > 1.0)
                t_param[t_too_high] = 1.0
                s_param[t_too_high] = np.clip((b_coeff[t_too_high] - c_coeff[t_too_high]) / a_coeff, 0.0, 1.0)

        closest_point_a = segment_start_a + s_param[:, None] * direction_a
        closest_point_b = segment_start_b + t_param[:, None] * direction_b
        separation = closest_point_a - closest_point_b
        return np.einsum("ij,ij->i", separation, separation)

    def _detect_pops(self, previous_balloon_positions, previous_rocket_position):
        previous_balloon_positions = np.asarray(previous_balloon_positions, dtype=float)
        previous_rocket_position = np.asarray(previous_rocket_position, dtype=float)
        current_balloon_positions = np.asarray(self._balloon_states[:, :3], dtype=float)
        current_rocket_position = np.asarray(self._rocket_states[:3], dtype=float)
        balloon_radius_squared = self.balloon_parameters["radius"] ** 2
        released_mask = self._balloon_status[:, 0] == 1
        if not np.any(released_mask):
            return

        distance_squared = self._segment_distance_squared_batch(
            previous_rocket_position, current_rocket_position,
            previous_balloon_positions[released_mask], current_balloon_positions[released_mask],
        )
        popped_released = distance_squared <= balloon_radius_squared
        released_indices = np.flatnonzero(released_mask)
        self._balloon_status[released_indices[popped_released], 0] = 2

    def _render_frame(self):
        if self.render_mode == "vpython":
            from vpython import arrow, canvas, color, rate, sphere, vector
            if self.render_canvas is None:
                self.render_canvas = canvas(title="Balloon Popping Environment", width=800, height=600, center=vector(0, 0, 0), background=color.white)
                self.render_balloons = [sphere(radius=1.5, color=color.magenta) for _ in range(self.balloon_parameters["num"])]
                self.render_rocket = arrow(pos=vector(0, 0, 0), axis=vector(0, 0, 5), shaftwidth=0.5, color=color.blue)

            status_colors = {0: color.gray(0.5), 1: color.magenta, 2: color.red}
            for balloon, state, status in zip(self.render_balloons, self._balloon_states, self._balloon_status[:, 0]):
                balloon.pos = vector(state[0], state[1], state[2])
                balloon.color = status_colors[int(status)]

            if not np.isnan(self._rocket_states[0]):
                nose_direction = Matrix.transformation(self._rocket_states[6:10]) @ Vector([0, 0, 1])
                self.render_rocket.pos = vector(self._rocket_states[0], self._rocket_states[1], self._rocket_states[2])
                self.render_rocket.axis = vector(nose_direction[0] * 10, nose_direction[1] * 10, nose_direction[2] * 10)
            rate(30)

        elif self.render_mode == "matplotlib":
            if self.render_canvas is None:
                self.render_canvas = plt.figure().add_subplot(projection="3d")
                self.render_balloons = self.render_canvas.scatter(self._balloon_states[:, 0], self._balloon_states[:, 1], self._balloon_states[:, 2], c="magenta")
                self.render_rocket = self.render_canvas.plot(self._rocket_states[0], self._rocket_states[1], self._rocket_states[2], "s", color="blue")
                self.render_canvas.set_xlabel("X position (m)")
                self.render_canvas.set_ylabel("Y position (m)")
                self.render_canvas.set_zlabel("Z position (m)")
                self.render_canvas.set_xlim(self._balloon_flights[:, 0, :].min() - 10, self._balloon_flights[:, 0, :].max() + 10)
                self.render_canvas.set_ylim(self._balloon_flights[:, 1, :].min() - 10, self._balloon_flights[:, 1, :].max() + 10)
                self.render_canvas.set_zlim(0, self._balloon_flights[:, 2, :].max() + 10)

            status_colors = {0: "grey", 1: "magenta", 2: "red"}
            colors = [status_colors[int(status)] for status in self._balloon_status[:, 0]]
            self.render_balloons._offsets3d = (self._balloon_states[:, 0], self._balloon_states[:, 1], self._balloon_states[:, 2])
            self.render_balloons.set_facecolors(colors)
            self.render_rocket[0].set_data([self._rocket_states[0]], [self._rocket_states[1]])
            self.render_rocket[0].set_3d_properties([self._rocket_states[2]])
            self.render_canvas.set_title(f"Time: {self.current_step * self.simulation_parameters['time_step']:.2f} sec\nTotal Reward: {self._popped_count}")
            plt.draw()
            plt.pause(0.001)

    def close(self):
        # print("closing environment")
        pass

    def set_wind_rotation(self, theta):
        """External API (training wrapper): rotate the atmosphere's wind field
        about z by ``theta`` radians for the next reset. Must match the
        rotation applied to the injected balloon tracks, so the wind pushing
        the rocket stays aligned with the wind that shaped the balloon drift
        (they share the same sky).
        """
        self._wind_rotation = float(theta)

    def __capture_base_wind_profile(self):
        """Snapshot the atmosphere's wind-vs-altitude table (one-time)."""
        self._base_wind_profile = None
        source_x = getattr(self._rocketpy_env.wind_velocity_x, "source", None)
        source_y = getattr(self._rocketpy_env.wind_velocity_y, "source", None)
        if isinstance(source_x, np.ndarray) and isinstance(source_y, np.ndarray):
            heights = source_x[:, 0]
            wind_u = source_x[:, 1]
            wind_v = np.interp(heights, source_y[:, 0], source_y[:, 1])
            self._base_wind_profile = (heights, wind_u, wind_v)

    def __apply_wind_rotation(self):
        """Rebuild the wind functions from the cached base profile, rotated by
        the current episode angle. Always reassigns from base so theta=0
        restores the original field."""
        if self._base_wind_profile is None:
            return  # constant/zero wind (standard atmosphere): nothing to rotate
        heights, wind_u, wind_v = self._base_wind_profile
        c, s = np.cos(self._wind_rotation), np.sin(self._wind_rotation)
        rotated_u = c * wind_u - s * wind_v
        rotated_v = s * wind_u + c * wind_v
        self._rocketpy_env.wind_velocity_x = Function(
            np.column_stack([heights, rotated_u]),
            "Height Above Sea Level (m)", "Wind Velocity X (m/s)",
            interpolation="linear", extrapolation="constant",
        )
        self._rocketpy_env.wind_velocity_y = Function(
            np.column_stack([heights, rotated_v]),
            "Height Above Sea Level (m)", "Wind Velocity Y (m/s)",
            interpolation="linear", extrapolation="constant",
        )

    def __create_environment(self):
        gust_enabled = self.environment_parameters["gust"]["enable"]

        # The environment is deterministic when gusts are off, so build it once
        # and reuse it across resets: reloading an Ensemble atmosphere file
        # every episode would dominate reset time. Only the per-episode wind
        # rotation changes between resets.
        if self._rocketpy_env is not None and not gust_enabled:
            self.__apply_wind_rotation()
            return

        self._rocketpy_env = Environment(
            date=self.environment_parameters["date"], latitude=self.environment_parameters["latitude"],
            longitude=self.environment_parameters["longitude"], elevation=self.environment_parameters["elevation"],
            datum="WGS84", timezone="UTC"
        )
        if self.environment_parameters["atmosphere_data_filename"] is None:
            self._rocketpy_env.set_atmospheric_model(type="standard_atmosphere")
        else:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", self.environment_parameters["atmosphere_data_filename"])
            self._rocketpy_env.set_atmospheric_model(type="Ensemble", file=path, dictionary="ECMWF")

        self.__capture_base_wind_profile()
        # Applied before the gust layer: the gust addition below stays
        # unrotated (it is isotropic random noise anyway).
        self.__apply_wind_rotation()

        if self.environment_parameters["gust"]["enable"]:
            gust_param = self.environment_parameters["gust"]
            altitude_nodes = np.arange(0.0, self._rocketpy_env.max_expected_height + gust_param["altitude_spacing"], gust_param["altitude_spacing"])
            x_gust_nodes = self.np_random.uniform(-gust_param["max_gust_speed"], gust_param["max_gust_speed"], size=len(altitude_nodes))
            y_gust_nodes = self.np_random.uniform(-gust_param["max_gust_speed"], gust_param["max_gust_speed"], size=len(altitude_nodes))
            gust_decay = np.exp(-altitude_nodes / gust_param["gust_decay_height"])

            def gust_x(height_asl): return np.interp(height_asl, altitude_nodes, x_gust_nodes * gust_decay)
            def gust_y(height_asl): return np.interp(height_asl, altitude_nodes, y_gust_nodes * gust_decay)
            self._rocketpy_env.add_wind_gust(gust_x, gust_y)

    def __reset_balloon_release_sequence(self):
        n = self.balloon_parameters["num"]
        i = self.balloon_parameters["release_interval"]
        t = self.simulation_parameters["time_step"]
        self._balloon_release_at_step = np.arange(n) * int(i / t)
        self.np_random.shuffle(self._balloon_release_at_step)

    def __get_init_rocket_states(self, inclination, heading):
        t_initial = 0
        x_init, y_init, z_init = 0, 0, self.environment_parameters["elevation"]
        vx_init, vy_init, vz_init = 0, 0, 0
        w1_init, w2_init, w3_init = 0, 0, 0
        e0_init, e1_init, e2_init, e3_init = get_initial_attitude(inclination, heading)

        self.initial_solution = [t_initial, x_init, y_init, z_init, vx_init, vy_init, vz_init, e0_init, e1_init, e2_init, e3_init, w1_init, w2_init, w3_init]

    def __init_rocket_simulation(self):
        tank_cfg = self.rocket_parameters["tank"]
        oxidizer_gas = Fluid(name=tank_cfg["gas"], density=tank_cfg["gas_density"])
        oxidizer_liq = Fluid(name=tank_cfg["liquid"], density=tank_cfg["liquid_density"])

        tank_shape = CylindricalTank(radius=tank_cfg["radius"], height=tank_cfg["height"])
        oxidizer_tank = MassFlowRateBasedTank(
            name="oxidizer_tank", geometry=tank_shape,
            flux_time=(self.initial_solution[0], self.initial_solution[0] + tank_cfg["flux_time"]),
            initial_liquid_mass=tank_cfg["initial_liquid_mass"], initial_gas_mass=tank_cfg["initial_gas_mass"],
            liquid_mass_flow_rate_in=0, liquid_mass_flow_rate_out=tank_cfg["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=0, gas_mass_flow_rate_out=0, liquid=oxidizer_liq, gas=oxidizer_gas
        )

        motor_cfg = self.rocket_parameters["motor"]
        hybrid_motor = HybridMotor(
            thrust_source=motor_cfg["thrust_source"], dry_mass=motor_cfg["dry_mass"], dry_inertia=tuple(motor_cfg["dry_inertia"]),
            center_of_dry_mass_position=motor_cfg["center_of_dry_mass_position"],
            burn_time=(self.initial_solution[0], motor_cfg["burn_time"] + self.initial_solution[0]),
            reshape_thrust_curve=False, grain_number=motor_cfg["grain_number"], grain_separation=motor_cfg["grain_separation"],
            grain_outer_radius=motor_cfg["grain_outer_radius"], grain_initial_inner_radius=motor_cfg["grain_initial_inner_radius"],
            grain_initial_height=motor_cfg["grain_initial_height"], grain_density=motor_cfg["grain_density"],
            nozzle_radius=motor_cfg["nozzle_radius"], throat_radius=motor_cfg["throat_radius"],
            interpolation_method="linear", nozzle_position=motor_cfg["nozzle_position"],
            grains_center_of_mass_position=motor_cfg["grains_center_of_mass_position"], coordinate_system_orientation="nozzle_to_combustion_chamber"
        )
        hybrid_motor.add_tank(tank=oxidizer_tank, position=tank_cfg["tank_position"])

        rocket_cfg = self.rocket_parameters["rocket_body"]
        rocket = Rocket(
            radius=rocket_cfg["radius"], mass=rocket_cfg["mass"], inertia=tuple(rocket_cfg["inertia"]),
            center_of_mass_without_motor=rocket_cfg["center_of_mass_without_motor"],
            power_off_drag=rocket_cfg["power_off_drag"], power_on_drag=rocket_cfg["power_on_drag"],
            coordinate_system_orientation="tail_to_nose", volume=rocket_cfg["volume"]
        )
        rocket.add_motor(hybrid_motor, position=motor_cfg["motor_position"])

        nose_cfg = self.rocket_parameters["nose"]
        rocket.add_nose(length=nose_cfg["length"], kind=nose_cfg["kind"], position=nose_cfg["position"])

        fins_cfg = self.rocket_parameters["fins"]
        if fins_cfg["useFins"]:
            rocket.add_trapezoidal_fins(n=fins_cfg["n"], span=fins_cfg["span"], root_chord=fins_cfg["root_chord"], tip_chord=fins_cfg["tip_chord"], position=fins_cfg["position"])

        sensors_cfg = self.rocket_parameters["sensors"]
        gyro = Gyroscope(sampling_rate=sensors_cfg["sampling_rate"], noise_density=sensors_cfg["gyro_noise_density"], random_walk_density=sensors_cfg["gyro_random_walk_density"], constant_bias=sensors_cfg["gyro_constant_bias"])
        accelerometer = Accelerometer(sampling_rate=sensors_cfg["sampling_rate"], noise_density=sensors_cfg["accelerometer_noise_density"], random_walk_density=sensors_cfg["accelerometer_random_walk_density"], constant_bias=sensors_cfg["accelerometer_constant_bias"], consider_gravity=True)
        gnss = GnssReceiver(sampling_rate=sensors_cfg["sampling_rate"], position_accuracy=sensors_cfg["gnss_position_accuracy"], altitude_accuracy=sensors_cfg["gnss_altitude_accuracy"], velocity_accuracy=sensors_cfg["gnss_velocity_accuracy"])
        rocket.add_sensor(gyro, position=sensors_cfg["gyro_position"])
        rocket.add_sensor(accelerometer, position=sensors_cfg["accelerometer_position"])
        rocket.add_sensor(gnss, position=sensors_cfg["gnss_position"])

        control_cfg = self.rocket_parameters["control"]
        rocket.add_tvc(gimbal_range=control_cfg["gimbal_range"], gimbal_rate_limit=control_cfg["gimbal_rate_limit"], sampling_rate=1 / self.simulation_parameters["time_step"], controller_function=lambda t, sr, s, sh, ov, tvc, sen: (t, tvc.gimbal_angle_x, tvc.gimbal_angle_y), return_controller=False)
        rocket.add_roll_control(max_roll_torque=control_cfg["max_roll_torque"], torque_rate_limit=control_cfg["torque_rate_limit"], sampling_rate=1 / self.simulation_parameters["time_step"], controller_function=lambda t, sr, s, sh, ov, rc, sen: (t, rc.roll_torque), return_controller=False)
        rocket.add_throttle_control(throttle_range=control_cfg["throttle_range"], throttle_rate_limit=control_cfg["throttle_rate_limit"], sampling_rate=1 / self.simulation_parameters["time_step"], controller_function=lambda t, sr, s, sh, ov, tc, sen: (t, tc.throttle), return_controller=False)

        self._rocket_flight = Flight(
            rocket=rocket, environment=self._rocketpy_env, rail_length=0.01,
            initial_solution=self.initial_solution, max_time=self.simulation_parameters["max_time"],
            time_overshoot=False, verbose=False, run_simulation=False, ode_solver="RK45"
        )


def get_initial_attitude(inclination, heading):
    psi_init = np.radians(-heading)
    theta_init = np.radians(inclination - 90)
    phi_init = 0
    return euler313_to_quaternions(phi_init, theta_init, psi_init)
