import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from rocketpy import Environment, Flight, Rocket
from rocketpy.mathutils.vector_matrix import Matrix, Vector
from rocketpy.motors import CylindricalTank, Fluid, HybridMotor
from rocketpy.motors.tank import MassFlowRateBasedTank
from rocketpy.sensors.accelerometer import Accelerometer
from rocketpy.sensors.gnss_receiver import GnssReceiver
from rocketpy.sensors.gyroscope import Gyroscope
from rocketpy.tools import euler313_to_quaternions


class BalloonPoppingEnv(gym.Env):
    """Balloon popping with a fixed balloon field.

    Balloons sit at constant positions and are all active from t=0, so there is
    no Monte Carlo trajectory generation, no release schedule, and no per-step
    balloon integration. The field is described by an (N, 3) position array,
    which ``update_balloons`` can replace between episodes.
    """

    metadata = {"render_modes": ["vpython", "matplotlib"]}

    def __init__(self, render_mode, parameters):
        self.scenario_parameters = parameters["scenario"]
        self.environment_parameters = parameters["environment"]
        self.simulation_parameters = parameters["simulation"]
        self.balloon_parameters = parameters["balloon"]
        self.rocket_parameters = parameters["rocket"]

        self.num_timesteps = len(
            np.arange(
                0,
                self.simulation_parameters["max_time"],
                self.simulation_parameters["time_step"],
            )
        )

        # ActiveRocketPy flight class for the rocket
        self._rocket_flight = None
        self._rocketpy_env = None

        # initial solution: [time, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        self.initial_solution = None
        # (posX, posY, posZ, velX, velY, velZ, e0, e1, e2, e3, w1, w2, w3)
        self._rocket_states = np.full(13, np.nan)
        # (gyroX, gyroY, gyroZ, accX, accY, accZ, posX, posY, posZ, velX, velY, velZ)
        self._rocket_sensors = np.full(12, np.nan)
        # save trajectories for logging
        self.trajectories = None

        # attributes for step()
        self.rocket_launched = False
        self.current_step = 0
        self._popped_count = 0

        # Balloon field: sets num, _balloon_positions, _balloon_states,
        # _balloon_status and the balloon-dependent observation spaces.
        self.update_balloons(self.balloon_parameters.get("positions"))

        # tvc, roll, and throttling actions
        self.action_space = spaces.Dict(
            {
                "launch": spaces.Box(low=0, high=1, shape=(), dtype=bool),
                "launch_inclination_heading": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([90, 360]),
                    shape=(2,),
                    dtype=np.float64,
                ),
                "tvc": spaces.Box(
                    low=-self.rocket_parameters["control"]["gimbal_range"] * np.ones(2),
                    high=self.rocket_parameters["control"]["gimbal_range"] * np.ones(2),
                    dtype=np.float64,
                ),
                "throttle": spaces.Box(
                    low=self.rocket_parameters["control"]["throttle_range"][0],
                    high=self.rocket_parameters["control"]["throttle_range"][1],
                    shape=(),
                    dtype=np.float64,
                ),
                "roll": spaces.Box(
                    low=-self.rocket_parameters["control"]["max_roll_torque"],
                    high=self.rocket_parameters["control"]["max_roll_torque"],
                    shape=(),
                    dtype=np.float64,
                ),
            }
        )

        # Graphics-related attributes
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None

    def update_balloons(self, positions=None):
        """Replace the balloon field.

        Parameters
        ----------
        positions : array-like or None
            Shape (N, 3) of world-frame [x, y, z] in metres, z above sea level.
            None falls back to ``balloon["num"]`` balloons stacked above the
            origin, 40 m apart in altitude.

        Changing N rebuilds the observation space, so call this before the env
        is wrapped by a vectoriser; changing it mid-run would break the shapes
        those wrappers cached. Takes effect on the next ``reset``.
        """
        if positions is None:
            num = self.balloon_parameters["num"]
            elevation = self.environment_parameters["elevation"]
            positions = np.zeros((num, 3))
            positions[:, 2] = 10.0 + elevation + np.arange(num) * 40.0
        else:
            positions = np.asarray(positions, dtype=float).reshape(-1, 3)

        num = len(positions)
        self.balloon_parameters["num"] = num
        self._balloon_positions = positions

        # Static field: velocities are always zero, so the state array is built
        # once and never integrated.
        self._balloon_states = np.zeros((num, 6))
        self._balloon_states[:, :3] = positions
        self._balloon_status = np.ones((num, 1), dtype=int)

        self.observation_space = spaces.Dict(
            {
                "simulation_time": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(), dtype=np.float64
                ),
                "balloon_status": spaces.MultiDiscrete(3 * np.ones((num, 1), dtype=int)),
                "balloon_states": spaces.Box(
                    low=-np.inf * np.ones((num, 6)),
                    high=np.inf * np.ones((num, 6)),
                    dtype=np.float64,
                ),
                "rocket_sensors": spaces.Box(
                    low=-np.inf * np.ones(12),
                    high=np.inf * np.ones(12),
                    dtype=np.float64,
                ),
            }
        )

    def _get_obs(self):
        return {
            "simulation_time": self.current_step
            * self.simulation_parameters["time_step"],
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
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.__create_environment()
        self._rocket_flight = None
        self.initial_solution = None

        # All balloons are active from the start; positions never change.
        self._balloon_status = np.ones((self.balloon_parameters["num"], 1), dtype=int)
        self._balloon_states = np.zeros((self.balloon_parameters["num"], 6))
        self._balloon_states[:, :3] = self._balloon_positions

        self._rocket_sensors = np.full(12, np.nan)
        self._rocket_states = np.full(13, np.nan)
        self.trajectories = None

        self.rocket_launched = False
        self.current_step = 0
        self._popped_count = 0

        observation = self._get_obs()
        info = self._get_info()

        self.render_canvas = None
        self.render_balloons = None
        self.render_rocket = None
        self._render_frame()

        return observation, info

    def step(self, action):
        previous_rocket_position = self._rocket_states[:3].copy()
        self.current_step += 1

        if not self.rocket_launched:
            _rocket_finished = False
            if action["launch"]:  # Init rocket flight with first launch action
                self.rocket_launched = True
                self.__get_init_rocket_states(
                    action["launch_inclination_heading"][0],
                    action["launch_inclination_heading"][1],
                )
                self.initial_solution[0] = (
                    self.current_step * self.simulation_parameters["time_step"]
                )
                self.__init_rocket_simulation()
        else:  # Apply action to step the rocket simulation and get sensor measurements
            self._rocket_flight.rocket.roll_control.roll_torque = action["roll"]
            self._rocket_flight.rocket.tvc.gimbal_angle_x = action["tvc"][0]
            self._rocket_flight.rocket.tvc.gimbal_angle_y = action["tvc"][1]
            self._rocket_flight.rocket.throttle_control.throttle = action["throttle"]
            try:
                self._rocket_flight.step_simulation()
                _sensor = self._rocket_flight.sensors
                self._rocket_sensors[:3] = _sensor[0].measurement  # gyro
                self._rocket_sensors[3:6] = _sensor[1].measurement  # accel
                self._rocket_sensors[6:12] = _sensor[2].measurement  # gnss
                self._rocket_states = self._rocket_flight.y_sol[:]
                _rocket_finished = self._rocket_flight._step_state["finished"]

                self._detect_pops(previous_rocket_position)
            except Exception as exc:
                # rocketpy's flight integrator can raise on numerical edge cases
                # (e.g. the cubic impact-time solver dividing by zero at landing).
                # It fires while detecting ground impact, so ending the flight is
                # the correct outcome - and it must never crash the worker/training.
                # Keep the last valid rocket state (not overwritten above).
                print(
                    f"[BalloonPoppingEnv] flight step failed "
                    f"({type(exc).__name__}: {exc}); ending episode."
                )
                _rocket_finished = True

        # Append rocket states to trajectories for logging
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

        # An episode is done iff it reaches max time or the flight ends
        _timeout = self.current_step >= self.num_timesteps - 1
        if _timeout:
            self._rocket_flight.post_process_simulation()
            self._rocket_flight.initialize_prints_plots()
        terminated = _timeout or _rocket_finished

        # Calculate reward based on newly popped balloons at this step
        new_count = np.sum(self._balloon_status[:, 0] == 2)
        reward = new_count - self._popped_count
        self._popped_count = new_count

        observation = self._get_obs()
        info = self._get_info()

        # Render every 0.1 sec or on termination to balance visualization and performance
        _remainder = np.remainder(
            self.current_step, 0.1 / self.simulation_parameters["time_step"]
        )
        if _remainder == 0 or terminated:
            self._render_frame()

        return observation, reward, terminated, False, info

    def _detect_pops(self, previous_rocket_position):
        """Pop balloons whose centre lies within a radius of the swept path.

        The balloons are stationary, so this reduces to point-to-segment
        distance over the rocket's motion this timestep -- still a swept test,
        which a plain endpoint check would miss at high speed.
        """
        previous_rocket_position = np.asarray(previous_rocket_position, dtype=float)
        current_rocket_position = np.asarray(self._rocket_states[:3], dtype=float)
        if not np.all(np.isfinite(previous_rocket_position)):
            return

        alive = self._balloon_status[:, 0] == 1
        if not np.any(alive):
            return

        travel = current_rocket_position - previous_rocket_position
        offsets = self._balloon_positions[alive] - previous_rocket_position

        travel_squared = float(np.dot(travel, travel))
        if travel_squared > 1e-12:
            fraction = np.clip((offsets @ travel) / travel_squared, 0.0, 1.0)
            offsets = offsets - fraction[:, None] * travel

        distance_squared = np.einsum("ij,ij->i", offsets, offsets)
        popped = distance_squared <= self.balloon_parameters["radius"] ** 2
        self._balloon_status[np.flatnonzero(alive)[popped], 0] = 2

    def _render_frame(self):
        if self.render_mode == "vpython":
            from vpython import arrow, canvas, color, rate, sphere, vector

            if self.render_canvas is None:
                self.render_canvas = canvas(
                    title="Balloon Popping Environment",
                    width=800,
                    height=600,
                    center=vector(0, 0, 0),
                    background=color.white,
                )
                self.render_balloons = [
                    sphere(radius=self.balloon_parameters["radius"], color=color.magenta)
                    for _ in range(self.balloon_parameters["num"])
                ]
                self.render_rocket = arrow(
                    pos=vector(0, 0, 0),
                    axis=vector(0, 0, 5),
                    shaftwidth=0.5,
                    color=color.blue,
                )

            # Status colors: 1=magenta (active), 2=red (popped)
            status_colors = {1: color.magenta, 2: color.red}
            for balloon, state, status in zip(
                self.render_balloons, self._balloon_states, self._balloon_status[:, 0]
            ):
                balloon.pos = vector(state[0], state[1], state[2])
                balloon.color = status_colors[int(status)]

            if not np.isnan(self._rocket_states[0]):
                # Convert quaternion to rocket nose direction vector
                nose_direction = Matrix.transformation(
                    self._rocket_states[6:10]
                ) @ Vector([0, 0, 1])

                self.render_rocket.pos = vector(
                    self._rocket_states[0],
                    self._rocket_states[1],
                    self._rocket_states[2],
                )
                self.render_rocket.axis = vector(
                    nose_direction[0] * 10,
                    nose_direction[1] * 10,
                    nose_direction[2] * 10,
                )

            rate(30)
        elif self.render_mode == "matplotlib":
            if self.render_canvas is None:
                self.render_canvas = plt.figure().add_subplot(projection="3d")
                self.render_balloons = self.render_canvas.scatter(
                    self._balloon_states[:, 0],
                    self._balloon_states[:, 1],
                    self._balloon_states[:, 2],
                    c="magenta",
                )
                self.render_rocket = self.render_canvas.plot(
                    self._rocket_states[0],
                    self._rocket_states[1],
                    self._rocket_states[2],
                    "s",
                    color="blue",
                )
                self.render_canvas.set_xlabel("X position (m)")
                self.render_canvas.set_ylabel("Y position (m)")
                self.render_canvas.set_zlabel("Z position (m)")
                self.render_canvas.set_xlim(
                    self._balloon_positions[:, 0].min() - 10,
                    self._balloon_positions[:, 0].max() + 10,
                )
                self.render_canvas.set_ylim(
                    self._balloon_positions[:, 1].min() - 10,
                    self._balloon_positions[:, 1].max() + 10,
                )
                self.render_canvas.set_zlim(0, self._balloon_positions[:, 2].max() + 10)

            status_colors = {1: "magenta", 2: "red"}
            colors = [
                status_colors[int(status)] for status in self._balloon_status[:, 0]
            ]
            self.render_balloons._offsets3d = (
                self._balloon_states[:, 0],
                self._balloon_states[:, 1],
                self._balloon_states[:, 2],
            )
            self.render_balloons.set_facecolors(colors)
            self.render_rocket[0].set_data(
                [self._rocket_states[0]], [self._rocket_states[1]]
            )
            self.render_rocket[0].set_3d_properties([self._rocket_states[2]])
            self.render_canvas.set_title(
                f"Time: {self.current_step * self.simulation_parameters['time_step']:.2f} sec\n"
                f"Total Reward: {self._popped_count}"
            )
            plt.draw()
            plt.pause(0.001)
        else:
            pass

    def close(self):
        print("closing environment")

    def __create_environment(self):
        self._rocketpy_env = Environment(
            date=self.environment_parameters["date"],
            latitude=self.environment_parameters["latitude"],
            longitude=self.environment_parameters["longitude"],
            elevation=self.environment_parameters["elevation"],
            datum="WGS84",
            timezone="UTC",
        )
        if self.environment_parameters["atmosphere_data_filename"] is None:
            self._rocketpy_env.set_atmospheric_model(type="standard_atmosphere")
        else:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "data",
                self.environment_parameters["atmosphere_data_filename"],
            )
            self._rocketpy_env.set_atmospheric_model(
                type="Ensemble",
                file=path,
                dictionary="ECMWF",
            )

        # Add gust after setting atmospheric model.
        if self.environment_parameters["gust"]["enable"]:
            gust_param = self.environment_parameters["gust"]

            altitude_nodes = np.arange(
                0.0,
                self._rocketpy_env.max_expected_height + gust_param["altitude_spacing"],
                gust_param["altitude_spacing"],
            )
            x_gust_nodes = self.np_random.uniform(
                -gust_param["max_gust_speed"],
                gust_param["max_gust_speed"],
                size=len(altitude_nodes),
            )
            y_gust_nodes = self.np_random.uniform(
                -gust_param["max_gust_speed"],
                gust_param["max_gust_speed"],
                size=len(altitude_nodes),
            )
            # Exponential decay of gust speed with altitude
            gust_decay = np.exp(-altitude_nodes / gust_param["gust_decay_height"])

            def gust_x(height_asl):
                # X direction = East
                return np.interp(height_asl, altitude_nodes, x_gust_nodes * gust_decay)

            def gust_y(height_asl):
                # Y direction = North
                return np.interp(height_asl, altitude_nodes, y_gust_nodes * gust_decay)

            self._rocketpy_env.add_wind_gust(gust_x, gust_y)

    def __get_init_rocket_states(self, inclination, heading):
        e0_init, e1_init, e2_init, e3_init = get_initial_attitude(inclination, heading)

        # [t, x, y, z, vx, vy, vz, e0, e1, e2, e3, w1, w2, w3]
        self.initial_solution = [
            0,
            0,
            0,
            self.environment_parameters["elevation"],
            0,
            0,
            0,
            e0_init,
            e1_init,
            e2_init,
            e3_init,
            0,
            0,
            0,
        ]

    def __init_rocket_simulation(self):
        # Create tank fluids from parameters
        tank_cfg = self.rocket_parameters["tank"]
        oxidizer_gas = Fluid(name=tank_cfg["gas"], density=tank_cfg["gas_density"])
        oxidizer_liq = Fluid(name=tank_cfg["liquid"], density=tank_cfg["liquid_density"])

        # Create tank from parameters
        tank_shape = CylindricalTank(
            radius=tank_cfg["radius"], height=tank_cfg["height"]
        )
        oxidizer_tank = MassFlowRateBasedTank(
            name="oxidizer_tank",
            geometry=tank_shape,
            flux_time=(
                self.initial_solution[0],
                self.initial_solution[0] + tank_cfg["flux_time"],
            ),
            initial_liquid_mass=tank_cfg["initial_liquid_mass"],
            initial_gas_mass=tank_cfg["initial_gas_mass"],
            liquid_mass_flow_rate_in=0,
            liquid_mass_flow_rate_out=tank_cfg["liquid_mass_flow_rate_out"],
            gas_mass_flow_rate_in=0,
            gas_mass_flow_rate_out=0,
            liquid=oxidizer_liq,
            gas=oxidizer_gas,
        )

        # Create motor from parameters
        motor_cfg = self.rocket_parameters["motor"]
        hybrid_motor = HybridMotor(
            thrust_source=motor_cfg["thrust_source"],
            dry_mass=motor_cfg["dry_mass"],
            dry_inertia=tuple(motor_cfg["dry_inertia"]),
            center_of_dry_mass_position=motor_cfg["center_of_dry_mass_position"],
            burn_time=(
                self.initial_solution[0],
                motor_cfg["burn_time"] + self.initial_solution[0],
            ),
            reshape_thrust_curve=False,
            grain_number=motor_cfg["grain_number"],
            grain_separation=motor_cfg["grain_separation"],
            grain_outer_radius=motor_cfg["grain_outer_radius"],
            grain_initial_inner_radius=motor_cfg["grain_initial_inner_radius"],
            grain_initial_height=motor_cfg["grain_initial_height"],
            grain_density=motor_cfg["grain_density"],
            nozzle_radius=motor_cfg["nozzle_radius"],
            throat_radius=motor_cfg["throat_radius"],
            interpolation_method="linear",
            nozzle_position=motor_cfg["nozzle_position"],
            grains_center_of_mass_position=motor_cfg["grains_center_of_mass_position"],
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )
        hybrid_motor.add_tank(tank=oxidizer_tank, position=tank_cfg["tank_position"])

        # Create rocket body from parameters
        rocket_cfg = self.rocket_parameters["rocket_body"]
        rocket = Rocket(
            radius=rocket_cfg["radius"],
            mass=rocket_cfg["mass"],
            inertia=tuple(rocket_cfg["inertia"]),
            center_of_mass_without_motor=rocket_cfg["center_of_mass_without_motor"],
            power_off_drag=rocket_cfg["power_off_drag"],
            power_on_drag=rocket_cfg["power_on_drag"],
            coordinate_system_orientation="tail_to_nose",
            volume=rocket_cfg["volume"],
        )
        rocket.add_motor(hybrid_motor, position=motor_cfg["motor_position"])

        # Add nose from parameters
        nose_cfg = self.rocket_parameters["nose"]
        rocket.add_nose(
            length=nose_cfg["length"],
            kind=nose_cfg["kind"],
            position=nose_cfg["position"],
        )

        # Add fins from parameters
        fins_cfg = self.rocket_parameters["fins"]
        if fins_cfg["useFins"]:
            rocket.add_trapezoidal_fins(
                n=fins_cfg["n"],
                span=fins_cfg["span"],
                root_chord=fins_cfg["root_chord"],
                tip_chord=fins_cfg["tip_chord"],
                position=fins_cfg["position"],
            )

        # Add sensors from parameters
        sensors_cfg = self.rocket_parameters["sensors"]
        gyro = Gyroscope(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["gyro_noise_density"],
            random_walk_density=sensors_cfg["gyro_random_walk_density"],
            constant_bias=sensors_cfg["gyro_constant_bias"],
        )
        accelerometer = Accelerometer(
            sampling_rate=sensors_cfg["sampling_rate"],
            noise_density=sensors_cfg["accelerometer_noise_density"],
            random_walk_density=sensors_cfg["accelerometer_random_walk_density"],
            constant_bias=sensors_cfg["accelerometer_constant_bias"],
            consider_gravity=True,
        )
        gnss = GnssReceiver(
            sampling_rate=sensors_cfg["sampling_rate"],
            position_accuracy=sensors_cfg["gnss_position_accuracy"],
            altitude_accuracy=sensors_cfg["gnss_altitude_accuracy"],
            velocity_accuracy=sensors_cfg["gnss_velocity_accuracy"],
        )
        rocket.add_sensor(gyro, position=sensors_cfg["gyro_position"])
        rocket.add_sensor(accelerometer, position=sensors_cfg["accelerometer_position"])
        rocket.add_sensor(gnss, position=sensors_cfg["gnss_position"])

        # Add control systems from parameters
        control_cfg = self.rocket_parameters["control"]
        sampling_rate = 1 / self.simulation_parameters["time_step"]

        def tvc_controller_function(
            time, sampling_rate, state, state_history, observed_variables, tvc, sensors
        ):
            return time, tvc.gimbal_angle_x, tvc.gimbal_angle_y

        rocket.add_tvc(
            gimbal_range=control_cfg["gimbal_range"],
            gimbal_rate_limit=control_cfg["gimbal_rate_limit"],
            sampling_rate=sampling_rate,
            controller_function=tvc_controller_function,
            return_controller=False,
        )

        def roll_controller_function(
            time,
            sampling_rate,
            state,
            state_history,
            observed_variables,
            roll_control,
            sensors,
        ):
            return time, roll_control.roll_torque

        rocket.add_roll_control(
            max_roll_torque=control_cfg["max_roll_torque"],
            torque_rate_limit=control_cfg["torque_rate_limit"],
            sampling_rate=sampling_rate,
            controller_function=roll_controller_function,
            return_controller=False,
        )

        def throttle_controller_function(
            time,
            sampling_rate,
            state,
            state_history,
            observed_variables,
            throttle_control,
            sensors,
        ):
            return time, throttle_control.throttle

        rocket.add_throttle_control(
            throttle_range=control_cfg["throttle_range"],
            throttle_rate_limit=control_cfg["throttle_rate_limit"],
            sampling_rate=sampling_rate,
            controller_function=throttle_controller_function,
            return_controller=False,
        )

        self._rocket_flight = Flight(
            rocket=rocket,
            environment=self._rocketpy_env,
            # No rail since we directly set initial conditions to simulate launch
            rail_length=0.01,
            initial_solution=self.initial_solution,
            max_time=self.simulation_parameters["max_time"],
            time_overshoot=False,
            verbose=False,
            run_simulation=False,
            ode_solver="RK45",
        )


# Helper function to convert inclination and heading to initial attitude quaternions
def get_initial_attitude(inclination, heading):
    # Precession / Heading Angle
    psi_init = np.radians(-heading)
    # Nutation / Attitude Angle
    theta_init = np.radians(inclination - 90)
    # Spin / Bank Angle
    phi_init = 0

    # 3-1-3 Euler Angles to Euler Parameters
    e0_init, e1_init, e2_init, e3_init = euler313_to_quaternions(
        phi_init, theta_init, psi_init
    )
    return e0_init, e1_init, e2_init, e3_init
