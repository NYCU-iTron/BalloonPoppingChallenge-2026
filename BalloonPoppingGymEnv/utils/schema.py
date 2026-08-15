from enum import Enum


class Schema:
    class Observation(str, Enum):
        SIMULATION_TIME = "simulation_time"
        BALLOON_STATUS = "balloon_status"
        BALLOON_STATES = "balloon_states"
        ROCKET_SENSORS = "rocket_sensors"

    class Given:
        class Section(str, Enum):
            ENVIRONMENT = "environment"
            SIMULATION = "simulation"
            BALLOON = "balloon"
            ROCKET = "rocket"

        class Environment(str, Enum):
            DATE = "date"
            LATITUDE = "latitude"
            LONGITUDE = "longitude"
            ELEVATION = "elevation"

        class Simulation(str, Enum):
            TIME_STEP = "time_step"
            MAX_TIME = "max_time"

        class Balloon(str, Enum):
            RELEASE_INTERVAL = "release_interval"
            NUM = "num"
            RADIUS = "radius"
            MASS = "mass"

        class Rocket(str, Enum):
            TANK = "tank"
            MOTOR = "motor"
            ROCKET_BODY = "rocket_body"
            NOSE = "nose"
            FINS = "fins"
            SENSORS = "sensors"
            CONTROL = "control"

        class Tank(str, Enum):
            LIQUID = "liquid"
            LIQUID_DENSITY = "liquid_density"
            GAS = "gas"
            GAS_DENSITY = "gas_density"
            RADIUS = "radius"
            HEIGHT = "height"
            FLUX_TIME = "flux_time"
            INITIAL_LIQUID_MASS = "initial_liquid_mass"
            INITIAL_GAS_MASS = "initial_gas_mass"
            LIQUID_MASS_FLOW_RATE_OUT = "liquid_mass_flow_rate_out"
            TANK_POSITION = "tank_position"

        class Motor(str, Enum):
            THRUST_SOURCE = "thrust_source"
            DRY_MASS = "dry_mass"
            DRY_INERTIA = "dry_inertia"
            CENTER_OF_DRY_MASS_POSITION = "center_of_dry_mass_position"
            BURN_TIME = "burn_time"
            GRAIN_NUMBER = "grain_number"
            GRAIN_SEPARATION = "grain_separation"
            GRAIN_OUTER_RADIUS = "grain_outer_radius"
            GRAIN_INITIAL_INNER_RADIUS = "grain_initial_inner_radius"
            GRAIN_INITIAL_HEIGHT = "grain_initial_height"
            GRAIN_DENSITY = "grain_density"
            NOZZLE_RADIUS = "nozzle_radius"
            THROAT_RADIUS = "throat_radius"
            NOZZLE_POSITION = "nozzle_position"
            GRAINS_CENTER_OF_MASS_POSITION = "grains_center_of_mass_position"
            MOTOR_POSITION = "motor_position"

        class RocketBody(str, Enum):
            RADIUS = "radius"
            MASS = "mass"
            INERTIA = "inertia"
            CENTER_OF_MASS_WITHOUT_MOTOR = "center_of_mass_without_motor"
            POWER_OFF_DRAG = "power_off_drag"
            POWER_ON_DRAG = "power_on_drag"
            VOLUME = "volume"

        class Nose(str, Enum):
            LENGTH = "length"
            KIND = "kind"
            POSITION = "position"

        class Fins(str, Enum):
            USE_FINS = "useFins"  # Matches environment exact camelCase key
            N = "n"
            SPAN = "span"
            ROOT_CHORD = "root_chord"
            TIP_CHORD = "tip_chord"
            POSITION = "position"

        class Sensors(str, Enum):
            SAMPLING_RATE = "sampling_rate"
            GYRO_POSITION = "gyro_position"
            GYRO_NOISE_DENSITY = "gyro_noise_density"
            GYRO_RANDOM_WALK_DENSITY = "gyro_random_walk_density"
            GYRO_CONSTANT_BIAS = "gyro_constant_bias"
            ACCELEROMETER_POSITION = "accelerometer_position"
            ACCELEROMETER_NOISE_DENSITY = "accelerometer_noise_density"
            ACCELEROMETER_RANDOM_WALK_DENSITY = "accelerometer_random_walk_density"
            ACCELEROMETER_CONSTANT_BIAS = "accelerometer_constant_bias"
            GNSS_POSITION = "gnss_position"
            GNSS_POSITION_ACCURACY = "gnss_position_accuracy"
            GNSS_ALTITUDE_ACCURACY = "gnss_altitude_accuracy"
            GNSS_VELOCITY_ACCURACY = "gnss_velocity_accuracy"

        class Control(str, Enum):
            MAX_GIMBAL_ANGLE = "max_gimbal_angle"
            GIMBAL_RATE_LIMIT = "gimbal_rate_limit"
            MAX_ROLL_TORQUE = "max_roll_torque"
            TORQUE_RATE_LIMIT = "torque_rate_limit"
            THROTTLE_RANGE = "throttle_range"
            THROTTLE_RATE_LIMIT = "throttle_rate_limit"
