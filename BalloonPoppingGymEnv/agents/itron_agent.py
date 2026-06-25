import logging
from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.agents.gnc.estimator import Estimator
from BalloonPoppingGymEnv.agents.gnc.selector import Selector
from BalloonPoppingGymEnv.agents.gnc.navigator import Navigator
from BalloonPoppingGymEnv.agents.gnc.controller import Controller
from BalloonPoppingGymEnv.utils.schema import Schema


class ITronAgent(BaseAgent):
    def __init__(self, given_parameters):
        """
        Initializes the agent with environment and rocket configurations.

        Parameters
        ----------
        given_parameters : dict
            Nested configuration metadata structured as follows:
                - environment:
                    date : list[int] -> [year, month, day, hour]
                    latitude / longitude / elevation : float -> [deg, deg, m]
                - simulation:
                    time_step / max_time : float -> [s, s]
                - balloon:
                    release_interval / num / radius / mass : [s, int, m, kg]
                - rocket:
                    tank : liquid, gas parameters and initial mass [kg, kg/s, m]
                    motor : thrust_source [N], burn_time [s], and geometric specs
                    rocket_body : structural mass [kg] and inertia [kg·m²]
                    nose / fins : aerodynamic shapes and assembly positions [m]
                    sensors : sampling_rate [Hz] and noise parameters
                    control : gimbal_range [deg], max_roll_torque [Nm], limits
        """
        super().__init__(given_parameters)
        self.logger = logging.getLogger(__name__)

        # Initialize GNC components
        self.estimator = Estimator(given_parameters)
        self.selector = Selector(given_parameters)
        self.navigator = Navigator(given_parameters)
        self.controller = Controller(given_parameters)

    def reset(self) -> None:
        self.estimator.reset()
        self.selector.reset()
        self.navigator.reset()
        self.controller.reset()

    def get_action(self, observation: dict) -> dict:
        rocket_state = self.estimator.update(observation)
        balloon_state = self.selector.select(observation, rocket_state)
        desired_rates, desired_throttle = self.navigator.compute(balloon_state, rocket_state)
        tvc, roll, throttle = self.controller.compute(observation, desired_rates, desired_throttle)

        # Set launch parameters
        t = observation[Schema.Observation.SIMULATION_TIME]
        is_launched = t >= self.selector.get_launch_time(observation)
        launch_inclination_heading = self.selector.get_launch_heading(observation)

        return {
            "launch": is_launched,
            "launch_inclination_heading": launch_inclination_heading,
            "tvc": tvc,
            "roll": roll,
            "throttle": throttle,
        }
