import gym
from gym import error, spaces, utils
from gym.utils import seeding
import logging
import numpy as np

class CO2VentilationSimulatorEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.step_logger = logging.getLogger("StepLogger")
        self.__version__ = "0.0.1"
        self.logger.info(f"CO2VentilationSimulatorEnv - Version {self.__version__}")

        # Define the action_space
        # 0=VentilationFanSpeed1
        # 1=VentilationFanSpeed2
        # 2=VentilationFanSpeed3
        # 3=VentilationFanSpeed4
        self.action_space = spaces.Discrete(4)

        # Define the observation_space
        # First dimension is VentilationFanSpeed (0..3)
        # Second dimension is CO2 level in the air (400...3000)
        # Third dimension is CO2 change from previous state (-100..100)
        low = np.array([0, 400, -100])
        high = np.array([3, 3000, 100])
        self.observation_space = spaces.Box(low, high)

        self.curr_iteration = 0
        self.current_co2_level = 400
        self.previous_co2_level = 400
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.curr_step += 1
        self.logger.info("")
        self.logger.info(f"Iteration #{self.curr_iteration} Step #{self.curr_step}")
        t0_ventilation_speed, t0_co2_level, t0_co2_diff = self.state

        # Execute action on environment (change ventilation fan speed)
        self._execute_action(action)
        
        # Wait for environment to transition to next state
        self._transition_to_next_state()

        # Get reward for new state
        reward = self._get_reward(self.current_co2_level, self.current_ventilation_speed, t0_ventilation_speed)
        self.total_reward += reward

        self.logger.info(f"Reward={reward}, Total reward={self.total_reward}")
        self.step_logger.info(f"{self.curr_iteration},{self.curr_step},{self.current_ventilation_speed + 1},{reward},{self.current_co2_level}")

        done = False

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.curr_iteration += 1
        self.curr_step = 0
        self.total_reward = 0.0
        ventilation_speed = 0   # VentilationFanSpeed1
        co2_level = self.current_co2_level
        co2_diff = self.current_co2_level - self.previous_co2_level
        self.state = (ventilation_speed, co2_level, co2_diff)
        return np.array(self.state)

    def render(self, mode='human'):
        ventilation_speed, co2_level, co2_diff = self.state
        self.logger.info(f"Environment state: Fan speed={ventilation_speed + 1}, CO2={co2_level}, CO2Diff={co2_diff}")

    def _execute_action(self, action):
        self.logger.info(f"Executing action, setting fan speed to {action + 1}")
        self.current_ventilation_speed = action

    def _transition_to_next_state(self):
        self.logger.info ("Waiting for environment to respond to action...")
                
        # Compute next state
        new_co2_diff = 0
        #new_co2_diff = 20 - (self._get_ventilation_volume(self.current_ventilation_speed) * 50)
        new_co2_level = self.current_co2_level + new_co2_diff
        if new_co2_level < 400:
            new_co2_level = 400
        elif new_co2_level > 3000:
            new_co2_level = 3000

        self._update_co2_level(new_co2_level)

        # Update environment state
        t0_ventilation_speed, t0_co2_level, t0_co2_diff = self.state
        co2_diff = self.current_co2_level - t0_co2_level
        self.state = (self.current_ventilation_speed, self.current_co2_level, co2_diff)

    def _get_reward(self, t1_co2_level, current_ventilation_speed, previous_ventilation_speed):
        if t1_co2_level < 900:
            reward = 1.0
        elif t1_co2_level < 950:
            reward = 0.9
        elif t1_co2_level < 1000:
            reward = 0.8
        elif t1_co2_level < 1200:
            reward = 0.4
        elif t1_co2_level < 1500:
            reward = -0.2
        else:
            reward = -0.6

        # Give penalty for energy consumption
        # Need to add outdoor / indoor temperature based calculation, but keep it simple for now
        ventilation_cost = self._get_ventilation_cost(current_ventilation_speed)
        reward = reward - ventilation_cost

        # Give a small penalty for changing ventilation fan speed (but, not for the 1st step in an Episode)
        if self.curr_step > 1:
            if current_ventilation_speed != previous_ventilation_speed:
                reward = reward - 0.1

        return reward

    def _update_co2_level(self, co2_level):
        self.previous_co2_level = self.current_co2_level
        self.current_co2_level = co2_level
        if self.previous_co2_level == 0:
            self.previous_co2_level = self.current_co2_level

    def _get_ventilation_volume(self, ventilation_speed):
        ventilation_speed_volume = [0.1, 0.2, 0.5, 1.0]
        ventilation_volume = ventilation_speed_volume[ventilation_speed]
        return ventilation_volume
        
    def _get_ventilation_cost(self, ventilation_speed):
        ventilation_speed_cost = [0.0, 0.2, 0.4, 0.8]
        ventilation_cost = ventilation_speed_cost[ventilation_speed]
        return ventilation_cost