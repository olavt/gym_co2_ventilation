import gym
from gym import error, spaces, utils
from gym.utils import seeding
import logging
import numpy as np
import random

class CO2VentilationSimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.logger = logging.getLogger("Logger")
        self.step_logger = logging.getLogger("StepLogger")
        self.__version__ = "0.0.1"
        self.logger.info(f"CO2VentilationSimpleEnv - Version {self.__version__}")

        # Define the action_space
        # 0=VentilationFanSpeed1
        # 1=VentilationFanSpeed2
        # 2=VentilationFanSpeed3
        # 3=VentilationFanSpeed4
        self.action_space = spaces.Discrete(4)

        # Define the observation_space
        # First dimension is VentilationFanSpeed (0..3)
        low = np.array([0, 400, 0])
        high = np.array([3, 3000, 0])
        self.observation_space = spaces.Box(low, high)

        self.curr_iteration = 0
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.curr_step += 1
        self.logger.info("")
        self.logger.info(f"Iteration #{self.curr_iteration} Step #{self.curr_step}")
        t0_ventilation_speed, dummy1, dummy2 = self.state

        # Execute action on environment (change ventilation fan speed)
        self._execute_action(action)
        
        # Wait for environment to transition to next state
        self._transition_to_next_state()

        # Get reward for new state
        reward = self._get_reward(self.current_ventilation_speed, t0_ventilation_speed)
        self.total_reward += reward

        self.logger.info(f"Reward={reward}, Total reward={self.total_reward}")
        self.step_logger.info(f"{self.curr_iteration},{self.curr_step},{self.current_ventilation_speed + 1},{reward}")

        done = False

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.curr_iteration += 1
        self.curr_step = 0
        self.total_reward = 0.0
        ventilation_speed = random.randint(0 , self.action_space.n - 1)
        self.state = (ventilation_speed, 400, 0)
        return np.array(self.state)

    def render(self, mode='human'):
        ventilation_speed, dummy1, dummy2 = self.state
        print(f"Environment state: Fan speed={ventilation_speed + 1}")

    def _execute_action(self, action):
        self.logger.info(f"Executing action, setting fan speed to {action + 1}")
        self.current_ventilation_speed = action

    def _transition_to_next_state(self):
        self.logger.info ("Waiting for environment to respond to action...")
                
        # Return new environment state
        self.state = (self.current_ventilation_speed, 400, 0)

    def _get_reward(self, current_ventilation_speed, previous_ventilation_speed):
        reward = 1.0

        # Give penalty for energy consumption
        # Need to add outdoor / indoor temperature based calculation, but keep it simple for now
        ventilation_cost = self._get_ventilation_cost(current_ventilation_speed)
        reward = reward - ventilation_cost

        # Give a small penalty for changing ventilation fan speed (but, not for the 1st step in an Episode)
        if self.curr_step > 1:
            if current_ventilation_speed != previous_ventilation_speed:
                reward = reward - 0.1

        return reward
        
    def _get_ventilation_cost(self, ventilation_speed):
        ventilation_speed_cost = [0.0, 0.2, 0.4, 0.8]
        ventilation_cost = ventilation_speed_cost[ventilation_speed]
        return ventilation_cost